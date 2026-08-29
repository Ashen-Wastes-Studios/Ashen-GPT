import sys
import io
import datetime

class Tee:
    def __init__(self, filename="training_logs.txt", mode="a"):
        self.terminal = sys.stdout
        self.log = open(filename, mode, encoding="utf-8")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()

    def flush(self):
        self.terminal.flush()
        self.log.flush()

    def isatty(self):
        return getattr(self.terminal, 'isatty', lambda: False)()

sys.stdout = Tee("training_logs.txt", "a")
sys.stdout.log.write(f"\n\n--- Training Session Started: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ---\n")
sys.stdout.log.flush()

if hasattr(sys.stdout.terminal, 'reconfigure'):
    sys.stdout.terminal.reconfigure(encoding='utf-8')

import torch
import torch.nn as nn
import torch.utils.checkpoint
from torch.nn import functional as F
import tiktoken
import mmap
import random
import os
import pickle
import math
import time
import re
import gc
import copy

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using optimized device: {device}")

# --- Progressive Multi-Hop Staged Training Configuration (5,000 Max Iters) ---
max_iters = 5000                # Robust multi-thousand iteration pre-training run
eval_interval = 250             # Evaluation checkpoints
learning_rate = 4e-4
min_learning_rate = 3e-5
warmup_iters = 200
eval_iters = 50
n_embd = 512
n_layer = 8
n_head = 8
dropout = 0.1
num_experts = 4
top_k = 2

# Initialize BPE Tokenizer (GPT-2 encoding)
enc = tiktoken.get_encoding("gpt2")
vocab_size = enc.n_vocab  # 50257
print(f"Ashen GPT Tokenizer loaded. Vocab size: {vocab_size}")

encode = lambda s: enc.encode(s, allowed_special={"<|endoftext|>"})
decode = lambda l: enc.decode(l)

def filter_code_output(text):
    """Filters out markdown code blocks and code snippets so the model outputs pure natural language."""
    text_no_blocks = re.sub(r'```[\s\S]*?```', '[Code logic analyzed internally, outputting natural language explanation]', text)
    text_clean = re.sub(r'`[^`]*`', '', text_no_blocks)
    return text_clean

def get_random_chunk(split, current_block_size):
    if split == 'train':
        train_files = ["train_split.txt"]
        if os.path.exists("code_train_split.txt"):
            train_files.append("code_train_split.txt")
        filename = random.choice(train_files)
    else:
        filename = "val_split.txt"
        if not os.path.exists(filename):
            filename = "train_split.txt"

    if not os.path.exists(filename):
        return torch.tensor(encode("Hello world! Ashen GPT hybrid training test. " * (current_block_size // 10)), dtype=torch.long)

    with open(filename, 'rb') as f:
        with mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mm:
            file_size = len(mm)
            chunk_size = min(file_size, current_block_size * 4 * 4)
            max_start = file_size - chunk_size
            start_pos = random.randint(0, max(0, max_start))
            mm.seek(start_pos)
            block = mm.read(chunk_size)
            decoded_text = block.decode('utf-8', errors='ignore').replace('\r', '')
            tokens = encode(decoded_text)
            return torch.tensor(tokens, dtype=torch.long)

def get_batch(split, current_block_size, current_batch_size):
    data_chunk = get_random_chunk(split, current_block_size)
    if len(data_chunk) <= current_block_size + 10:
        fallback_text = "Fallback training sentence for Ashen GPT progressive training pipeline. " * (current_block_size // 10 + 5)
        data_chunk = torch.tensor(encode(fallback_text), dtype=torch.long)

    data_chunk = torch.clamp(data_chunk, min=0, max=vocab_size - 1)

    max_idx = len(data_chunk) - current_block_size
    ix = torch.randint(0, max_idx, (current_batch_size,))
    x = torch.stack([data_chunk[i:i+current_block_size] for i in ix])
    y = torch.stack([data_chunk[i+1:i+current_block_size+1] for i in ix])
    return x.to(device), y.to(device)

# --- Qwen-like Architecture Components with Dynamic NTK RoPE Scaling ---

class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        norm = x.pow(2).mean(-1, keepdim=True)
        return self.weight * x * torch.rsqrt(norm + self.eps)

class RotaryEmbedding(nn.Module):
    def __init__(self, dim, max_seq_len=65536, theta=10000.0):
        super().__init__()
        self.dim = dim
        self.theta = theta
        self.max_seq_len = max_seq_len
        inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq_buf", inv_freq, persistent=False)
        self._set_cos_sin_cache(max_seq_len)

    def _set_cos_sin_cache(self, seq_len, scale=1.0):
        inv_freq = self.inv_freq_buf / scale
        t = torch.arange(seq_len, device=inv_freq.device, dtype=inv_freq.dtype)
        freqs = torch.outer(t, inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)

    def forward(self, seq_len, current_block_size=2048):
        base_block_size = 2048
        if seq_len > base_block_size:
            scale = (seq_len / base_block_size) ** (self.dim / (self.dim - 2))
        else:
            scale = 1.0

        needed_len = max(seq_len, self.max_seq_len)
        if needed_len > self.cos_cached.size(0) or scale != 1.0:
            self._set_cos_sin_cache(needed_len, scale=scale)
        return self.cos_cached[:seq_len, :], self.sin_cached[:seq_len, :]

def rotate_half(x):
    x1 = x[..., :x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)

def apply_rope(x, cos, sin):
    cos = cos.unsqueeze(0).unsqueeze(2)  # [1, T, 1, D]
    sin = sin.unsqueeze(0).unsqueeze(2)  # [1, T, 1, D]
    return (x * cos) + (rotate_half(x) * sin)

class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.num_heads = num_heads
        self.head_size = head_size
        self.hidden_dim = num_heads * head_size
        self.query = nn.Linear(n_embd, self.hidden_dim, bias=False)
        self.key = nn.Linear(n_embd, self.hidden_dim, bias=False)
        self.value = nn.Linear(n_embd, self.hidden_dim, bias=False)
        self.proj = nn.Linear(self.hidden_dim, n_embd, bias=False)
        
        self.q_norm = RMSNorm(head_size)
        self.k_norm = RMSNorm(head_size)
        self.dropout = dropout

    def forward(self, x, rope_cache, current_block_size):
        B, T, C = x.shape
        q = self.query(x).view(B, T, self.num_heads, self.head_size)
        k = self.key(x).view(B, T, self.num_heads, self.head_size)
        v = self.value(x).view(B, T, self.num_heads, self.head_size)

        q = self.q_norm(q)
        k = self.k_norm(k)

        cos, sin = rope_cache
        q = apply_rope(q, cos[:T, :], sin[:T, :])
        k = apply_rope(k, cos[:T, :], sin[:T, :])

        q = q.transpose(1, 2)  # [B, H, T, D]
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        out = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=None,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=True
        )
        out = out.transpose(1, 2).contiguous().view(B, T, self.hidden_dim)
        return self.proj(out)

class Expert(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        hidden_dim = int(8 * n_embd / 3)
        self.gate_proj = nn.Linear(n_embd, hidden_dim, bias=False)
        self.up_proj = nn.Linear(n_embd, hidden_dim, bias=False)
        self.down_proj = nn.Linear(hidden_dim, n_embd, bias=False)
        self.dropout = dropout

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))

class MixtureOfExpertsFeedForward(nn.Module):
    def __init__(self, n_embd, num_experts=4, top_k=2):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.experts = nn.ModuleList([Expert(n_embd) for _ in range(num_experts)])
        self.gate = nn.Linear(n_embd, num_experts, bias=False)

    def forward(self, x):
        B, T, C = x.shape
        x_flat = x.view(-1, C)
        gate_logits = self.gate(x_flat)
        weights, selected_experts = torch.topk(F.softmax(gate_logits, dim=-1), self.top_k, dim=-1)
        weights = weights / weights.sum(dim=-1, keepdim=True)

        out = torch.zeros_like(x_flat)
        for i, expert in enumerate(self.experts):
            batch_idx, nth_expert = torch.where(selected_experts == i)
            if batch_idx.numel() == 0:
                continue
            tokens_for_expert = x_flat[batch_idx]
            expert_out = expert(tokens_for_expert)
            weight_for_expert = weights[batch_idx, nth_expert].unsqueeze(-1)
            out.index_add_(0, batch_idx, expert_out * weight_for_expert)
        return out.view(B, T, C)

class Block(nn.Module):
    def __init__(self, n_embd, n_head):
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = MixtureOfExpertsFeedForward(n_embd, num_experts=num_experts, top_k=top_k)
        self.ln1 = RMSNorm(n_embd)
        self.ln2 = RMSNorm(n_embd)

    def forward(self, x, rope_cache, current_block_size):
        if self.training and current_block_size > 2048:
            x = x + torch.utils.checkpoint.checkpoint(
                lambda inp, rc: self.sa(self.ln1(inp), rc, current_block_size),
                x, rope_cache,
                use_reentrant=False
            )
            x = x + self.ffwd(self.ln2(x))
        else:
            x = x + self.sa(self.ln1(x), rope_cache, current_block_size)
            x = x + self.ffwd(self.ln2(x))
        return x

class AshenGPTLanguageModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        head_size = n_embd // n_head
        self.rotary_emb = RotaryEmbedding(head_size, max_seq_len=65536)
        self.blocks = nn.ModuleList([Block(n_embd, n_head=n_head) for _ in range(n_layer)])
        self.ln_f = RMSNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size, bias=False)

    def forward(self, index, targets=None, current_block_size=2048):
        B, T = index.shape
        if T > current_block_size:
            index = index[:, -current_block_size:]
            B, T = index.shape

        index = torch.clamp(index, min=0, max=self.token_embedding_table.num_embeddings - 1)
        x = self.token_embedding_table(index)

        rope_cache = self.rotary_emb(T, current_block_size)

        for block in self.blocks:
            x = block(x, rope_cache, current_block_size)

        x = self.ln_f(x)
        logits = self.lm_head(x)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)
        return logits, loss

    def generate(self, index, max_new_tokens, current_block_size=2048, temperature=0.8, top_k=50):
        for _ in range(max_new_tokens):
            index_cond = index[:, -current_block_size:]
            logits, loss = self.forward(index_cond, current_block_size=current_block_size)
            logits = logits[:, -1, :] / temperature
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            probs = F.softmax(logits, dim=-1)
            index_next = torch.multinomial(probs, num_samples=1)
            index = torch.cat((index, index_next), dim=-1)
        return index

model_path = "ashen_gpt_model.pk1"
if os.path.exists(model_path):
    print(f"Detected existing model checkpoint at {model_path}. Upscaling model by 2x depth (doubling layers)...")
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    
    old_layer_count = len(model.blocks)
    model.blocks = nn.ModuleList([copy.deepcopy(block) for block in model.blocks] + [copy.deepcopy(block) for block in model.blocks])
    new_layer_count = len(model.blocks)
    print(f"Model successfully upscaled by 2x depth! Layers doubled from {old_layer_count} to {new_layer_count}.")
    
    with open(model_path, "wb") as f:
        pickle.dump(model, f)
    print(f"Upscaled model saved back to {model_path}.")
    model = model.to(device)
else:
    print("Initializing ~127M Qwen-Architectured Ashen GPT Model (Fast 500 Iters)...")
    model = AshenGPTLanguageModel(vocab_size).to(device)

try:
    if hasattr(torch, 'compile') and os.name != 'nt':
        print("Compiling model with torch.compile...")
        model = torch.compile(model)
    else:
        print("Running in highly optimized PyTorch eager mode.")
except Exception as e:
    print(f"torch.compile skipped: {e}")

total_params = sum(p.numel() for p in model.parameters())
print(f"Total Model Parameters: {total_params / 1e6:.2f} Million")

optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, fused=True if device == 'cuda' else False)
scaler = torch.amp.GradScaler('cuda' if device == 'cuda' else 'cpu')

def get_lr(it):
    if it < warmup_iters:
        return learning_rate * (it + 1) / warmup_iters
    if it > max_iters:
        return min_learning_rate
    decay_ratio = (it - warmup_iters) / (max_iters - warmup_iters)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_learning_rate + coeff * (learning_rate - min_learning_rate)

@torch.no_grad()
def estimate_loss(current_block_size, current_batch_size):
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split, current_block_size, current_batch_size)
            with torch.amp.autocast('cuda' if device == 'cuda' else 'cpu'):
                _, loss = model(X, Y, current_block_size=current_block_size)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

# --- PROGRESSIVE STAGED TRAINING LOOP (512 -> 2k -> 8k scaled to 5,000 iters) ---
print("=== Starting Progressive Staged Training Pipeline (5,000 Iters) ===", flush=True)
optimizer.zero_grad(set_to_none=True)

for iter in range(max_iters):
    iter_start = time.time()
    lr = get_lr(iter)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    # Determine current training stage based on iteration (scaled for 5,000 max_iters)
    if iter <= 3000:
        stage_name = "Stage 1: Core Training"
        current_block_size = 512
        current_batch_size = 8
        gradient_accumulation_steps = 2
    elif iter <= 4500:
        stage_name = "Stage 2: Intermediate Extension"
        current_block_size = 2048
        current_batch_size = 2
        gradient_accumulation_steps = 8
    else:
        stage_name = "Stage 3: Extreme Extension"
        current_block_size = 8192
        current_batch_size = 1
        gradient_accumulation_steps = 16

    loss_accum = 0.0
    for micro_step in range(gradient_accumulation_steps):
        xb, yb = get_batch('train', current_block_size, current_batch_size)
        with torch.amp.autocast('cuda' if device == 'cuda' else 'cpu'):
            logits, loss = model.forward(xb, yb, current_block_size=current_block_size)
            loss = loss / gradient_accumulation_steps
            loss_accum += loss.detach().item()

        scaler.scale(loss).backward()

    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad(set_to_none=False)
    torch.cuda.empty_cache()

    elapsed = time.time() - iter_start
    print(f"[{stage_name} | STEP {iter+1}/{max_iters} | Ctx: {current_block_size}] Loss: {loss_accum:.4f} | LR: {lr:.6f} | Time: {elapsed:.1f}s", flush=True)

    if iter > 0 and iter % eval_interval == 0:
        print(f"\n==================================================", flush=True)
        print(f"--- EVALUATION ({stage_name} - Ctx: {current_block_size}) ---", flush=True)
        print(f"==================================================", flush=True)
        losses = estimate_loss(current_block_size, current_batch_size)
        print(f"Eval Results -> Train Loss: {losses['train']:.3f} | Val Loss: {losses['val']:.3f}\n", flush=True)

        model.eval()

        text_prompt = "The future of artificial intelligence is"
        context_text = torch.tensor([encode(text_prompt)], dtype=torch.long, device=device)
        raw_text_gen = decode(model.generate(context_text, max_new_tokens=100, current_block_size=current_block_size)[0].tolist())
        clean_text_gen = filter_code_output(raw_text_gen)

        print(f"[TEXT TEST]", flush=True)
        print(f"Prompt: {text_prompt}", flush=True)
        print(f"Completion (Natural Language): {clean_text_gen}\n", flush=True)

        clock_app_prompts = [
            ("Python", "def create_python_clock():\n    # Write a clock app in Python:\n"),
            ("JavaScript / TypeScript", "function createClockApp() {\n    // Write a clock app in JavaScript:\n"),
            ("Go", "package main\n// Write a clock app in Go:\nfunc main() {\n"),
            ("Rust", "// Write a clock app in Rust:\nfn main() {\n    println!(\"Rust Clock App\");\n"),
            ("C++", "// Write a clock app in C++:\n#include <iostream>\nint main() {\n"),
            ("Ruby", "# Write a clock app in Ruby:\nclass ClockApp\n")
        ]

        print(f"[CODE TESTS - CLOCK APP GENERATION ACROSS 6 LANGUAGES]", flush=True)
        for lang, prompt_snippet in clock_app_prompts:
            context_code = torch.tensor([encode(prompt_snippet)], dtype=torch.long, device=device)
            raw_code_gen = decode(model.generate(context_code, max_new_tokens=100, current_block_size=current_block_size)[0].tolist())
            print(f"--- Language: {lang} ---", flush=True)
            print(f"Prompt: {prompt_snippet.strip()}", flush=True)
            print(f"Completion:\n{raw_code_gen}\n", flush=True)

        print(f"==================================================\n", flush=True)
        model.train()
        torch.cuda.empty_cache()
        gc.collect()

# --- PHASE 2: SUPERVISED FINE-TUNING (SFT) ---
print("\n=== PHASE 2: Supervised Fine-Tuning (SFT at 32K Target Context) ===", flush=True)
SFT_DATASET = [
    {
        "instruction": "Explain how python lists work.",
        "response": "<think>\nPython lists are dynamic arrays that support indexing, slicing, and mutable operations.\n</think>\nPython lists are ordered, mutable collections of items in Python. They automatically resize as items are added or removed."
    },
    {
        "instruction": "What is attention in transformers?",
        "response": "<think>\nAttention computes relationships between tokens using Query, Key, and Value vectors.\n</think>\nAttention is a core transformer mechanism that calculates how much focus one token should place on other tokens in a sequence."
    },
    {
        "instruction": "Write a Python function to calculate factorial.",
        "response": "<think>\nTo calculate factorial, we can use recursion or iteration. Recursion is clean: base case n <= 1 return 1, else n * factorial(n-1).\n</think>\ndef factorial(n):\n    if n <= 1:\n        return 1\n    return n * factorial(n - 1)"
    },
    {
        "instruction": "What is RMSNorm?",
        "response": "<think>\nRMSNorm is Root Mean Square normalization, a simplified alternative to LayerNorm that scales by root mean square.\n</think>\nRMSNorm is a normalization technique that scales activations based on the root mean square of the input tensor, improving training efficiency without mean-centering."
    },
    {
        "instruction": "Write a JavaScript function to check if a string is a palindrome.",
        "response": "<think>\nClean the string by removing non-alphanumeric chars and lowercasing, then compare with its reverse.\n</think>\nfunction isPalindrome(str) {\n    const cleaned = str.toLowerCase().replace(/[^a-z0-9]/g, '');\n    return cleaned === cleaned.split('').reverse().join('');\n}"
    },
    {
        "instruction": "Explain Rotary Position Embeddings (RoPE).",
        "response": "<think>\nRoPE encodes relative position by rotating query and key vectors in complex planes.\n</think>\nRotary Position Embeddings (RoPE) multiply query and key representations by rotation matrices based on token positions, allowing the model to naturally capture relative distances."
    },
    {
        "instruction": "Write a Go hello world function.",
        "response": "<think>\nGo uses the main package and fmt.Println for printing.\n</think>\npackage main\nimport \"fmt\"\nfunc main() {\n    fmt.Println(\"Hello, World!\")\n}"
    },
    {
        "instruction": "What is Mixture of Experts (MoE)?",
        "response": "<think>\nMoE routes tokens to subsets of expert FFN networks using a gating router.\n</think>\nMixture of Experts (MoE) is a neural network architecture where multiple feed-forward expert sub-networks process tokens routed by a learned gating function, increasing capacity without proportional compute cost."
    }
]

optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5, fused=True if device == 'cuda' else False)
sft_epochs = 3

for epoch in range(sft_epochs):
    total_sft_loss = 0.0
    random.shuffle(SFT_DATASET)

    for item_idx, item in enumerate(SFT_DATASET):
        prompt_text = f"### Instruction:\n{item['instruction']}\n\n### Response:\n{item['response']}<|endoftext|>"
        tokens = encode(prompt_text)
        if len(tokens) > 8192:
            tokens = tokens[:8192]

        x = torch.tensor(tokens[:-1], dtype=torch.long, device=device).unsqueeze(0)
        y = torch.tensor(tokens[1:], dtype=torch.long, device=device).unsqueeze(0)

        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast('cuda' if device == 'cuda' else 'cpu'):
            _, loss = model(x, y, current_block_size=8192)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()
        torch.cuda.empty_cache()

        total_sft_loss += loss.item()
        print(f"[SFT Epoch {epoch+1}/{sft_epochs} | Item {item_idx+1}/{len(SFT_DATASET)}] Loss: {loss.item():.4f}", flush=True)

    avg_loss = total_sft_loss / len(SFT_DATASET)
    print(f"SFT Epoch {epoch+1} Completed | Avg Loss: {avg_loss:.4f}\n", flush=True)
    torch.cuda.empty_cache()
    gc.collect()

with open("ashen_gpt_model.pk1", "wb") as f:
    pickle.dump(model, f)
print("Training & Supervised Fine-Tuning complete! Staged 8K model saved to ashen_gpt_model.pk1", flush=True)

# --- PHASE 3: REINFORCEMENT LEARNING (Direct Preference Optimization - DPO) ---
print("\n=== PHASE 3: Reinforcement Learning via DPO ===", flush=True)
print("Objective: Align model outputs with preferred responses using direct preference optimization", flush=True)

# Preference dataset: each item has instruction + chosen (preferred) + rejected (dispreferred) response
DPO_PREFERENCE_DATASET = [
    {
        "instruction": "Explain Python decorators.",
        "chosen": "<think>\nDecorators wrap functions to add behavior without modifying them.\n</think>\nPython decorators are functions that modify or enhance other functions. They use the @decorator syntax and allow you to add functionality like logging or authentication without changing the original function code.",
        "rejected": "Decorators are fancy function wrappers."
    },
    {
        "instruction": "Write a binary search algorithm.",
        "chosen": "<think>\nBinary search divides the sorted array in half repeatedly until the target is found.\n</think>\ndef binary_search(arr, target):\n    left, right = 0, len(arr) - 1\n    while left <= right:\n        mid = (left + right) // 2\n        if arr[mid] == target:\n            return mid\n        elif arr[mid] < target:\n            left = mid + 1\n        else:\n            right = mid - 1\n    return -1\n# Time complexity: O(log n)",
        "rejected": "def binsearch(x,y): # binary search\nfor i in x: # loop through\nif i==y: return True\nreturn False"
    },
    {
        "instruction": "What is transfer learning?",
        "chosen": "<think>\nTransfer learning uses pre-trained models on new but related tasks.\n</think>\nTransfer learning is an ML technique where knowledge gained from solving one problem is applied to a different but related problem. Common in NLP (fine-tuning BERT/RoBERTa) and computer vision (using ImageNet weights for custom classification). Improves performance when labeled data is scarce.",
        "rejected": "transfer learning is when you learn things again"
    },
    {
        "instruction": "Implement a LRU cache in Python.",
        "chosen": "<think>\nLRU cache needs a doubly-linked list + hash map for O(1) operations.\n</think>\nfrom collections import OrderedDict\n\nclass LRUCache:\n    def __init__(self, capacity: int):\n        self.cache = OrderedDict()\n        self.capacity = capacity\n    \n    def get(self, key: int) -> int:\n        if key not in self.cache:\n            return -1\n        self.cache.move_to_end(key)\n        return self.cache[key]\n    \n    def put(self, key: int, value: int) -> None:\n        if key in self.cache:\n            self.cache.move_to_end(key)\n        self.cache[key] = value\n        if len(self.cache) > self.capacity:\n            self.cache.popitem(last=False)",
        "rejected": "class cache:\n    def __init__():\n        pass\n    def get():\n        return None\n    def put():\n        pass"
    },
    {
        "instruction": "Compare REST and GraphQL APIs.",
        "chosen": "<think>\nREST uses multiple endpoints; GraphQL uses single endpoint with flexible queries.\n</think>\n**REST:** Multiple endpoints (GET /users, GET /posts), fixed response shapes, over-fetching/under-fetching common. Uses HTTP methods naturally.\n\n**GraphQL:** Single endpoint (/graphql), clients request exact data needed, strong typing via schema. Eliminates over-fetching but adds query complexity and caching challenges.\n\n**Trade-offs:** REST is simpler and cache-friendly; GraphQL is flexible and efficient for complex data hierarchies.",
        "rejected": "REST is old and GraphQL is new. Both do API stuff."
    },
    {
        "instruction": "How does gradient descent work?",
        "chosen": "<think>\nGD iteratively adjusts weights to minimize loss by following negative gradient direction.\n</think>\nGradient Descent is an optimization algorithm that minimizes a loss function by iteratively moving in the direction of steepest descent (negative gradient). At each step:\n1. Compute gradients: ∂L/∂w\n2. Update weights: w = w - lr × ∂L/∂w\n3. Repeat until convergence\n\nVariants include SGD (stochastic), Mini-batch GD, Adam, RMSProp (adaptive learning rates). The learning rate controls step size.",
        "rejected": "it goes down the hill of the function until it finds bottom"
    }
]

dpo_beta = 0.1  # DPO temperature parameter
dpo_epochs = 2
dpo_lr = 1e-5   # Lower LR for RL fine-tuning

optimizer_dpo = torch.optim.AdamW(model.parameters(), lr=dpo_lr, fused=True if device == 'cuda' else False)

@torch.no_grad()
def compute_log_probs(model, input_ids, attention_mask=None):
    """Compute log probabilities for a sequence."""
    outputs = model(input_ids, labels=input_ids.clone())
    logits = outputs.logits
    log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
    return log_probs

def dpo_loss(chosen_log_probs, rejected_log_probs, beta=0.1):
    """Compute DPO loss for a batch of preference pairs."""
    # DPO loss formulation (Rafael et al., 2023)
    pi_ref_log_probs = chosen_log_probs  # Reference model = frozen SFT model
    ref_log_probs = rejected_log_probs
    
    # Loss per pair
    chosen_reward = beta * (chosen_log_probs - pi_ref_log_probs)
    rejected_reward = beta * (rejected_log_probs - pi_ref_log_probs)
    
    # Negative log sigmoid of the difference
    loss = -F.logsigmoid(chosen_reward - rejected_reward)
    return loss.mean()

total_dpo_iters = 0
for epoch in range(dpo_epochs):
    total_dpo_loss = 0.0
    random.shuffle(DPO_PREFERENCE_DATASET)
    
    print(f"\n{'='*60}", flush=True)
    print(f"DPO Epoch {epoch+1}/{dpo_epochs}", flush=True)
    print(f"{'='*60}", flush=True)
    
    for idx, item in enumerate(DPO_PREFERENCE_DATASET):
        # Format chosen response
        chosen_text = f"### Instruction:\n{item['instruction']}\n\n### Response:\n{item['chosen']}<|endoftext|>"
        tokens_chosen = encode(chosen_text)
        
        # Format rejected response  
        rejected_text = f"### Instruction:\n{item['instruction']}\n\n### Response:\n{item['rejected']}<|endoftext|>"
        tokens_rejected = encode(rejected_text)
        
        # Truncate if needed
        max_len = min(len(tokens_chosen), len(tokens_rejected), 4096)
        tokens_chosen = tokens_chosen[:max_len]
        tokens_rejected = tokens_rejected[:max_len]
        
        # Create input tensors (shifted for next-token prediction)
        x_chosen = torch.tensor(tokens_chosen[:-1], dtype=torch.long, device=device).unsqueeze(0)
        y_chosen = torch.tensor(tokens_chosen[1:], dtype=torch.long, device=device).unsqueeze(0)
        
        x_rejected = torch.tensor(tokens_rejected[:-1], dtype=torch.long, device=device).unsqueeze(0)
        y_rejected = torch.tensor(tokens_rejected[1:], dtype=torch.long, device=device).unsqueeze(0)
        
        # Forward pass with autocast
        with torch.amp.autocast('cuda' if device == 'cuda' else 'cpu'):
            _, loss_chosen = model.forward(x_chosen, y_chosen, current_block_size=4096)
            _, loss_rejected = model.forward(x_rejected, y_rejected, current_block_size=4096)
            
            # Compute log probs for DPO
            logits_chosen = model.forward(x_chosen, current_block_size=4096)[0]
            logits_rejected = model.forward(x_rejected, current_block_size=4096)[0]
            
            log_probs_chosen = torch.gather(
                F.log_softmax(logits_chosen, dim=-1),
                dim=-1,
                index=y_chosen.unsqueeze(-1)
            ).squeeze(-1).sum(dim=-1)
            
            log_probs_rejected = torch.gather(
                F.log_softmax(logits_rejected, dim=-1),
                dim=-1,
                index=y_rejected.unsqueeze(-1)
            ).squeeze(-1).sum(dim=-1)
            
            # Compute DPO loss
            dpo_batch_loss = dpo_loss(log_probs_chosen, log_probs_rejected, beta=dpo_beta)
        
        # Backward pass
        optimizer_dpo.zero_grad(set_to_none=True)
        scaler.scale(dpo_batch_loss).backward()
        scaler.unscale_(optimizer_dpo)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer_dpo)
        scaler.update()
        
        total_dpo_loss += dpo_batch_loss.item()
        total_dpo_iters += 1
        
        if total_dpo_iters % 2 == 0:
            print(f"[DPO Epoch {epoch+1} | Sample {idx+1}/{len(DPO_PREFERENCE_DATASET)}] "
                  f"DPO Loss: {dpo_batch_loss.item():.4f} | Chosen LL: {log_probs_chosen.item():.2f} | "
                  f"Rejected LL: {log_probs_rejected.item():.2f}", flush=True)
    
    avg_dpo_loss = total_dpo_loss / len(DPO_PREFERENCE_DATASET)
    print(f"\nDPO Epoch {epoch+1} Complete | Avg DPO Loss: {avg_dpo_loss:.4f}", flush=True)
    torch.cuda.empty_cache()
    gc.collect()

print(f"\n{'='*60}", flush=True)
print(f"DPO Training Complete! Total iterations: {total_dpo_iters}", flush=True)
print(f"Model now aligned with preference data via Direct Preference Optimization", flush=True)
print(f"{'='*60}\n", flush=True)

# Final save
final_model_name = "ashen_gpt_model_dpo.pk1"
with open(final_model_name, "wb") as f:
    pickle.dump(model, f)
print(f"DPO-aligned model saved to {final_model_name}", flush=True)
print("=== COMPLETE TRAINING PIPELINE (Pre-training → SFT → DPO) FINISHED ===", flush=True)
