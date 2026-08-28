import sys
import io
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

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

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using optimized device: {device}")

# --- Peak Safe Limit for 8GB GPU (~450M Parameters) ---
block_size = 256
batch_size = 4
gradient_accumulation_steps = 8
max_iters = 2000
eval_interval = 20
learning_rate = 3e-4
min_learning_rate = 3e-5
warmup_iters = 20
eval_iters = 20
n_embd = 896
n_layer = 16
n_head = 14
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

def get_random_chunk(split):
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
        return torch.tensor(encode("Hello world! Ashen GPT hybrid training test. " * 50), dtype=torch.long)

    with open(filename, 'rb') as f:
        with mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mm:
            file_size = len(mm)
            chunk_size = min(file_size, block_size * batch_size * 16)
            max_start = file_size - chunk_size
            start_pos = random.randint(0, max(0, max_start))
            mm.seek(start_pos)
            block = mm.read(chunk_size)
            decoded_text = block.decode('utf-8', errors='ignore').replace('\r', '')
            tokens = encode(decoded_text)
            return torch.tensor(tokens, dtype=torch.long)

def get_batch(split):
    data_chunk = get_random_chunk(split)
    if len(data_chunk) <= block_size + 10:
        fallback_text = "Fallback training sentence for Ashen GPT large language model training. " * 100
        data_chunk = torch.tensor(encode(fallback_text), dtype=torch.long)

    data_chunk = torch.clamp(data_chunk, min=0, max=vocab_size - 1)

    max_idx = len(data_chunk) - block_size
    ix = torch.randint(0, max_idx, (batch_size,))
    x = torch.stack([data_chunk[i:i+block_size] for i in ix])
    y = torch.stack([data_chunk[i+1:i+block_size+1] for i in ix])
    return x.to(device), y.to(device)

# --- Qwen-like Architecture Components ---

class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        norm = x.pow(2).mean(-1, keepdim=True)
        return self.weight * x * torch.rsqrt(norm + self.eps)

class RotaryEmbedding(nn.Module):
    def __init__(self, dim, max_seq_len=2048, theta=10000.0):
        super().__init__()
        inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._set_cos_sin_cache(max_seq_len)

    def _set_cos_sin_cache(self, seq_len):
        t = torch.arange(seq_len, device=self.inv_freq.device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)

    def forward(self, seq_len):
        if seq_len > self.cos_cached.size(0):
            self._set_cos_sin_cache(seq_len)
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

    def forward(self, x, rope_cache):
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

    def forward(self, x, rope_cache):
        def custom_forward(tensor_x):
            tensor_x = tensor_x + self.sa(self.ln1(tensor_x), rope_cache)
            tensor_x = tensor_x + self.ffwd(self.ln2(tensor_x))
            return tensor_x
        return torch.utils.checkpoint.checkpoint(custom_forward, x, use_reentrant=False)

class AshenGPTLanguageModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        head_size = n_embd // n_head
        self.rotary_emb = RotaryEmbedding(head_size, max_seq_len=block_size * 2)
        self.blocks = nn.ModuleList([Block(n_embd, n_head=n_head) for _ in range(n_layer)])
        self.ln_f = RMSNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size, bias=False)

    def forward(self, index, targets=None):
        B, T = index.shape
        if T > block_size:
            index = index[:, -block_size:]
            B, T = index.shape

        index = torch.clamp(index, min=0, max=self.token_embedding_table.num_embeddings - 1)
        x = self.token_embedding_table(index)

        rope_cache = self.rotary_emb(T)

        for block in self.blocks:
            x = block(x, rope_cache)

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

    def generate(self, index, max_new_tokens, temperature=0.8, top_k=50):
        for _ in range(max_new_tokens):
            index_cond = index[:, -block_size:]
            logits, loss = self.forward(index_cond)
            logits = logits[:, -1, :] / temperature
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            probs = F.softmax(logits, dim=-1)
            index_next = torch.multinomial(probs, num_samples=1)
            index = torch.cat((index, index_next), dim=-1)
        return index

print("Initializing Peak Safe 450M Qwen-Architectured Ashen GPT Model...")
model = AshenGPTLanguageModel(vocab_size).to(device)

total_params = sum(p.numel() for p in model.parameters())
print(f"Total Model Parameters: {total_params / 1e6:.2f} Million")

try:
    import bitsandbytes as bnb
    optimizer = bnb.optim.AdamW8bit(model.parameters(), lr=learning_rate)
    print("Using bitsandbytes 8-bit AdamW optimizer for VRAM savings.")
except Exception:
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    print("Using standard torch AdamW optimizer.")

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
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            with torch.amp.autocast('cuda' if device == 'cuda' else 'cpu'):
                _, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

# --- PHASE 1: PRE-TRAINING ---
print("=== PHASE 1: Pre-training Peak 450M Model (With Gradient Checkpointing) ===", flush=True)
optimizer.zero_grad(set_to_none=True)

for iter in range(max_iters):
    iter_start = time.time()
    lr = get_lr(iter)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    loss_accum = 0.0
    for micro_step in range(gradient_accumulation_steps):
        xb, yb = get_batch('train')
        with torch.amp.autocast('cuda' if device == 'cuda' else 'cpu'):
            logits, loss = model.forward(xb, yb)
            loss = loss / gradient_accumulation_steps
            loss_accum += loss.detach().item()

        scaler.scale(loss).backward()

    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad(set_to_none=False)

    elapsed = time.time() - iter_start
    print(f"[STEP {iter+1}/{max_iters}] Loss: {loss_accum:.4f} | LR: {lr:.6f} | Time: {elapsed:.1f}s", flush=True)

    if iter > 0 and iter % eval_interval == 0:
        print(f"\n==================================================", flush=True)
        print(f"--- EVALUATION & GENERATION TESTS AT STEP {iter} ---", flush=True)
        print(f"==================================================", flush=True)
        losses = estimate_loss()
        print(f"Eval Results -> Train Loss: {losses['train']:.3f} | Val Loss: {losses['val']:.3f}\n", flush=True)

        model.eval()

        # 1. TEXT TEST (Natural Language - Filtered, No Code)
        text_prompt = "The future of artificial intelligence is"
        context_text = torch.tensor([encode(text_prompt)], dtype=torch.long, device=device)
        raw_text_gen = decode(model.generate(context_text, max_new_tokens=80)[0].tolist())
        clean_text_gen = filter_code_output(raw_text_gen)

        print(f"[TEXT TEST]", flush=True)
        print(f"Prompt: {text_prompt}", flush=True)
        print(f"Completion (Natural Language): {clean_text_gen}\n", flush=True)

        # 2. CODE TESTS (Clock App Generation across Python, JavaScript/TypeScript, Go, Rust, C++, Ruby)
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
            raw_code_gen = decode(model.generate(context_code, max_new_tokens=80)[0].tolist())
            print(f"--- Language: {lang} ---", flush=True)
            print(f"Prompt: {prompt_snippet.strip()}", flush=True)
            print(f"Completion:\n{raw_code_gen}\n", flush=True)

        print(f"==================================================\n", flush=True)
        model.train()

# --- PHASE 2: SUPERVISED FINE-TUNING (SFT) ---
print("\n=== PHASE 2: Supervised Fine-Tuning (SFT) ===", flush=True)
SFT_DATASET = [
    {
        "instruction": "Explain how python lists work.",
        "response": "<think>\nPython lists are dynamic arrays that support indexing, slicing, and mutable operations.\n</think>\nPython lists are ordered, mutable collections of items in Python. They automatically resize as items are added or removed."
    },
    {
        "instruction": "What is attention in transformers?",
        "response": "<think>\nAttention computes relationships between tokens using Query, Key, and Value vectors.\n</think>\nAttention is a core transformer mechanism that calculates how much focus one token should place on other tokens in a sequence."
    }
]

optimizer = bnb.optim.AdamW8bit(model.parameters(), lr=5e-5) if 'bnb' in sys.modules else torch.optim.AdamW(model.parameters(), lr=5e-5)
sft_epochs = 5

for epoch in range(sft_epochs):
    total_sft_loss = 0.0
    random.shuffle(SFT_DATASET)

    for item_idx, item in enumerate(SFT_DATASET):
        prompt_text = f"### Instruction:\n{item['instruction']}\n\n### Response:\n{item['response']}<|endoftext|>"
        tokens = encode(prompt_text)
        if len(tokens) > block_size:
            tokens = tokens[:block_size]

        x = torch.tensor(tokens[:-1], dtype=torch.long, device=device).unsqueeze(0)
        y = torch.tensor(tokens[1:], dtype=torch.long, device=device).unsqueeze(0)

        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast('cuda' if device == 'cuda' else 'cpu'):
            _, loss = model(x, y)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()

        total_sft_loss += loss.item()
        print(f"[SFT Epoch {epoch+1}/{sft_epochs} | Item {item_idx+1}/{len(SFT_DATASET)}] Loss: {loss.item():.4f}", flush=True)

    avg_loss = total_sft_loss / len(SFT_DATASET)
    print(f"SFT Epoch {epoch+1} Completed | Avg Loss: {avg_loss:.4f}\n", flush=True)

with open("ashen_gpt_model.pk1", "wb") as f:
    pickle.dump(model, f)
print("Training & Supervised Fine-Tuning complete! Peak 450M Qwen-architectured model saved to ashen_gpt_model.pk1", flush=True)
