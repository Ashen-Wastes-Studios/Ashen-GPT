import sys
import io
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

import torch
import torch.nn as nn
from torch.nn import functional as F
import tiktoken
import pickle
import os
import re

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Device: {device}")

# --- ~127M Parameter Scale with 8K Context Window for Inference ---
block_size = 8192
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

def user_wants_code(prompt):
    """Detects if the user explicitly asked for code, scripts, functions, or syntax."""
    code_keywords = [
        'code', 'write a', 'function', 'script', 'program', 'python', 'javascript',
        'html', 'css', 'sql', 'syntax', 'class ', 'def ', 'implementation', 'algorithm'
    ]
    prompt_lower = prompt.lower()
    return any(keyword in prompt_lower for keyword in code_keywords)

def filter_code_output(text):
    """Filters out markdown code blocks and code snippets when the user did not ask for code."""
    text_no_blocks = re.sub(r'```[\s\S]*?```', '[Code logic analyzed internally. Ask me to write code if you want to see the implementation snippet.]', text)
    text_clean = re.sub(r'`[^`]*`', '', text_no_blocks)
    return text_clean

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

    def forward(self, seq_len, current_block_size=8192):
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

    def forward(self, index, targets=None, current_block_size=8192):
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

    def generate(self, index, max_new_tokens, current_block_size=8192, temperature=0.8, top_k=50):
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

# Load model checkpoint
model_path = 'ashen_gpt_model.pk1'
if os.path.exists(model_path):
    print(f"Loading Ashen GPT model parameters from {model_path}...")
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    print("Model loaded successfully!")
else:
    print(f"No checkpoint found at {model_path}. Initializing new Ashen GPT model...")
    model = AshenGPTLanguageModel(vocab_size)

m = model.to(device)

class ReasoningEngine:
    def __init__(self, model, decode_fn, encode_fn, device):
        self.model = model
        self.decode = decode_fn
        self.encode = encode_fn
        self.device = device

    @torch.no_grad()
    def solve_with_cot(self, prompt, max_new_tokens=250):
        self.model.eval()
        formatted_prompt = f"### Instruction:\n{prompt}\n\n### Response:\n<think>\n"
        encoded = self.encode(formatted_prompt)
        if len(encoded) > block_size:
            encoded = encoded[-block_size:]
        input_ids = torch.tensor([encoded], dtype=torch.long, device=self.device)

        output_ids = self.model.generate(input_ids, max_new_tokens=max_new_tokens, current_block_size=block_size, temperature=0.7, top_k=40)
        raw_generated = self.decode(output_ids[0].tolist())

        if raw_generated.startswith(formatted_prompt):
            response_text = raw_generated[len(formatted_prompt):]
        else:
            response_text = raw_generated

        response_text = "<think>\n" + response_text

        think_match = re.search(r'<think>([\s\S]*?)(?:</think>|$)', response_text)
        if think_match:
            thought_process = think_match.group(1).strip()
            remainder_start = think_match.end()
            final_answer = response_text[remainder_start:].replace('</think>', '').strip()
        else:
            thought_process = "Analyzing instruction internally..."
            final_answer = response_text

        if user_wants_code(prompt):
            clean_final = final_answer
        else:
            clean_final = filter_code_output(final_answer)

        GREY = "\033[90m"
        RESET = "\033[0m"
        print(f"\n{GREY}<think>\n{thought_process}\n</think>{RESET}")

        return clean_final

reasoner = ReasoningEngine(m, decode, encode, device)

if __name__ == "__main__":
    print("\n--- Ashen GPT Chatbot Ready (~127M Scale & 8K Context) ---")
    while True:
        try:
            prompt = input("\nPrompt:\n> ")
            if not prompt.strip():
                continue
            if prompt.lower() in ['exit', 'quit']:
                break
            reasoned_solution = reasoner.solve_with_cot(prompt, max_new_tokens=200)
            print(f"\nCompletion:\n{reasoned_solution}")
        except (KeyboardInterrupt, EOFError):
            break
