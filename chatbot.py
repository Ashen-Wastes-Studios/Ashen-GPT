import torch
import torch.nn as nn
from torch.nn import functional as F
import tiktoken
import pickle
import os

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Device: {device}")

block_size = 256
n_embd = 384
n_layer = 6
n_head = 6
dropout = 0.2
num_experts = 4
top_k = 2

# Initialize BPE Tokenizer (GPT-2 encoding)
enc = tiktoken.get_encoding("gpt2")
vocab_size = enc.n_vocab  # 50257
print(f"BPE Tokenizer loaded. Vocab size: {vocab_size}")

encode = lambda s: enc.encode(s, allowed_special={"<|endoftext|>"})
decode = lambda l: enc.decode(l)

class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)
        wei = q @ k.transpose(-2, -1) * k.shape[-1]**-0.5
        tril = torch.tril(torch.ones(T, T, device=x.device))
        wei = wei.masked_fill(tril == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        return wei @ self.value(x)

class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(head_size * num_heads, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        return self.dropout(self.proj(out))

class Expert(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )
    def forward(self, x):
        return self.net(x)

class MixtureOfExpertsFeedForward(nn.Module):
    def __init__(self, n_embd, num_experts=8, top_k=2):
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
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

class BPEMoELanguageModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, index, targets=None):
        B, T = index.shape
        if T > block_size:
            index = index[:, -block_size:]
            B, T = index.shape

        index = torch.clamp(index, min=0, max=self.token_embedding_table.num_embeddings - 1)
        tok_emb = self.token_embedding_table(index)
        
        pos_indices = torch.arange(T, device=device)
        pos_indices = torch.clamp(pos_indices, min=0, max=self.position_embedding_table.num_embeddings - 1)
        pos_emb = self.position_embedding_table(pos_indices)
        
        x = tok_emb + pos_emb
        x = self.blocks(x)
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

# Load model checkpoint
model_path = 'bpe_moe_model.pk1'
if os.path.exists(model_path):
    print(f"Loading BPE + MoE model parameters from {model_path}...")
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    print("Model loaded successfully!")
else:
    print(f"No checkpoint found at {model_path}. Initializing new BPE + MoE model...")
    model = BPEMoELanguageModel(vocab_size)

m = model.to(device)

class ReasoningEngine:
    def __init__(self, model, decode_fn, encode_fn, device):
        self.model = model
        self.decode = decode_fn
        self.encode = encode_fn
        self.device = device

    @torch.no_grad()
    def solve_with_cot(self, prompt, max_new_tokens=250, num_samples=1):
        self.model.eval()
        cot_prompt = f"Problem: {prompt}\nLet's think step by step:\n<think>\n"
        encoded = self.encode(cot_prompt)
        if len(encoded) > block_size:
            encoded = encoded[-block_size:]
        input_ids = torch.tensor([encoded], dtype=torch.long, device=self.device)
        
        output_ids = self.model.generate(input_ids, max_new_tokens=max_new_tokens, temperature=0.8, top_k=50)
        generated_text = self.decode(output_ids[0].tolist())
        return generated_text

reasoner = ReasoningEngine(m, decode, encode, device)

print("\n--- BPE + MoE Chatbot Ready ---")
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
