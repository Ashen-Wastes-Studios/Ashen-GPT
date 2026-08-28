import torch
import torch.nn as nn
from torch.nn import functional as F
import tiktoken
import mmap
import random
import os
import pickle

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

# Ashen GPT Hyperparameters (Optimized for 8GB VRAM)
block_size = 256
batch_size = 8
max_iters = 500
eval_interval = 100
learning_rate = 3e-4
eval_iters = 50
n_embd = 384
n_layer = 6
n_head = 6
dropout = 0.2
num_experts = 4
top_k = 2

# Initialize BPE Tokenizer (GPT-2 encoding)
enc = tiktoken.get_encoding("gpt2")
vocab_size = enc.n_vocab  # 50257
print(f"Ashen GPT Tokenizer loaded. Vocab size: {vocab_size}")

encode = lambda s: enc.encode(s, allowed_special={"<|endoftext|>"})
decode = lambda l: enc.decode(l)

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
        return torch.tensor(encode("Hello world! Ashen GPT hybrid training test."), dtype=torch.long)

    with open(filename, 'rb') as f:
        with mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mm:
            file_size = len(mm)
            chunk_size = min(file_size, block_size * batch_size * 4)
            max_start = file_size - chunk_size
            start_pos = random.randint(0, max(0, max_start))
            mm.seek(start_pos)
            block = mm.read(chunk_size)
            decoded_text = block.decode('utf-8', errors='ignore').replace('\r', '')
            tokens = encode(decoded_text)
            return torch.tensor(tokens, dtype=torch.long)

def get_batch(split):
    data_chunk = get_random_chunk(split)
    if len(data_chunk) <= block_size:
        data_chunk = torch.tensor(encode("Fallback training sentence for Ashen GPT."), dtype=torch.long)
    
    data_chunk = torch.clamp(data_chunk, min=0, max=vocab_size - 1)
    
    ix = torch.randint(len(data_chunk) - block_size, (batch_size,))
    x = torch.stack([data_chunk[i:i+block_size] for i in ix])
    y = torch.stack([data_chunk[i+1:i+block_size+1] for i in ix])
    return x.to(device), y.to(device)

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
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

class AshenGPTLanguageModel(nn.Module):
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

print("Initializing Ashen GPT Model...")
model = AshenGPTLanguageModel(vocab_size).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            _, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

print("Starting Ashen GPT training loop...")
for iter in range(max_iters):
    print(f"Iteration {iter}/{max_iters}", flush=True)
    if iter % eval_interval == 0:
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.3f}, val loss {losses['val']:.3f}")
        
        model.eval()
        context = torch.tensor([encode("Once upon a time")], dtype=torch.long, device=device)
        generated = decode(model.generate(context, max_new_tokens=100)[0].tolist())
        print(f"\n--- Ashen GPT Sample Gen at Step {iter} ---")
        print(generated)
        print("-" * 40, "\n")
        model.train()

    xb, yb = get_batch('train')
    logits, loss = model.forward(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

with open("ashen_gpt_model.pk1", "wb") as f:
    pickle.dump(model, f)
print("Training complete! Model saved to ashen_gpt_model.pk1")
