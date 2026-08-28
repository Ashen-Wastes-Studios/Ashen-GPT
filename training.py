import torch
import torch.nn as nn
from torch.nn import functional as F
import mmap
import random
import os
import pickle
import argparse

parser = argparse.ArgumentParser(description='This is a demonstration program')

# Here we add an argument to the parser, specifying the expected type, a help message, etc.
parser.add_argument('-batch_size', type=str, required=True, help='Please provide a batch_size')

args = parser.parse_args()

# Now we can use the argument value in our program
print(f"batch_size: {args.batch_size}")
device = 'cuda' if torch.cuda.is_available() else 'cpu'

print(device)

batch_size = 32
block_size = 512
max_iters = 10000
eval_interval = 100
learning_rate = 3e-4
eval_iters = 100
n_embd = 1024
n_layer = 16
n_head = 16
dropout = 0.2
num_experts = 8
top_k = 2

chars = ""
with open('vocab.txt', 'r', encoding='utf-8') as f:
    text = f.read()
    chars = sorted(set(text))

vocab_size = len(chars)

string_to_int = { ch:i for i,ch in enumerate(chars) }
int_to_string = { i:ch for i,ch in enumerate(chars) }
encode = lambda s: [string_to_int.get(c, 0) for c in s]
decode = lambda l: ''.join([int_to_string.get(i, '') for i in l])

def get_random_chunk(split):
    # Pool of training files (literature + code)
    train_files = ["train_split.txt"]
    if os.path.exists("code_train_split.txt"):
        train_files.append("code_train_split.txt")
        
    filename = random.choice(train_files) if split == 'train' else "train_split.txt"
    if not os.path.exists(filename):
        filename = "train_split.txt"
        
    with open(filename, 'rb') as f:
        with mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mm:
            file_size = len(mm)
            max_start = file_size - block_size * batch_size
            if max_start <= 0:
                mm.seek(0)
                block = mm.read()
            else:
                start_pos = random.randint(0, max_start)
                mm.seek(start_pos)
                block = mm.read(block_size * batch_size)

            decoded_block = block.decode('utf-8', errors='ignore').replace('\r', '')
            data_chunk = torch.tensor(encode(decoded_block), dtype=torch.long)
    return data_chunk

def get_batch(split):
    data_chunk = get_random_chunk(split)
    if len(data_chunk) <= block_size:
        # Fallback if chunk is too small
        data_chunk = torch.tensor(encode(text[:10000]), dtype=torch.long)
    
    ix = torch.randint(len(data_chunk) - block_size, (batch_size,))
    x = torch.stack([data_chunk[i:i+block_size] for i in ix])
    y = torch.stack([data_chunk[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y

x, y = get_batch('train')
print('inputs shape:', x.shape)
print('targets shape:', y.shape)

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in {'train', 'val'}:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

class Head(nn.Module):
    """ one head of self-attention """

    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # input of size (batch, time-step, channels)
        # output of size (batch, time-step, head size)
        B, T, C = x.shape
        k = self.key(x) # (B, T, hs)
        q = self.query(x) # (B, T, hs)
        # compute attention scores ("affinities")
        wei = q @ k.transpose(-2, -1) * k.shape[-1]**-0.5 # (B, T, hs) @ (B, hs, T) -> (B, T, T)
        tril = torch.tril(torch.ones(T, T, device=x.device))
        wei = wei.masked_fill(tril == 0, float('-inf')) # (B, T, T)
        wei = F.softmax(wei, dim=-1) # (B, T, T)
        wei = self.dropout(wei)
        # perform the weighted aggregation of the values
        v = self.value(x)
        out = wei @ v # (B, T, T) @ (B, T, hs) -> (B, T, hs)
        return out

class MultiHeadAttention(nn.Module):
    """ multiple heads of self-attention in parallel """

    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(head_size * num_heads, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1) # (B, T, F) -> (B, T, { h1, h1, h1, h1, h2, h2, h2, h2, h3, h3, h3, h3 })
        out = self.dropout(self.proj(out))
        return out

class Expert(nn.Module):
    """ An individual expert network (standard FFN) """
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
    """ Mixture of Experts FeedForward with Top-k Gating """
    def __init__(self, n_embd, num_experts=4, top_k=2):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.experts = nn.ModuleList([Expert(n_embd) for _ in range(num_experts)])
        self.gate = nn.Linear(n_embd, num_experts, bias=False)

    def forward(self, x):
        B, T, C = x.shape
        x_flat = x.view(-1, C)
        
        gate_logits = self.gate(x_flat) # (B*T, num_experts)
        weights, selected_experts = torch.topk(F.softmax(gate_logits, dim=-1), self.top_k, dim=-1) # (B*T, top_k)
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
    """ Transformer block with Mixture of Experts (MoE) """

    def __init__(self, n_embd, n_head):
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        n_exp = globals().get('num_experts', 4)
        t_k = globals().get('top_k', 2)
        self.ffwd = MixtureOfExpertsFeedForward(n_embd, num_experts=n_exp, top_k=t_k)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        y = self.sa(x)
        x = self.ln1(x + y)
        y = self.ffwd(x)
        x = self.ln2(x + y)
        return x

class GPTLanguageModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd) # final layer norm
        self.lm_head = nn.Linear(n_embd, vocab_size)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, index, targets=None):
        B, T = index.shape
        if T > block_size:
            index = index[:, -block_size:]
            B, T = index.shape

        index = torch.clamp(index, min=0, max=self.token_embedding_table.num_embeddings - 1)

        # idx and targets are both (B, T) tensor of integers
        tok_emb = self.token_embedding_table(index) # (B, T, C)
        pos_indices = torch.arange(T, device=device)
        max_pos = self.position_embedding_table.num_embeddings
        pos_indices = torch.clamp(pos_indices, min=0, max=max_pos - 1)
        pos_emb = self.position_embedding_table(pos_indices) # (T, C)
        x = tok_emb + pos_emb # (B, T, C)
        x = self.blocks(x) # (B, T, C)
        x = self.ln_f(x) # (B, T, C)
        logits = self.lm_head(x) # (B, T, vocab_size)
        B, T, C = logits.shape
        
        if targets is None:
            loss = None
        else:
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, index, max_new_tokens):
        # index is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # crop index to the last block_size tokens
            index_cond = index[:, -block_size:]
            # get the predictions
            logits, loss = self.forward(index_cond)
            # focus only on the last time step
            logits = logits[:, -1, :] # becomes (B, C)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1) # (B, C)
            # sample from the distribution
            index_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            # append sampled index to the running sequence
            index = torch.cat((index, index_next), dim=-1) # (B, T+1)
        return index

model = GPTLanguageModel(vocab_size)
print('loading model parameters...')
with open('ashen-gpt-moe-v1.pk1', 'rb') as f:
    model = pickle.load(f)
print('loaded successfully')
m = model.to(device)

# Create a PyTorch optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

for iter in range(max_iters):
    print(iter)

    # every once in a while evaluate the loss on train and val sets
    if iter % eval_interval == 0:
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.3f}, val loss {losses['val']:.3f}")

    # sample a batch of data
    xb, yb = get_batch('train')

    # evaluate the loss
    logits, loss = model.forward(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

print(loss.item())

with open ('ashen-gpt-moe-v1.pk1', 'wb') as f:
    pickle.dump(model, f)
print('model saved')

class ReasoningEngine:
    """ Reasoning engine wrapping GPTLanguageModel with Chain-of-Thought and Self-Consistency """
    def __init__(self, model, decode_fn, encode_fn, device):
        self.model = model
        self.decode = decode_fn
        self.encode = encode_fn
        self.device = device

    @torch.no_grad()
    def solve_with_cot(self, prompt, max_new_tokens=300, num_samples=3):
        """ 
        Chain-of-Thought reasoning with Self-Consistency Best-of-N sampling.
        Guides the model to think step-by-step and selects the best completion.
        """
        self.model.eval()
        cot_prompt = f"Problem: {prompt}\nLet's think step by step:\n<think>\n"
        input_ids = torch.tensor([self.encode(cot_prompt)], dtype=torch.long, device=self.device)
        
        candidates = []
        for _ in range(num_samples):
            output_ids = self.model.generate(input_ids, max_new_tokens=max_new_tokens)
            generated_text = self.decode(output_ids[0].tolist())
            candidates.append(generated_text)
            
        # Select best candidate by length/detail heuristic
        best_candidate = max(candidates, key=len)
        return best_candidate

# Initialize the reasoning engine
reasoner = ReasoningEngine(m, decode, encode, device)