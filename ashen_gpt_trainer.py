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

# --- PERFORMANCE / VRAM CONFIG (scales as the checkpoint grows) ---
# Expandable CUDA memory segments reduce fragmentation so a larger model can
# grow into contiguous memory instead of OOMing on a fragmented heap.
try:
    torch.cuda.memory._set_allocator_settings("expandable_segments:True")
except Exception:
    pass

# Enable the fastest SDPA backend available on this GPU (FlashAttention on Ampere+).
torch.backends.cuda.enable_flash_sdp(True)
torch.backends.cuda.enable_mem_efficient_sdp(True)
# TF32 gives a large matmul speedup on Ampere with negligible quality loss.
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# bf16 is used for the autocast region: faster compute + half the memory of fp16,
# with a wider dynamic range (no grad-scale overflow). This is the single biggest
# "train faster as the model grows" lever.
USE_BF16 = (device == 'cuda' and torch.cuda.is_bf16_supported())
DT_BF16 = torch.bfloat16 if USE_BF16 else torch.float16
print(f"Mixed-precision autocast: {'bf16' if USE_BF16 else 'fp16'}")

# Dynamic-shape compilation: big win on the repeated (SFT/DPO) token loops, and it
# scales with model size because more params => more kernel launches amortized.
ENABLE_COMPILE = True        # now safe on Windows in torch >= 2.12
COMPILE_MODE = 'max-autotune' if (device == 'cpu') else 'reduce-overhead'
# torch.compile default backend; on Windows NVCC-free inductor still works via the
# prebuilt CUDA kernels shipped with the cu132 wheel.
COMPILE_BACKEND = 'inductor'

# Run a few warmup iterations so the first real step isn't paying compile/autograd
# tracing costs (keeps the per-step time flat as the model scales).
COMPILE_WARMUP_STEPS = 2

# Grad-checkpoint the big model: recompute attention/FFN activations instead of
# storing them. Set True whenever activations would dominate VRAM. Decided once
# from total VRAM at startup, then re-evaluated after each upscale.
GRAD_CKPT_LARGE = False
TARGET_VRAM_GB = 7.0         # keep the live model+optimizer safely under this

# --- INTENT CLASSIFICATION HEAD (spam / not_spam / question / answer / request) ---
# A small auxiliary head on top of the shared transformer. It is trained with
# cls_loss alongside the language-model loss so the checkpoint learns to classify
# incoming messages, while the generative path (used by the chatbot) is untouched.
# Both the trainer and web_chatbot.py define AshenGPTLanguageModel with this head
# so the pickled checkpoint round-trips and the chatbot can call classify().
CLASS_LABELS = ["spam", "not_spam", "question", "answer", "request"]
NUM_CLASSES = len(CLASS_LABELS)
CLASS_INDEX = {c: i for i, c in enumerate(CLASS_LABELS)}
CLA_WEIGHT = 0.3             # weight of cls_loss inside the combined training loss
CLA_EVAL_SAMPLES = 25        # held-out examples per eval to estimate classify_acc

# Balanced classification corpus for the auxiliary head. Defined up here (independent
# of SFT_DATASET ordering) so the MAIN loop can train the head and the eval step can
# report classify_acc. Each item is a RAW user message (how the chatbot will call
# classify()), not a "Classify:" prompt — the head learns to label real inputs.
CLASSIFY_TRAIN = [
    {"text": "Buy now!!! Limited offer, 90% OFF only today!!! Click here to claim your prize.", "label": "spam"},
    {"text": "Congratulations! You've won a $1000 Amazon gift card. Reply with your bank details to receive it.", "label": "spam"},
    {"text": "URGENT: Your account will be closed. Verify your password at this.link-secure.xyz immediately.", "label": "spam"},
    {"text": "Make $5000 a week working from home! No experience needed, sign up now!!!", "label": "spam"},
    {"text": "You are the 1,000,000th visitor! Claim your free iPhone by clicking this link.", "label": "spam"},
    {"text": "Final notice: your package is undeliverable. Pay customs fee here or lose it.", "label": "spam"},
    {"text": "Hot singles in your area want to chat! Meet them tonight.", "label": "spam"},
    {"text": "Your PayPal account has been limited. Confirm your login at paypa1-secure.ru", "label": "spam"},
    {"text": "Act now! This crypto will 100x — send BTC to join the presale.", "label": "spam"},
    {"text": "Re: Your loan is approved! Click to accept funds (small processing fee required).", "label": "spam"},
    {"text": "Last chance! Your warranty expires tonight. Renew at this link.", "label": "spam"},
    {"text": "You've been selected for a free cruise — just pay $99 shipping.", "label": "spam"},
    {"text": "Re: Invoice #4471 attached. Please review and process payment by Friday.", "label": "not_spam"},
    {"text": "The quarterly report is attached for your review.", "label": "not_spam"},
    {"text": "Your order #88213 has shipped and will arrive Tuesday.", "label": "not_spam"},
    {"text": "Mom: Don't forget we're having dinner at 7.", "label": "not_spam"},
    {"text": "Reminder: dentist appointment tomorrow at 3pm.", "label": "not_spam"},
    {"text": "Hey, are we still on for lunch tomorrow?", "label": "not_spam"},
    {"text": "The meeting notes from today are in the shared drive.", "label": "not_spam"},
    {"text": "Thanks for sending the files — they look great.", "label": "not_spam"},
    {"text": "Your subscription renews on the 1st; no action needed.", "label": "not_spam"},
    {"text": "Can you confirm the venue for the offsite?", "label": "not_spam"},
    {"text": "The code review is done; please take a look when free.", "label": "not_spam"},
    {"text": "Happy birthday! Hope you have a great day.", "label": "not_spam"},
    {"text": "What is the capital of France?", "label": "question"},
    {"text": "How does a combustion engine work?", "label": "question"},
    {"text": "Why is the sky blue?", "label": "question"},
    {"text": "Who wrote Pride and Prejudice?", "label": "question"},
    {"text": "Can you explain the difference between TCP and UDP?", "label": "question"},
    {"text": "What time does the store close?", "label": "question"},
    {"text": "Where is the nearest gas station?", "label": "question"},
    {"text": "When is the project due?", "label": "question"},
    {"text": "How do I reset my password?", "label": "question"},
    {"text": "What's the weather like tomorrow?", "label": "question"},
    {"text": "Why does my laptop overheat?", "label": "question"},
    {"text": "Which framework is best for this task?", "label": "question"},
    {"text": "The capital of France is Paris.", "label": "answer"},
    {"text": "A combustion engine burns fuel to produce motion through controlled explosions.", "label": "answer"},
    {"text": "The sky appears blue because air molecules scatter shorter (blue) wavelengths of sunlight more.", "label": "answer"},
    {"text": "Pride and Prejudice was written by Jane Austen.", "label": "answer"},
    {"text": "TCP is connection-oriented and reliable; UDP is connectionless and faster.", "label": "answer"},
    {"text": "The store closes at 9pm.", "label": "answer"},
    {"text": "The nearest gas station is two blocks east on Main.", "label": "answer"},
    {"text": "The project is due next Friday.", "label": "answer"},
    {"text": "You reset your password from the account settings page.", "label": "answer"},
    {"text": "Tomorrow will be sunny, high of 72F.", "label": "answer"},
    {"text": "Your laptop overheats because the fan is clogged with dust.", "label": "answer"},
    {"text": "For this task, scikit-learn is the simplest good choice.", "label": "answer"},
    {"text": "Remind me to call Mom at 5pm.", "label": "request"},
    {"text": "Schedule a meeting with the team for Thursday at 2pm.", "label": "request"},
    {"text": "Please send the report to the client.", "label": "request"},
    {"text": "Add 'buy milk' to my shopping list.", "label": "request"},
    {"text": "Set a timer for 20 minutes.", "label": "request"},
    {"text": "Book a table for two at 7pm.", "label": "request"},
    {"text": "Forward this email to Sarah.", "label": "request"},
    {"text": "Turn off the lights in the kitchen.", "label": "request"},
    {"text": "Order a taxi for the airport.", "label": "request"},
    {"text": "Save this document to the shared folder.", "label": "request"},
    {"text": "Mute my phone for an hour.", "label": "request"},
    {"text": "Print the agenda for the meeting.", "label": "request"},
]
CLASSIFY_EVAL = [
    {"text": "Claim your free Netflix account now — just enter your login.", "label": "spam"},
    {"text": "You have 1 unclaimed reward. Tap to collect before it expires!", "label": "spam"},
    {"text": "Investor alert: double your money in 24 hours, guaranteed.", "label": "spam"},
    {"text": "Your car warranty can be extended cheaply — call this number.", "label": "spam"},
    {"text": "Win a free holiday! Click to enter our exclusive draw.", "label": "spam"},
    {"text": "The design mockups are ready for feedback.", "label": "not_spam"},
    {"text": "Reminder: standup at 9:30.", "label": "not_spam"},
    {"text": "Your ticket #5521 was resolved.", "label": "not_spam"},
    {"text": "Dad: pick up bread on the way home.", "label": "not_spam"},
    {"text": "Please find the attached contract for signature.", "label": "not_spam"},
    {"text": "What causes earthquakes?", "label": "question"},
    {"text": "How do I merge a git branch?", "label": "question"},
    {"text": "Who is the current prime minister?", "label": "question"},
    {"text": "When does daylight saving start?", "label": "question"},
    {"text": "Why is the ocean salty?", "label": "question"},
    {"text": "Earthquakes are caused by tectonic plate movement.", "label": "answer"},
    {"text": "You merge a git branch with git merge <branch>.", "label": "answer"},
    {"text": "The current prime minister is the head of government.", "label": "answer"},
    {"text": "Daylight saving starts in late March in most regions.", "label": "answer"},
    {"text": "The ocean is salty because rivers carry dissolved minerals to it.", "label": "answer"},
    {"text": "Remind me to take my medicine at 8pm.", "label": "request"},
    {"text": "Send the invoice to accounting.", "label": "request"},
    {"text": "Add 'call plumber' to my tasks.", "label": "request"},
    {"text": "Set an alarm for 6am.", "label": "request"},
    {"text": "Book a flight to New York.", "label": "request"},
]


# --- Progressive Multi-Hop Staged Training Configuration (5,000 Max Iters - Optimized) ---
max_iters = 5000                # Optimized for faster training while maintaining quality
eval_interval = 500             # Evaluation checkpoints (less frequent = faster)
learning_rate = 4e-4
min_learning_rate = 3e-5
warmup_iters = 150              # Shorter warmup for faster convergence
eval_iters = 20                 # Fewer eval samples = faster evaluation
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

# --- MODEL UPSCALING (exact ~2x parameters, width-only) -------------------------
# Doubles parameters by widening n_embd by sqrt(2) (rounded to a multiple of
# n_head). Because per-block params scale as n_embd^2, a sqrt(2) width increase
# gives exactly 2x params with the SAME depth (width scaling is the "better"
# upscale: it grows representational capacity per layer rather than just stacking
# more of the same). New weights use Net2Net-style copy-initialization: the old
# learned block is pasted into the top-left of the wider matrix and the fresh
# rows/cols are zeroed, so the function the checkpoint already learned is
# preserved on day one (minimal post-upscale loss spike).
def _widen_linear(old_mod, new_in, new_out):
    """Rectangular Net2Net widen: build Linear(new_in, new_out) and copy the old
    old_in x old_out weight block into its top-left; the rest is zeroed. Used for
    both square (attention) and rectangular (MoE expert) projections."""
    old_out, old_in = old_mod.weight.shape  # nn.Linear stores [out, in]
    new_mod = nn.Linear(new_in, new_out, bias=False)
    with torch.no_grad():
        new_mod.weight.zero_()
        new_mod.weight[:old_out, :old_in] = old_mod.weight
    return new_mod

def upscale_model_2x(model, old_embd, n_head, num_experts, top_k):
    """Return a NEW model whose parameter count is ~exactly 2x the input.

    Width-only scaling: n_embd -> new_embd = round(old * sqrt(2)) aligned to
    n_head. Param count per block scales with n_embd^2, so sqrt(2) width =>
    exactly 2x params; depth (n_layer) is unchanged.
    """
    new_embd = round(old_embd * math.sqrt(2))
    # Align to 2*n_head so head_size stays EVEN. RotaryEmbedding emits
    # 2*(dim//2) features, so an odd head dim would break apply_rope; this also
    # keeps head_size divisible by n_head. Production 512 -> 720 (head_size 90)
    # is unchanged; small widths just round up a notch.
    new_embd = (new_embd // (2 * n_head)) * (2 * n_head)
    if new_embd <= old_embd:
        new_embd = old_embd + 2 * n_head
    print(f"[Upscale] width {old_embd} -> {new_embd} (sqrt(2), width-only => ~2x params); "
          f"depth {len(model.blocks)} (unchanged)")
    m = AshenGPTLanguageModel(vocab_size, n_embd=new_embd, n_head=n_head,
                              num_experts=num_experts, top_k=top_k)
    m.token_embedding_table = nn.Embedding(vocab_size, new_embd)
    with torch.no_grad():
        m.token_embedding_table.weight[:, :old_embd] = model.token_embedding_table.weight
        m.lm_head.weight[:, :old_embd] = model.lm_head.weight
    # Classification head carries through the upscale: not widened (tiny), but any
    # already-trained weights are copied so a checkpoint that learned intent labels
    # keeps them when the backbone doubles in size.
    m.class_head = nn.Linear(new_embd, NUM_CLASSES, bias=False)
    if hasattr(model, 'class_head') and model.class_head is not None:
        with torch.no_grad():
            m.class_head.weight[:, :old_embd] = model.class_head.weight
    m.rotary_emb = RotaryEmbedding(new_embd // n_head, max_seq_len=65536)
    new_blocks = []
    for b in model.blocks:
        nb = Block(new_embd, n_head=int(n_head))
        with torch.no_grad():
            nb.sa.query = _widen_linear(b.sa.query, new_embd, new_embd)
            nb.sa.key   = _widen_linear(b.sa.key,   new_embd, new_embd)
            nb.sa.value = _widen_linear(b.sa.value, new_embd, new_embd)
            nb.sa.proj  = _widen_linear(b.sa.proj,  new_embd, new_embd)
            nb.sa.q_norm = _widen_rmsnorm(b.sa.q_norm, new_embd // n_head)
            nb.sa.k_norm = _widen_rmsnorm(b.sa.k_norm, new_embd // n_head)
            nb.ffwd = _widen_moe(b.ffwd, old_embd, new_embd, num_experts, top_k)
            nb.ln1 = _widen_rmsnorm(b.ln1, new_embd)
            nb.ln2 = _widen_rmsnorm(b.ln2, new_embd)
        new_blocks.append(nb)
    m.blocks = nn.ModuleList(new_blocks)
    return m

def _widen_rmsnorm(old_mod, new_embd):
    """Widen an RMSNorm from its current dim to new_embd. The source dim is read
    from the weight itself so this works for both full-width norms and the
    per-head q/k norms (dim = head_size)."""
    old_embd = old_mod.weight.shape[0]
    new_mod = RMSNorm(new_embd)
    with torch.no_grad():
        new_mod.weight[:old_embd] = old_mod.weight
        new_mod.weight[old_embd:] = 1.0
    return new_mod

def _widen_moe(old_ff, old_embd, new_embd, num_experts, top_k):
    new_ff = MixtureOfExpertsFeedForward(new_embd, num_experts=num_experts, top_k=top_k)
    with torch.no_grad():
        for i, ex in enumerate(old_ff.experts):
            o = old_ff.experts[i]
            n = new_ff.experts[i]
            # Derive each new matrix's (in, out) from its weight shape so the
            # gate/up (Linear(embd, hidden)) and down (Linear(hidden, embd))
            # projections are widened with the correct input/output dims.
            n.gate_proj = _widen_linear(o.gate_proj, n.gate_proj.weight.shape[1], n.gate_proj.weight.shape[0])
            n.up_proj   = _widen_linear(o.up_proj,   n.up_proj.weight.shape[1],   n.up_proj.weight.shape[0])
            n.down_proj = _widen_linear(o.down_proj, n.down_proj.weight.shape[1], n.down_proj.weight.shape[0])
        new_ff.gate = _widen_linear(old_ff.gate, new_embd, num_experts)
    return new_ff

def count_parameters(model):
    return sum(p.numel() for p in model.parameters())

# --- VRAM-AWARE PLANNING (keep the model inside GPU memory as it scales) -------
def get_free_vram_gb():
    if device != 'cuda':
        return 16.0  # pretend-plenty on CPU
    return torch.cuda.mem_get_info()[0] / 1e9

def plan_config(params_m, free_gb):
    """Pick batch/ctx/accumulation so the model trains fast AND fits in VRAM.
    As the model grows (params_m up), we shrink the live footprint so per-step
    time stays bounded instead of OOMing or thrashing the allocator."""
    # Rough peak VRAM (GB) heuristic: params*K (fwd+grads+optim) + batch*ctx*C*layers*act.
    # We solve for a batch size that keeps total under TARGET_VRAM_GB.
    # Empirically ~ per-param bytes: fp32 master(4) + bf16 grad(2) + Adam(8) ~ 14B.
    per_param_gb = 14e-9
    base_gb = params_m * 1e6 * per_param_gb
    budget = max(1.0, TARGET_VRAM_GB - base_gb)
    # activation cost proxy at batch=b, ctx=512, ~ 0.0009 GB per (b * ctx/512) per 100M params
    act_per_unit = 0.0009 * (params_m / 100.0)
    b = max(1, int(budget / max(0.05, act_per_unit)))
    b = min(b, 16)
    ctx = 512 if params_m <= 250 else (512 if b >= 4 else 512)
    # If we are very tight, reduce ctx to 256 to stay safe on an 8GB card.
    if budget < 1.5:
        ctx = 256
        b = max(1, b // 2)
    if free_gb < 1.0:           # almost out of room -> minimal
        ctx = 256
        b = 1
    accum = max(1, 16 // b)
    return dict(current_batch_size=b, current_block_size=ctx,
                gradient_accumulation_steps=accum, stage_name="VRAM-adaptive")

def safe_ctx(key, default_ctx):
    """Return a context length that fits the current free VRAM for SFT/DPO runs."""
    free = get_free_vram_gb()
    if free < 2.0:
        return min(default_ctx, 1024)
    if free < 4.0:
        return min(default_ctx, 2048)
    return default_ctx

def unwrap_compiled(model):
    """torch.compile wraps the module; pickle the real nn.Module, not the wrapper."""
    return model._orig_mod if hasattr(model, '_orig_mod') else model

# --- LIGHTWEIGHT DISK-STREAMED DATA SAMPLING (PREVENTS SYSTEM RAM BLOAT) ---

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
        fallback_text = "Fallback training sentence for Ashen GPT progressive training pipeline. " * (current_block_size // 10 + 5)
        return torch.tensor(encode(fallback_text), dtype=torch.long)

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

    max_idx = len(data_chunk) - current_block_size - 1
    ix = torch.randint(0, max_idx, (current_batch_size,))
    x = torch.stack([data_chunk[i:i+current_block_size] for i in ix])
    y = torch.stack([data_chunk[i+1:i+current_block_size+1] for i in ix])
    return x.to(device, non_blocking=True), y.to(device, non_blocking=True)

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
        self.cached_scale = None
        self.cached_len = 0
        self._set_cos_sin_cache(max_seq_len)

    def _set_cos_sin_cache(self, seq_len, scale=1.0):
        inv_freq = self.inv_freq_buf / scale
        t = torch.arange(seq_len, device=inv_freq.device, dtype=inv_freq.dtype)
        freqs = torch.outer(t, inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)
        self.cached_scale = scale
        self.cached_len = seq_len

    def forward(self, seq_len, current_block_size=2048):
        base_block_size = 2048
        if seq_len > base_block_size:
            scale = (seq_len / base_block_size) ** (self.dim / (self.dim - 2))
        else:
            scale = 1.0

        needed_len = max(seq_len, self.max_seq_len)
        if needed_len > self.cached_len or self.cached_scale != scale:
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
        self.query = nn.Linear(self.hidden_dim, self.hidden_dim, bias=False)
        self.key = nn.Linear(self.hidden_dim, self.hidden_dim, bias=False)
        self.value = nn.Linear(self.hidden_dim, self.hidden_dim, bias=False)
        self.proj = nn.Linear(self.hidden_dim, self.hidden_dim, bias=False)
        
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
        # Gradient-checkpoint when the model is large OR the context is long:
        # recompute the self-attention activations instead of storing them,
        # trading ~30% extra compute for a large VRAM saving so the bigger
        # checkpoint stays resident on the GPU as it scales up.
        if self.training and (GRAD_CKPT_LARGE or current_block_size > 2048):
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
    def __init__(self, vocab_size, n_embd=n_embd, n_head=n_head, num_experts=num_experts, top_k=top_k):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        head_size = n_embd // n_head
        self.rotary_emb = RotaryEmbedding(head_size, max_seq_len=65536)
        self.blocks = nn.ModuleList([Block(n_embd, n_head=n_head) for _ in range(n_layer)])
        self.ln_f = RMSNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size, bias=False)
        # Auxiliary intent-classification head (mean-pooled final hidden state).
        self.class_head = nn.Linear(n_embd, NUM_CLASSES, bias=False)

    def forward(self, index, targets=None, current_block_size=2048, cls_targets=None):
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

        # Auxiliary classification head (mean-pooled over the sequence). Trained
        # only when cls_targets is supplied; at inference it is used via classify().
        cls_logits = self.class_head(x.mean(dim=1))
        cls_loss = None
        if cls_targets is not None:
            cls_loss = F.cross_entropy(cls_logits, cls_targets)
        return logits, loss, cls_logits, cls_loss

    @torch.no_grad()
    def classify(self, text, current_block_size=2048):
        """Return (label_string, index, confidence) for an incoming message."""
        ids = torch.tensor([encode(text)], dtype=torch.long, device=device)
        self.eval()
        cls_logits, _ = self.forward(ids, current_block_size=current_block_size)[2], None
        probs = F.softmax(cls_logits, dim=-1)
        idx = int(probs.argmax(dim=-1).item())
        return CLASS_LABELS[idx], idx, float(probs[0, idx].item())

    def generate(self, index, max_new_tokens, current_block_size=2048, temperature=0.8, top_k=50):
        for _ in range(max_new_tokens):
            index_cond = index[:, -current_block_size:]
            logits, _loss, _cls, _ = self.forward(index_cond, current_block_size=current_block_size)
            logits = logits[:, -1, :] / temperature
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            probs = F.softmax(logits, dim=-1)
            index_next = torch.multinomial(probs, num_samples=1)
            index = torch.cat((index, index_next), dim=-1)
        return index

# --- Checkpoint registry -------------------------------------------------
# Ashen GPT ships in TWO formats: the Qwen3.5 HF dir (ashen_gpt_model/) and the
# legacy custom pickle (ashen_gpt_model.pk1). The custom trainer only ingests
# .pk1; the Qwen dir is a DIFFERENT architecture (vocab 248320, different layers)
# and is trained by qwen_finetune.py, which resumes from / saves to the dir.
CHECKPOINT_DIR = "ashen_gpt_model"
def _is_hf_model_dir(path):
    if not os.path.isdir(path) or not os.path.isfile(os.path.join(path, "config.json")):
        return False
    return any(f.endswith(".safetensors") or f == "pytorch_model.bin"
               for f in os.listdir(path))
if _is_hf_model_dir(CHECKPOINT_DIR):
    import subprocess as _sp, sys as _sys
    print(f"Registered Ashen model checkpoint (Qwen HF dir): {CHECKPOINT_DIR}/")
    print("  -> Different architecture from the custom trainer; delegating to qwen_finetune.py.")
    if not os.path.exists("qwen_finetune.py"):
        print("ERROR: qwen_finetune.py not found; cannot train the Qwen checkpoint.")
        _sys.exit(1)
    print("Delegating training to qwen_finetune.py (resumes + saves ashen_gpt_model/)...")
    _res = _sp.run([_sys.executable, "qwen_finetune.py"])
    print(f"qwen_finetune.py finished (exit code {_res.returncode}).")
    _sys.exit(_res.returncode)

model_path = "ashen_gpt_model.pk1"
if os.path.exists(model_path):
    print(f"Detected existing model checkpoint at {model_path}. Loading then upscaling 2x (exact param doubling)...")
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    old_params = count_parameters(model)
    # Back up the ORIGINAL checkpoint before mutating it, so a bad upscale is
    # recoverable and the 8GB+ file is never clobbered in place.
    backup_path = model_path + ".pre_upscale.bak"
    if not os.path.exists(backup_path):
        import shutil
        shutil.copy2(model_path, backup_path)
        print(f"Backup of original checkpoint written to {backup_path}")
    # Exact 2x: width (n_embd) + depth (blocks). Net2Net init reuses learned
    # weights so the loss does not spike after upscaling.
    model = upscale_model_2x(model, n_embd, n_head, num_experts, top_k)
    new_params = count_parameters(model)
    print(f"Model upscaled: {old_params/1e6:.1f}M -> {new_params/1e6:.1f}M params "
          f"({new_params/old_params:.3f}x)")

    # Persist the upscaled checkpoint (overwrites the working path; original is
    # safely in the .pre_upscale.bak backup).
    with open(model_path, "wb") as f:
        pickle.dump(unwrap_compiled(model), f)
    print(f"Upscaled model saved back to {model_path}.")
    model = model.to(device)
else:
    print("Initializing ~127M Ashen GPT Model (Qwen-like architecture)...")
    model = AshenGPTLanguageModel(vocab_size).to(device)

# Adaptive gradient checkpointing: recompute activations for the big model so it
# fits in VRAM as it scales. Decided from the post-upscale size on this GPU.
GRAD_CKPT_LARGE = (TARGET_VRAM_GB - count_parameters(model) * 14e-9) < 1.5
print(f"Gradient checkpointing (large model): {GRAD_CKPT_LARGE}")


import importlib.util as _ilu
_TRITON_AVAILABLE = _ilu.find_spec('triton') is not None
_MODEL_COMPILED = False
try:
    if ENABLE_COMPILE and hasattr(torch, 'compile'):
        # The inductor backend only fails when the compiled graph first EXECUTES
        # (it needs triton at that point, not at compile time), so torch.compile
        # itself succeeds even without triton and the error surfaces later in the
        # warmup. Skip inductor compilation when triton is missing so training
        # falls back to eager mode instead of crashing on the first compiled step.
        if COMPILE_BACKEND == 'inductor' and not _TRITON_AVAILABLE:
            print("torch.compile backend='inductor' needs triton (not installed) -> running eager mode.")
        else:
            print(f"Compiling model with torch.compile (mode={COMPILE_MODE}, backend={COMPILE_BACKEND})...")
            model = torch.compile(model, mode=COMPILE_MODE, backend=COMPILE_BACKEND)
            _MODEL_COMPILED = True
    else:
        print("Running in optimized PyTorch eager mode (compile disabled).")
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
            with torch.amp.autocast('cuda' if device == 'cuda' else 'cpu', **({} if device != 'cuda' else {'dtype': DT_BF16})):
                _, loss, _, _ = model(X, Y, current_block_size=current_block_size)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

@torch.no_grad()
def eval_classify(n=CLA_EVAL_SAMPLES):
    """Held-out accuracy of the auxiliary intent-classification head."""
    correct = total = 0
    model.eval()
    for e in random.sample(CLASSIFY_EVAL, min(n, len(CLASSIFY_EVAL))):
        _, idx, _ = model.classify(e["text"], current_block_size=2048)
        total += 1
        if CLASS_LABELS[idx] == e["label"]:
            correct += 1
    model.train()
    return (correct / total) if total else 0.0

def _sample_cls_batch(n, ctx):
    """Left-padded batch of raw messages + class indices for head training."""
    if n <= 0 or not CLASSIFY_TRAIN:
        return None, None
    ex = random.sample(CLASSIFY_TRAIN, min(n, len(CLASSIFY_TRAIN)))
    seqs = [encode(e["text"])[:ctx] for e in ex]
    maxlen = max(len(s) for s in seqs)
    # Left-pad so the LAST token is always real content (head pools x[:, -1, :]).
    padded = [[0] * (maxlen - len(s)) + s for s in seqs]
    x = torch.tensor(padded, dtype=torch.long, device=device)
    y = torch.tensor([CLASS_INDEX[e["label"]] for e in ex], dtype=torch.long, device=device)
    return x, y

# --- PROGRESSIVE STAGED TRAINING LOOP (VRAM-adaptive) ---
# Plan batch/ctx/accumulation from the live model size + free VRAM so training
# stays fast as the checkpoint scales up (smaller batch + more accumulation when
# the model gets big, keeping the whole model+optimizer inside GPU memory).
_model_params_m = count_parameters(model) / 1e6
_plan = plan_config(_model_params_m, get_free_vram_gb())
print(f"VRAM-adaptive training plan: {_plan}", flush=True)

# Warm up torch.compile / autograd tracing before the timed loop so the first
# real steps aren't paying one-time compile costs (per-step time stays flat as
# the model scales).
# Only warm up when the model was actually compiled (inductor needs triton; if
# it's missing we run eager and warmup is a pointless no-op that would otherwise
# re-trigger the missing-triton crash).
if ENABLE_COMPILE and _MODEL_COMPILED:
    print("Warming up torch.compile...", flush=True)
    for _w in range(COMPILE_WARMUP_STEPS):
        _xb, _yb = get_batch('train', _plan['current_block_size'], _plan['current_batch_size'])
        with torch.amp.autocast('cuda' if device == 'cuda' else 'cpu', **({} if device != 'cuda' else {'dtype': DT_BF16})):
            _, _l, _, _ = model(_xb, _yb, current_block_size=_plan['current_block_size'])
        _l.backward()
        optimizer.zero_grad(set_to_none=True)
    torch.cuda.empty_cache()
    gc.collect()
elif ENABLE_COMPILE:
    print("Running eager (no torch.compile warmup).", flush=True)

print("=== Starting Progressive Staged Training Pipeline (5,000 Iters) ===", flush=True)

for iter in range(max_iters):
    iter_start = time.time()
    lr = get_lr(iter)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    # VRAM-adaptive schedule: derived once from the (post-upscale) model size and
    # live free memory, then held constant so each step trains fast while the
    # whole model + optimizer + activations stay under TARGET_VRAM_GB. As the
    # checkpoint grows, plan_config automatically shrinks batch/ctx and raises
    # accumulation to keep throughput high without OOMing.
    stage_name = _plan['stage_name']
    current_block_size = _plan['current_block_size']
    current_batch_size = _plan['current_batch_size']
    gradient_accumulation_steps = _plan['gradient_accumulation_steps']
    optimizer.zero_grad(set_to_none=True)
    loss_accum = 0.0
    cls_batch_n = min(8, current_batch_size)
    for micro_step in range(gradient_accumulation_steps):
        xb, yb = get_batch('train', current_block_size, current_batch_size)
        with torch.amp.autocast('cuda' if device == 'cuda' else 'cpu', **({} if device != 'cuda' else {'dtype': DT_BF16})):
            logits, lm_loss, _, _ = model(xb, yb, current_block_size=current_block_size)
            lm_loss = lm_loss / gradient_accumulation_steps
            loss_accum += lm_loss.detach().item()
            # Auxiliary intent-classification head (spam / not_spam / question /
            # answer / request). Trained every step on a small balanced batch so
            # the checkpoint learns to classify incoming messages (used by the
            # chatbot for spam/request routing) without disturbing the LM loss.
            cls_x, cls_y = _sample_cls_batch(cls_batch_n, current_block_size)
            if cls_x is not None:
                _, _, cls_logits, cls_loss = model(cls_x, cls_targets=cls_y, current_block_size=current_block_size)
                total_loss = lm_loss + CLA_WEIGHT * (cls_loss / gradient_accumulation_steps)
            else:
                total_loss = lm_loss

        scaler.scale(total_loss).backward()

    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    scaler.step(optimizer)
    scaler.update()

    elapsed = time.time() - iter_start
    print(f"[{stage_name} | STEP {iter+1}/{max_iters} | Ctx: {current_block_size}] Loss: {loss_accum:.4f} | LR: {lr:.6f} | Time: {elapsed:.1f}s", flush=True)

    if iter > 0 and iter % eval_interval == 0:
        print(f"\n==================================================", flush=True)
        print(f"--- EVALUATION ({stage_name} - Ctx: {current_block_size}) ---", flush=True)
        print(f"==================================================", flush=True)
        
        # Quick loss estimation (fast path)
        losses = estimate_loss(current_block_size, current_batch_size)
        print(f"Eval Results -> Train Loss: {losses['train']:.3f} | Val Loss: {losses['val']:.3f}\n", flush=True)
        try:
            _acc = eval_classify(CLA_EVAL_SAMPLES)
            print(f"Intent-Classification Accuracy (held-out {CLA_EVAL_SAMPLES}): {_acc:.3f}", flush=True)
        except Exception as _ce:
            print(f"[Classify eval] skipped: {_ce}", flush=True)

        # Lightweight text generation test (1 context window only, max 50 tokens)
        model.eval()
        with torch.no_grad():
            test_prompts = ["The future of artificial intelligence is"]
            for prompt in test_prompts:
                context_text = torch.tensor([encode(prompt)], dtype=torch.long, device=device)
                raw_gen = decode(model.generate(context_text, max_new_tokens=50, current_block_size=current_block_size)[0].tolist())
                clean_gen = filter_code_output(raw_gen)
                print(f"[Quick Test] Prompt: {prompt}", flush=True)
                print(f"Completion: {clean_gen}\n", flush=True)

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
    },
    # --- In-data guidance previously enforced at inference time by _is_gibberish /
    #     _synthesize_relevant_cot_and_response / user_wants_code / filter_code_output.
    #     These behaviors are now baked into the checkpoint so the web chatbot can
    #     drop the runtime guards entirely. ---
    {
        "instruction": "can you provide a answer",
        "response": "<think>\nThe user is asking me to answer something but has not specified what. I should acknowledge the request and ask for the topic or specifics rather than refusing.\n</think>\nI understand what you're asking. Tell me the topic or specifics you'd like me to answer, and I'll give you a complete, concrete response rather than stopping here."
    },
    {
        "instruction": "What can you do?",
        "response": "<think>\nThe user is asking about my capabilities. I should give a clear, self-contained list of what I can help with.\n</think>\nI can answer questions on science, technology, history and general knowledge; write, review and explain code; summarize, brainstorm and draft text; reason step-by-step before answering; and use tools like web search when enabled. Tell me what you need and I'll help."
    },
    {
        "instruction": "hello",
        "response": "<think>\nThe user is greeting me. A friendly, brief acknowledgment is the right response.\n</think>\nHello! I'm Ashen AI — how can I help you today?"
    },
    {
        "instruction": "Explain quantum entanglement in one sentence.",
        "response": "<think>\nThe user wants a concise single-sentence explanation of a technical topic. I should define it directly and simply.\n</think>\nQuantum entanglement is a phenomenon where two particles become linked so that the state of one instantly determines the state of the other, no matter how far apart they are."
    },
    {
        "instruction": "What is a transformer architecture?",
        "response": "<think>\nThe user asked a knowledge question. I should answer directly in coherent prose with a clear definition.\n</think>\nA transformer is a neural network architecture that uses self-attention to weigh the importance of every token in a sequence relative to every other token, enabling strong parallelization and long-range dependency modeling. It consists of stacked encoder and/or decoder blocks built from multi-head attention and feed-forward layers."
    },
    {
        "instruction": "Write a Python function to reverse a string.",
        "response": "<think>\nThe user explicitly asked for code, so I should output a working code snippet rather than natural-language prose.\n</think>\ndef reverse_string(s: str) -> str:\n    return s[::-1]"
    },
    {
        "instruction": "can you make a response for me",
        "response": "<think>\nThe user wants me to produce a response but hasn't given a topic. I should acknowledge and ask for the specifics rather than refusing or deflecting.\n</think>\nI understand what you're asking. Tell me the topic, goal, or specifics you'd like me to respond to, and I'll give you a complete, concrete answer rather than stopping here."
    },
    {
        "instruction": "What should I ask you?",
        "response": "<think>\nThe user is unsure what to request. I should offer concrete examples of what I can help with instead of deflecting.\n</think>\nYou can ask me to explain technical topics, write or debug code, summarize text, brainstorm ideas, or answer knowledge questions. Tell me what you're working on and I'll help directly."
    },
    {
        "instruction": "hi there",
        "response": "<think>\nThe user is greeting me casually. A short friendly acknowledgment is appropriate.\n</think>\nHi there! I'm Ashen AI — what can I help you with today?"
    },
    {
        "instruction": "Help me with my homework.",
        "response": "<think>\nThe user needs help but hasn't specified the subject or the exact problem. I should ask for the specifics rather than guessing.\n</think>\nI'd be glad to help with your homework. Tell me the subject and the specific problem or question, and I'll work through it with you step by step."
    },
    {
        "instruction": "What is a vector database?",
        "response": "<think>\nA vector database stores embeddings and supports similarity search. I should define it and give the key use case.\n</think>\nA vector database stores data as high-dimensional vectors (embeddings) and lets you search by similarity rather than exact match. It's widely used for semantic search, recommendations, and retrieval-augmented generation (RAG), where the nearest vectors correspond to the most relevant content."
    },
    {
        "instruction": "Explain the difference between a list and a tuple in Python.",
        "response": "<think>\nLists are mutable; tuples are immutable. That's the core distinction, with downstream effects on usage.\n</think>\nIn Python, a list is mutable (you can change, add, or remove elements) and is defined with square brackets [ ]. A tuple is immutable (fixed once created) and is defined with parentheses ( ). Use tuples for fixed collections like coordinates, and lists when you need to modify the collection."
    },
    {
        "instruction": "Write a Python function to find the nth Fibonacci number.",
        "response": "<think>\nThe Fibonacci sequence starts 0, 1 and each next term is the sum of the previous two. An iterative approach is efficient and avoids recursion depth issues.\n</think>\ndef fibonacci(n: int) -> int:\n    a, b = 0, 1\n    for _ in range(n):\n        a, b = b, a + b\n    return a"
    },
    {
        "instruction": "What is the average time complexity of quicksort?",
        "response": "<think>\nQuicksort is divide-and-conquer; average case is O(n log n) but worst case O(n^2) without good pivot selection.\n</think>\nQuicksort has an average time complexity of O(n log n) and a worst case of O(n^2) when the pivot is consistently poor (e.g., already-sorted input with a naive pivot). In practice, with random or median-of-three pivot selection, it performs close to its average and is very fast in place."
    },
    {
        "instruction": "Summarize the benefits of regular exercise in one paragraph.",
        "response": "<think>\nThe user wants a concise paragraph covering the main benefits of exercise. I should hit physical, mental, and long-term health points.\n</think>\nRegular exercise strengthens the heart and muscles, improves cardiovascular health, and helps maintain a healthy weight. It also boosts mood and reduces anxiety through endorphin release, sharpens sleep quality, and lowers the risk of chronic diseases such as diabetes and hypertension. Even moderate activity most days yields meaningful long-term benefits."
    },
    {
        "instruction": "Translate 'Good morning' to French.",
        "response": "<think>\nA direct, accurate translation is needed.\n</think>\n'Good morning' in French is 'Bonjour'."
    },
    {
        "instruction": "How do I center a div with CSS?",
        "response": "<think>\nFlexbox is the simplest modern way to center both axes. I'll give a concise snippet.\n</think>\nUse flexbox on the parent:\n\n.container {\n    display: flex;\n    justify-content: center; /* horizontal */\n    align-items: center;     /* vertical */\n    height: 100vh;\n}"
    },
    {
        "instruction": "What is the difference between supervised and unsupervised learning?",
        "response": "<think>\nSupervised uses labeled data; unsupervised finds structure in unlabeled data. That's the key distinction.\n</think>\nSupervised learning trains on labeled examples (input -> known output) to predict outcomes, such as classification or regression. Unsupervised learning finds hidden patterns in unlabeled data, such as clustering or dimensionality reduction. The presence or absence of labels is the main difference."
    },
    {
        "instruction": "Write a SQL query to get the top 5 highest-paid employees.",
        "response": "<think>\nOrder by salary descending and limit to 5.\n</think>\nSELECT name, salary\nFROM employees\nORDER BY salary DESC\nLIMIT 5;"
    },
    {
        "instruction": "Explain what an API is to a beginner.",
        "response": "<think>\nAn API is a contract that lets programs talk to each other. I should use a simple analogy.\n</think>\nAn API (Application Programming Interface) is a set of rules that lets one program ask another program for data or services. Think of it like a restaurant menu: you pick a dish (request), the kitchen prepares it (the server processes it), and you get your meal (the response) — without needing to know how the kitchen works."
    },
    {
        "instruction": "Give me a quick healthy dinner idea.",
        "response": "<think>\nThe user wants a concrete, actionable suggestion, not a question back. I should give one clear idea.\n</think>\nTry sheet-pan salmon with broccoli and sweet potato: toss with olive oil, salt, and pepper, roast at 400°F (200°C) for about 20 minutes, and finish with a squeeze of lemon. It's high in protein and fiber and takes minimal prep."
    },
    # --- Intent classification training (spam / not_spam / question / answer / request) ---
    # Teaches the model to label an incoming message with exactly one of the five
    # classes when asked. The label is a single token after ### Response:
    {
        "instruction": "Classify: Buy now!!! Limited offer, 90% OFF only today!!! Click here to claim your prize.",
        "response": "<think>\nThis is unsolicited promotional junk with urgency tricks and a prize lure, sent without the user requesting it.\n</think>\nspam"
    },
    {
        "instruction": "Classify: Congratulations! You've won a $1000 Amazon gift card. Reply with your bank details to receive it.",
        "response": "<think>\nA phishing lure asking for bank details via a fake prize is classic spam.\n</think>\nspam"
    },
    {
        "instruction": "Classify: URGENT: Your account will be closed. Verify your password at this.link-secure.xyz immediately.",
        "response": "<think>\nFake account-verify phishing with a lookalike URL and urgency is spam.\n</think>\nspam"
    },
    {
        "instruction": "Classify: Make $5000 a week working from home! No experience needed, sign up now!!!",
        "response": "<think>\nGet-rich-quick scam with hype and a sign-up push is spam.\n</think>\nspam"
    },
    {
        "instruction": "Classify: You are the 1,000,000th visitor! Claim your free iPhone by clicking this link.",
        "response": "<think>\nLottery-style bait with a free-gift lure is spam.\n</think>\nspam"
    },
    {
        "instruction": "Classify: Re: Invoice #4471 attached. Please review and process payment by Friday.",
        "response": "<think>\nA legitimate business invoice from a sender the user corresponds with is not spam.\n</think>\nnot_spam"
    },
    {
        "instruction": "Classify: The quarterly report is attached for your review.",
        "response": "<think>\nA work document share with no promotional or deceptive intent is not spam.\n</think>\nnot_spam"
    },
    {
        "instruction": "Classify: Your order #88213 has shipped and will arrive Tuesday.",
        "response": "<think>\nA transactional shipping notification from a real order is not spam.\n</think>\nnot_spam"
    },
    {
        "instruction": "Classify: Mom: Don't forget we're having dinner at 7.",
        "response": "<think>\nA personal message from a known contact with no marketing intent is not spam.\n</think>\nnot_spam"
    },
    {
        "instruction": "Classify: Reminder: dentist appointment tomorrow at 3pm.",
        "response": "<think>\nA personal calendar reminder is legitimate, not spam.\n</think>\nnot_spam"
    },
    {
        "instruction": "Classify: What is the capital of France?",
        "response": "<think>\nThis asks for factual information, so it's a question.\n</think>\nquestion"
    },
    {
        "instruction": "Classify: How does a combustion engine work?",
        "response": "<think>\nIt asks for an explanation of a mechanism, so it's a question.\n</think>\nquestion"
    },
    {
        "instruction": "Classify: Why is the sky blue?",
        "response": "<think>\nIt requests a reason/cause, so it's a question.\n</think>\nquestion"
    },
    {
        "instruction": "Classify: Who wrote Pride and Prejudice?",
        "response": "<think>\nIt asks for a factual identity, so it's a question.\n</think>\nquestion"
    },
    {
        "instruction": "Classify: Can you explain the difference between TCP and UDP?",
        "response": "<think>\nIt asks for an explanation, so it's a question.\n</think>\nquestion"
    },
    {
        "instruction": "Classify: The capital of France is Paris.",
        "response": "<think>\nThis is a declarative statement that provides information, so it's an answer.\n</think>\nanswer"
    },
    {
        "instruction": "Classify: A combustion engine burns fuel to produce motion through controlled explosions.",
        "response": "<think>\nThis states a fact in response to a topic, so it's an answer.\n</think>\nanswer"
    },
    {
        "instruction": "Classify: The sky appears blue because air molecules scatter shorter (blue) wavelengths of sunlight more.",
        "response": "<think>\nThis is a declarative explanation, so it's an answer.\n</think>\nanswer"
    },
    {
        "instruction": "Classify: Pride and Prejudice was written by Jane Austen.",
        "response": "<think>\nThis states a fact as a response, so it's an answer.\n</think>\nanswer"
    },
    {
        "instruction": "Classify: TCP is connection-oriented and reliable; UDP is connectionless and faster.",
        "response": "<think>\nThis is a declarative comparison given as information, so it's an answer.\n</think>\nanswer"
    },
    {
        "instruction": "Classify: Remind me to call Mom at 5pm.",
        "response": "<think>\nThis asks the assistant to perform an action (set a reminder), so it's a request.\n</think>\nrequest"
    },
    {
        "instruction": "Classify: Schedule a meeting with the team for Thursday at 2pm.",
        "response": "<think>\nIt asks to create a calendar event, so it's a request.\n</think>\nrequest"
    },
    {
        "instruction": "Classify: Please send the report to the client.",
        "response": "<think>\nIt asks the assistant to perform a task, so it's a request.\n</think>\nrequest"
    },
    {
        "instruction": "Classify: Add 'buy milk' to my shopping list.",
        "response": "<think>\nIt asks to record an item, so it's a request.\n</think>\nrequest"
    },
    {
        "instruction": "Classify: Set a timer for 20 minutes.",
        "response": "<think>\nIt asks to start a timer, so it's a request.\n</think>\nrequest"
    },
    # --- Personal-assistant behavior training (reminders, scheduling, task tracking) ---
    {
        "instruction": "I have a meeting tomorrow at 10am.",
        "response": "<think>\nThe user shared an event. I should acknowledge and offer to remind them, treating it as a personal-assistant task.\n</think>\nGot it — I've noted your meeting tomorrow at 10am. Want me to remind you 15 minutes beforehand?"
    },
    {
        "instruction": "Remind me to call Mom at 5pm.",
        "response": "<think>\nThis is a concrete reminder request. I should confirm I've captured it with the time and action.\n</think>\nDone — I'll remind you to call Mom at 5pm. I've set a reminder for that time."
    },
    {
        "instruction": "Schedule a meeting with the team for Thursday at 2pm.",
        "response": "<think>\nThe user wants a calendar event created. I should confirm the details back.\n</think>\nScheduled: team meeting Thursday at 2pm. I've blocked that slot and will notify the team if you'd like."
    },
    {
        "instruction": "Add 'buy milk' and 'pick up dry cleaning' to my shopping list.",
        "response": "<think>\nTwo list items to capture. I should confirm both were added.\n</think>\nAdded to your shopping list: buy milk, pick up dry cleaning. Anything else to add?"
    },
    {
        "instruction": "What's on my calendar today?",
        "response": "<think>\nThis is an assistant query about the user's schedule. I should answer from what they've shared, noting I only know what they've told me.\n</think>\nBased on what you've shared, today you have a 10am meeting. If you tell me more events I'll keep track of them for you."
    },
    {
        "instruction": "Set a timer for 20 minutes.",
        "response": "<think>\nA simple action request. I should confirm the timer is running.\n</think>\nTimer set for 20 minutes — I'll let you know when it's up."
    },
    {
        "instruction": "Help me plan my day.",
        "response": "<think>\nThe user wants proactive personal-assistant planning. I should offer a concrete structure from what I know.\n</think>\nSure — here's a draft plan from what you've shared: 10am meeting, then deep-work block, 5pm call Mom. Tell me your priorities and I'll arrange the rest around them."
    },
    {
        "instruction": "Don't let me forget the deadline on Friday.",
        "response": "<think>\nA reminder to remember a deadline. I should capture it and offer a nudge.\n</think>\nNoted — Friday deadline. I'll remind you Thursday evening so you have time to wrap up. Want a second nudge Friday morning?"
    },
    {
        "instruction": "Can you keep track of my project tasks?",
        "response": "<think>\nThe user wants ongoing task tracking. I should confirm I'll maintain the list as they add items.\n</think>\nYes — share your tasks and I'll keep a running list, flag due dates, and surface what's next. What's the first task?"
    }
]

sft_ctx = safe_ctx('sft', 8192)   # VRAM-scaled context for SFT
print(f"SFT context length: {sft_ctx}")
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5, fused=True if device == 'cuda' else False)
sft_epochs = 3

for epoch in range(sft_epochs):
    total_sft_loss = 0.0
    random.shuffle(SFT_DATASET)

    for item_idx, item in enumerate(SFT_DATASET):
        prompt_text = f"### Instruction:\n{item['instruction']}\n\n### Response:\n{item['response']}<|endoftext|>"
        tokens = encode(prompt_text)
        if len(tokens) > sft_ctx:
            tokens = tokens[:sft_ctx]

        x = torch.tensor(tokens[:-1], dtype=torch.long, device=device).unsqueeze(0)
        y = torch.tensor(tokens[1:], dtype=torch.long, device=device).unsqueeze(0)

        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast('cuda' if device == 'cuda' else 'cpu', **({} if device != 'cuda' else {'dtype': DT_BF16})):
            _, loss, _, _ = model(x, y, current_block_size=sft_ctx)

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
    pickle.dump(unwrap_compiled(model), f)
print("Training & Supervised Fine-Tuning complete! Upscaled model saved to ashen_gpt_model.pk1", flush=True)

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
    },
    {
        "instruction": "Explain what a neural network is.",
        "chosen": "<think>\nA neural network is layers of connected nodes that learn patterns from data via weighted sums and nonlinear activations.\n</think>\nA neural network is a model inspired by the brain, made of layers of neurons (nodes) that transform input through weighted connections and activation functions. During training, it adjusts weights to minimize error on examples, letting it recognize patterns in images, text, and other data.",
        "rejected": "a neural network is like a brain but for computers"
    },
    {
        "instruction": "Write a function to check if a number is prime.",
        "chosen": "<think>\nCheck divisors up to sqrt(n); handle small and even cases first.\n</think>\ndef is_prime(n: int) -> bool:\n    if n < 2:\n        return False\n    if n < 4:\n        return True\n    if n % 2 == 0:\n        return False\n    for i in range(3, int(n ** 0.5) + 1, 2):\n        if n % i == 0:\n            return False\n    return True",
        "rejected": "def isprime(n):\n  for i in range(2,n):\n    if n%i==0: return False\n  return True"
    },
    {
        "instruction": "What is the capital of Australia?",
        "chosen": "<think>\nThe capital of Australia is Canberra, not Sydney or Melbourne.\n</think>\nThe capital of Australia is Canberra, located in the Australian Capital Territory between Sydney and Melbourne.",
        "rejected": "the capital of australia is sydney"
    },
    {
        "instruction": "How do I sort a list in Python?",
        "chosen": "<think>\nPython's built-in sorted() returns a new sorted list; .sort() sorts in place. I'll show both.\n</think>\nUse sorted() to get a new sorted list:\n\nnums = [3, 1, 2]\nprint(sorted(nums))  # [1, 2, 3]\n\nOr .sort() to sort in place:\n\nnums.sort()\nprint(nums)  # [1, 2, 3]",
        "rejected": "you can sort a list by using the sort method"
    },
    {
        "instruction": "Explain the difference between HTTP and HTTPS.",
        "chosen": "<think>\nHTTPS adds TLS encryption on top of HTTP. That's the core security difference.\n</think>\nHTTP is the protocol for transferring web data in plaintext, while HTTPS wraps HTTP with TLS/SSL encryption. HTTPS protects data in transit (passwords, payments) from interception and confirms the server's identity via certificates. Always use HTTPS for anything sensitive.",
        "rejected": "https is the secure version of http"
    },
    {
        "instruction": "Write a Python class for a simple bank account.",
        "chosen": "<think>\nModel balance with deposit/withdraw methods, guarding against negative balances.\n</think>\nclass BankAccount:\n    def __init__(self, owner: str, balance: float = 0.0):\n        self.owner = owner\n        self.balance = balance\n\n    def deposit(self, amount: float) -> None:\n        if amount <= 0:\n            raise ValueError('amount must be positive')\n        self.balance += amount\n\n    def withdraw(self, amount: float) -> None:\n        if amount > self.balance:\n            raise ValueError('insufficient funds')\n        self.balance -= amount",
        "rejected": "class account:\n    def __init__(self):\n        self.money = 0"
    },
    {
        "instruction": "What is overfitting in machine learning?",
        "chosen": "<think>\nOverfitting is when a model memorizes training data and fails to generalize. I'll explain causes and fixes.\n</think>\nOverfitting happens when a model learns noise and idiosyncrasies of the training set instead of the underlying pattern, so it performs well on training data but poorly on new data. It's caused by too much model capacity or too little data, and is addressed with regularization, dropout, more data, and validation-set monitoring.",
        "rejected": "overfitting is when the model fits too much"
    },
    {
        "instruction": "Summarize why sleep is important in two sentences.",
        "chosen": "<think>\nSleep supports memory consolidation and physical repair. I'll state the two key points concisely.\n</think>\nSleep consolidates memories and lets the brain clear metabolic waste, while the body repairs tissue and regulates hormones. Chronic sleep loss impairs focus, mood, and immune function, raising long-term health risks.",
        "rejected": "sleep is important because you need rest"
    }
]

dpo_beta = 0.1  # DPO temperature parameter
dpo_epochs = 2
dpo_lr = 1e-5   # Lower LR for RL fine-tuning

dpo_ctx = safe_ctx('dpo', 4096)   # VRAM-scaled context for DPO
print(f"DPO context length: {dpo_ctx}")
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
        
        # Truncate if needed (VRAM-scaled context)
        max_len = min(len(tokens_chosen), len(tokens_rejected), dpo_ctx)
        tokens_chosen = tokens_chosen[:max_len]
        tokens_rejected = tokens_rejected[:max_len]
        
        # Create input tensors (shifted for next-token prediction)
        x_chosen = torch.tensor(tokens_chosen[:-1], dtype=torch.long, device=device).unsqueeze(0)
        y_chosen = torch.tensor(tokens_chosen[1:], dtype=torch.long, device=device).unsqueeze(0)
        
        x_rejected = torch.tensor(tokens_rejected[:-1], dtype=torch.long, device=device).unsqueeze(0)
        y_rejected = torch.tensor(tokens_rejected[1:], dtype=torch.long, device=device).unsqueeze(0)
        
        # Forward pass with autocast (bf16)
        with torch.amp.autocast('cuda' if device == 'cuda' else 'cpu', **({} if device != 'cuda' else {'dtype': DT_BF16})):
            _, loss_chosen, _, _ = model(x_chosen, y_chosen, current_block_size=dpo_ctx)
            _, loss_rejected, _, _ = model(x_rejected, y_rejected, current_block_size=dpo_ctx)

            # Compute log probs for DPO
            logits_chosen = model(x_chosen, current_block_size=dpo_ctx)[0]
            logits_rejected = model(x_rejected, current_block_size=dpo_ctx)[0]
            
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
    pickle.dump(unwrap_compiled(model), f)
print(f"DPO-aligned model saved to {final_model_name}", flush=True)
print("=== COMPLETE TRAINING PIPELINE (Pre-training → SFT → DPO) FINISHED ===", flush=True)