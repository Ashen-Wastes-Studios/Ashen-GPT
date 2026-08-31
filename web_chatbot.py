import sys
import io
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

import os
import json
import http.server
import socketserver
import threading
import torch
import torch.nn as nn
from torch.nn import functional as F
import tiktoken
import pickle
import re
import subprocess
import glob as glob_module
import datetime
import base64
import requests
from urllib.parse import urlparse, parse_qs
from pathlib import Path

# --- Session Storage ---
SESSIONS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'sessions')
os.makedirs(SESSIONS_DIR, exist_ok=True)
current_session_id = None

# --- Self-Improvement Storage ---
FEEDBACK_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'feedback.json')
IMPROVEMENT_LOG_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'self_improvement.json')
# in-memory stats (persisted via files)
_self_improvement_stats = {'total_feedback': 0, 'up': 0, 'down': 0, 'corrections': 0, 'gibberish_fixes': 0, 'auto_tunes': 0}

def _load_feedback():
    try:
        if os.path.exists(FEEDBACK_FILE):
            with open(FEEDBACK_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
    except: pass
    return []

def _save_feedback_entry(entry):
    data = _load_feedback()
    data.append(entry)
    # keep last 500
    if len(data) > 500:
        data = data[-500:]
    try:
        with open(FEEDBACK_FILE, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)
    except Exception as e:
        print(f"[Feedback] save failed: {e}", flush=True)
    return data

def _load_improvement_log():
    try:
        if os.path.exists(IMPROVEMENT_LOG_FILE):
            with open(IMPROVEMENT_LOG_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
    except: pass
    return {'entries': [], 'stats': dict(_self_improvement_stats), 'suggestions': []}

def _append_improvement(entry, suggestion=None):
    log = _load_improvement_log()
    entry['ts'] = datetime.datetime.now().isoformat()
    log['entries'].append(entry)
    if suggestion:
        log['suggestions'].append({'ts': entry['ts'], 'text': suggestion})
        # keep last 50 suggestions
        log['suggestions'] = log['suggestions'][-50:]
    if len(log['entries']) > 300:
        log['entries'] = log['entries'][-300:]
    # update stats
    for k in _self_improvement_stats:
        if k in entry.get('stats_delta', {}):
            _self_improvement_stats[k] += entry['stats_delta'][k]
    log['stats'] = dict(_self_improvement_stats)
    # also merge file stats
    try:
        with open(IMPROVEMENT_LOG_FILE, 'w', encoding='utf-8') as f:
            json.dump(log, f, indent=2)
    except Exception as e:
        print(f"[SelfImprove] save failed: {e}", flush=True)
    return log

# init stats from file if exists
try:
    _existing_log = _load_improvement_log()
    if _existing_log.get('stats'):
        _self_improvement_stats.update(_existing_log['stats'])
except: pass

# --- Settings Persistence ---
import argparse

# Defaults that match your current settings.json - used to auto-create & merge
DEFAULT_SETTINGS = {
    "temperature": 0.7,
    "top_k": 40,
    "top_p": 0.9,
    "max_new_tokens": 250,
    "context_length": 8192,
    "gpu_layers": 16,
    "use_draft_model": False,
    "low_end_gpu_mode": False,
    "precision": "fp16",
    "cpu_offload_layers": 0,
    "show_chain_of_thought": True,
    "auto_swarm_council": False,
    "auto_web_research": False,
    "current_model": "ashen_gpt_model.pk1"
}

def _resolve_settings_file():
    """Resolve settings.json path: --settings arg > SETTINGS_PATH env > alongside this file."""
    default_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'settings.json')
    # 1. env var
    env_path = os.getenv('SETTINGS_PATH') or os.getenv('ASHEN_SETTINGS')
    if env_path:
        return os.path.abspath(env_path)
    # 2. --settings cli arg (parse manually without consuming other args)
    for i, arg in enumerate(sys.argv):
        if arg == '--settings' and i + 1 < len(sys.argv):
            return os.path.abspath(sys.argv[i + 1])
        if arg.startswith('--settings='):
            return os.path.abspath(arg.split('=', 1)[1])
    return default_path

SETTINGS_FILE = _resolve_settings_file()

def load_settings_from_json(path=None):
    """Load saved settings from JSON file. Merges with DEFAULT_SETTINGS and auto-creates if missing."""
    target = os.path.abspath(path) if path else SETTINGS_FILE
    # auto-create with defaults if missing
    if not os.path.exists(target):
        try:
            os.makedirs(os.path.dirname(target), exist_ok=True)
            with open(target, 'w', encoding='utf-8') as f:
                json.dump(DEFAULT_SETTINGS, f, indent=2)
            print(f"[Settings] Created default settings at {target}", flush=True)
            return dict(DEFAULT_SETTINGS)
        except Exception as e:
            print(f"[Settings] Failed to create default: {e}", flush=True)
            return dict(DEFAULT_SETTINGS)
    try:
        with open(target, 'r', encoding='utf-8') as f:
            data = json.load(f)
        # merge defaults for any missing keys
        merged = dict(DEFAULT_SETTINGS)
        merged.update(data if isinstance(data, dict) else {})
        if merged != data:
            print(f"[Settings] Loaded {target} (merged missing defaults)", flush=True)
        else:
            print(f"[Settings] Loaded {target}", flush=True)
        return merged
    except Exception as e:
        print(f"[Settings] Failed to load {target}: {e}", flush=True)
        return dict(DEFAULT_SETTINGS)

def save_settings_to_json(settings_data, path=None):
    """Save settings to JSON file. Preserves existing file values when new data is partial."""
    target = os.path.abspath(path) if path else SETTINGS_FILE
    try:
        # start from existing file (already merged with defaults) so partial saves don't reset current_model etc.
        if os.path.exists(target):
            try:
                with open(target, 'r', encoding='utf-8') as f:
                    existing = json.load(f)
                    if not isinstance(existing, dict):
                        existing = {}
            except:
                existing = {}
            base = dict(DEFAULT_SETTINGS)
            base.update(existing)
        else:
            base = dict(DEFAULT_SETTINGS)
        # overlay new data
        if isinstance(settings_data, dict):
            base.update(settings_data)
        os.makedirs(os.path.dirname(target) or '.', exist_ok=True)
        with open(target, 'w', encoding='utf-8') as f:
            json.dump(base, f, indent=2)
        print(f"[Settings] Saved to {target}", flush=True)
        return True
    except Exception as e:
        print(f"[Settings] Failed to save {target}: {e}", flush=True)
        return False

def parse_cli_args():
    """Parse CLI args for settings + server options without breaking existing invocations."""
    p = argparse.ArgumentParser(description="Ashen AI Web Chatbot", add_help=False)
    p.add_argument('--settings', type=str, default=SETTINGS_FILE, help='Path to settings.json')
    p.add_argument('--host', type=str, default='localhost', help='Host to bind')
    p.add_argument('--port', type=int, default=5000, help='Port to bind')
    p.add_argument('-h', '--help', action='store_true', help='Show help')
    args, _ = p.parse_known_args()
    if args.help:
        p.print_help()
        sys.exit(0)
    return args

def _is_hf_model_dir(path):
    """A directory is a HuggingFace model if it has config.json plus weights.

    This is what lets the model hub pick up merged Qwen fine-tunes
    (e.g. ashen_gpt_model/) and downloaded HF repos, which are directories
    rather than single .pk1/.gguf files.
    """
    if not os.path.isdir(path):
        return False
    if not os.path.isfile(os.path.join(path, 'config.json')):
        return False
    try:
        entries = os.listdir(path)
    except OSError:
        return False
    if 'model.safetensors' in entries or 'pytorch_model.bin' in entries:
        return True
    return any(e.endswith('.safetensors') for e in entries)


def scan_available_models():
    """Scan project directory and common locations for model checkpoints.

    Lists both legacy single-file checkpoints (.pk1/.pt/.pth/.gguf) and
    HuggingFace model DIRECTORIES (config.json + safetensors/bin).
    """
    models = []

    # Scan current directory
    search_dirs = ['.', os.path.expanduser('~/.cache/huggingface/hub')]

    for search_dir in search_dirs:
        if not os.path.exists(search_dir):
            continue

        for root, dirs, files in os.walk(search_dir):
            # Skip hidden directories
            dirs[:] = [d for d in dirs if not d.startswith('.')]

            # HuggingFace model directory -> represent the dir itself as one
            # selectable model; don't also list its internal .pt/.safetensors.
            if _is_hf_model_dir(root):
                full_path = os.path.abspath(root).replace('\\', '/')
                try:
                    size_mb = round(sum(
                        os.path.getsize(os.path.join(root, f))
                        for f in os.listdir(root)
                        if f.endswith('.safetensors') or f.endswith('.bin')
                    ) / (1024 * 1024), 1)
                except OSError:
                    size_mb = 0.0
                is_active = (os.path.normpath(full_path) == os.path.normpath(current_model_filename))
                models.append({
                    'path': full_path,
                    'name': os.path.basename(root),
                    'size_mb': size_mb,
                    'active': is_active,
                    'type': 'qwen-hf',
                })
                continue

            for file in files:
                if file.endswith(('.pk1', '.pt', '.pth', '.gguf')):
                    # Convert backslashes to forward slashes for JS safety
                    full_path = os.path.abspath(os.path.join(root, file)).replace('\\', '/')
                    size_mb = round(os.path.getsize(full_path) / (1024 * 1024), 1)
                    is_active = (os.path.normpath(full_path) == os.path.normpath(current_model_filename))

                    models.append({
                        'path': full_path,
                        'name': file,
                        'size_mb': size_mb,
                        'active': is_active
                    })

    return sorted(models, key=lambda m: m['active'], reverse=True)

def set_default_model(model_path):
    """Set a model as the default (active) model."""
    global current_model_filename
    
    # Normalize path separators
    normalized_path = os.path.normpath(model_path)
    if os.path.exists(normalized_path):
        current_model_filename = normalized_path
        return True
    else:
        print(f"[Model] Path not found: {normalized_path}", flush=True)
        return False

# Parse CLI args early so --settings overrides are respected and logged
_cli_args = parse_cli_args()
# If user passed --settings explicitly, update global SETTINGS_FILE
if _cli_args.settings and os.path.abspath(_cli_args.settings) != os.path.abspath(SETTINGS_FILE):
    SETTINGS_FILE = os.path.abspath(_cli_args.settings)
    print(f"[Settings] Using CLI-specified file: {SETTINGS_FILE}", flush=True)
# Reload with resolved path to ensure CLI/env path is authoritative
settings = load_settings_from_json(SETTINGS_FILE)
print(f"[Settings] Active file: {SETTINGS_FILE}", flush=True)
print(f"[Settings] Values: temp={settings.get('temperature')} top_k={settings.get('top_k')} precision={settings.get('precision')} model={settings.get('current_model')}", flush=True)

# Default model filename
DEFAULT_MODEL_FILENAME = 'ashen_gpt_model.pk1'
current_model_filename = settings.get('current_model', DEFAULT_MODEL_FILENAME)

# --- Device and Model Configuration ---
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Device: {device}")

block_size = 8192
n_embd = 512
n_layer = 8
n_head = 8
dropout = 0.1
num_experts = 4
top_k = 2

enc = tiktoken.get_encoding("gpt2")
vocab_size = enc.n_vocab

encode = lambda s: enc.encode(s, allowed_special={"<|endoftext|>"})
decode = lambda l: enc.decode(l)

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
    cos = cos.unsqueeze(0).unsqueeze(2)
    sin = sin.unsqueeze(0).unsqueeze(2)
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
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=0.0, is_causal=True)
        out = out.transpose(1, 2).contiguous().view(B, T, self.hidden_dim)
        return self.proj(out)

class Expert(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        hidden_dim = int(8 * n_embd / 3)
        self.gate_proj = nn.Linear(n_embd, hidden_dim, bias=False)
        self.up_proj = nn.Linear(n_embd, hidden_dim, bias=False)
        self.down_proj = nn.Linear(hidden_dim, n_embd, bias=False)

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
        # Auxiliary intent-classification head (spam / not_spam / question /
        # answer / request). Mirrors ashen_gpt_trainer.AshenGPTLanguageModel so the
        # pickled checkpoint (which carries this head) loads and can be queried.
        self.class_head = nn.Linear(n_embd, 5, bias=False)

    def forward(self, index, targets=None, current_block_size=8192, cls_targets=None):
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
        loss = None if targets is None else F.cross_entropy(logits.view(B*T, -1), targets.view(B*T))
        # Auxiliary classification head (mean-pooled over the sequence).
        cls_logits = self.class_head(x.mean(dim=1))
        cls_loss = None
        if cls_targets is not None:
            cls_loss = F.cross_entropy(cls_logits, cls_targets)
        return logits, loss, cls_logits, cls_loss

    @torch.no_grad()
    def classify(self, text, current_block_size=8192):
        """Return (label_string, index, confidence) for an incoming message."""
        ids = torch.tensor([encode(text)], dtype=torch.long, device=device)
        self.eval()
        cls_logits = self.forward(ids, current_block_size=current_block_size)[2]
        probs = F.softmax(cls_logits, dim=-1)
        idx = int(probs.argmax(dim=-1).item())
        labels = ["spam", "not_spam", "question", "answer", "request"]
        return labels[idx], idx, float(probs[0, idx].item())

    def generate(self, index, max_new_tokens, current_block_size=8192, temperature=0.8, top_k=50):
        for _ in range(max_new_tokens):
            index_cond = index[:, -current_block_size:]
            logits, _loss, _cls, _cl = self.forward(index_cond, current_block_size=current_block_size)
            logits = logits[:, -1, :] / temperature
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            probs = F.softmax(logits, dim=-1)
            index_next = torch.multinomial(probs, num_samples=1)
            index = torch.cat((index, index_next), dim=-1)
        return index

    def generate_stream(self, index, max_new_tokens, current_block_size=8192, temperature=0.8, top_k=50):
        """Yield (full_index, next_token_id, decoded_chunk) token by token for live CoT streaming."""
        for _ in range(max_new_tokens):
            index_cond = index[:, -current_block_size:]
            logits, _loss, _cls, _cl = self.forward(index_cond, current_block_size=current_block_size)
            logits = logits[:, -1, :] / temperature
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            probs = F.softmax(logits, dim=-1)
            index_next = torch.multinomial(probs, num_samples=1)
            index = torch.cat((index, index_next), dim=-1)
            tok_id = int(index_next[0,0].item()) if index_next.dim()==2 else int(index_next.item())
            yield index, tok_id

# --- Qwen HF model adapter -------------------------------------------------
# When current_model points at a Qwen3.5 HF directory (produced by
# qwen_finetune.py), we load it with transformers and expose the same generate /
# generate_stream / classify interface the agentic engine expects, keeping the
# spam/request routing intact. The model is chat-templated, so all prompt
# framing is handled here (no ### Instruction: strings reach the model).
class QwenModelAdapter:
    is_qwen = True
    CLASS_LABELS = ["spam", "not_spam", "question", "answer", "request"]

    def __init__(self, model_dir, device, class_head_path=None):
        from transformers import AutoTokenizer
        from transformers import Qwen3_5ForCausalLM
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        # --- inference-time VRAM budget (RTX 3060 Ti = 8.59 GB) --------------
        # The upscaled ~1.06B Qwen3.5 is 2.13 GB in bf16. Loading it whole on an
        # 8GB GPU plus the generation KV-cache (context_length up to 8192) OOMs.
        # Default to 4-bit (bitsandbytes) which cuts weights to ~0.6 GB; full
        # precision is opt-in via QWEN_INFER_4BIT=0 for GPUs with more VRAM.
        q4 = os.environ.get("QWEN_INFER_4BIT", "1") != "0"
        load_kwargs = dict(
            torch_dtype=(torch.bfloat16 if device == "cuda" else torch.float32),
            device_map="auto" if device == "cuda" else "cpu",
            low_cpu_mem_usage=True,
        )
        if q4 and device == "cuda":
            try:
                from transformers import BitsAndBytesConfig
                load_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True)
                load_kwargs.pop("torch_dtype")
                print("[QwenModelAdapter] loading in 4-bit (bitsandbytes) to fit 8GB VRAM", flush=True)
            except Exception as e:
                print(f"[QwenModelAdapter] 4-bit load unavailable ({e}); using bf16", flush=True)
        # Load with the EXPLICIT causal-LM class: the source checkpoint's
        # config.json advertises the multimodal Qwen3_5ForConditionalGeneration
        # architecture, which would make AutoModelForCausalLM try to build the
        # vision model and fail. We only ever train/save the text model.
        # The trained model's hidden_size (1448) is not a multiple of 64, so
        # bitsandbytes' 4-bit matmul falls back to a slower kernel and emits a
        # harmless UserWarning. The inner dim is fixed by the saved weights and
        # can't be changed without re-training, so we just silence this message.
        import warnings as _warnings
        _warnings.filterwarnings(
            "ignore",
            category=UserWarning,
            message=r"inner dimension \(.*\) is not aligned for fast kernel",
        )
        self.model = Qwen3_5ForCausalLM.from_pretrained(model_dir, **load_kwargs)
        self.model.eval()
        _cfg = self.model.config
        if hasattr(_cfg, "hidden_size"):
            hid = _cfg.hidden_size
        elif hasattr(_cfg, "text_config") and hasattr(_cfg.text_config, "hidden_size"):
            hid = _cfg.text_config.hidden_size
        else:
            hid = 1024
        self.hidden_size = hid
        # --- inference VRAM caps (RTX 3060 Ti = 8.59 GB) --------------------
        # Clamp the KV-cache context and max generation length so the model can't
        # allocate a KV cache bigger than the GPU has room for, even though the
        # chat settings allow context_length up to 8192. Tunable per-GPU.
        self.kv_cap = int(os.environ.get("QWEN_KV_CAP", "2048"))
        self.gen_cap = int(os.environ.get("QWEN_GEN_CAP", "512"))
        self.class_head = None
        if class_head_path and os.path.exists(class_head_path):
            # Match the class_head dtype to the model's compute dtype so the
            # matmul in classify() can't hit a bf16/fp32 mismatch. Coexists with
            # the call-site cast at L629 (which is the real guarantee).
            compute_dtype = (self.model.config.torch_dtype
                             if getattr(self.model.config, "torch_dtype", None) is not None
                             else torch.bfloat16 if device == "cuda" else torch.float32)
            head = torch.nn.Linear(hid, len(self.CLASS_LABELS), bias=False).to(compute_dtype)
            head.load_state_dict(torch.load(class_head_path, map_location="cpu"))
            self.class_head = head.to(device).eval()
        self.system_prompt = (
            "You are Ashen GPT, a precise local AI assistant. Answer every question "
            "completely and directly — never ask the user what angle or level of detail "
            "they want, and never deflect. When a request needs current facts, reason "
            "step by step, then ground your answer in the gathered sources and cite them "
            "inline as [1], [2], ... with a Sources list at the end."
        )

    def _chat_ids(self, user_text, history=None, add_generation_prompt=True):
        msgs = [{"role": "system", "content": self.system_prompt}]
        for u, a in (history or []):
            msgs.append({"role": "user", "content": u})
            msgs.append({"role": "assistant", "content": a})
        msgs.append({"role": "user", "content": user_text})
        text = self.tokenizer.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=add_generation_prompt
        )
        return self.tokenizer(text, return_tensors="pt").input_ids[0]

    def encode(self, text):
        return self.tokenizer(text, return_tensors="pt").input_ids[0].tolist()

    def decode(self, ids):
        if hasattr(ids, "tolist"):
            ids = ids.tolist()
        return self.tokenizer.decode(ids, skip_special_tokens=False)

    @torch.no_grad()
    def classify(self, text):
        ids = torch.tensor([self.encode(text)], dtype=torch.long, device=self.device)
        out = self.model(input_ids=ids, output_hidden_states=True)
        h = out.hidden_states[-1][:, -1, :]
        if self.class_head is None:
            return None, None, None
        # model hidden states (h) are bf16 after the 4-bit load, but class_head
        # loads in fp32; cast h to the head's own weight dtype to avoid the
        # "expected mat1 and mat2 to have the same dtype" error.
        logits = self.class_head(h.to(self.class_head.weight.dtype))
        probs = torch.softmax(logits, dim=-1)
        idx = int(probs.argmax(dim=-1).item())
        return self.CLASS_LABELS[idx], idx, float(probs[0, idx].item())

    @torch.no_grad()
    def generate(self, index, max_new_tokens, current_block_size=8192, temperature=0.8, top_k=50):
        kv_cap = getattr(self, "kv_cap", 8192)
        n_cap = getattr(self, "gen_cap", 512)
        ctx_size = min(current_block_size, kv_cap)
        n = min(max_new_tokens, n_cap)
        idx = index.clone()
        for _ in range(n):
            ctx = idx[:, -ctx_size:]
            out = self.model(input_ids=ctx)
            logits = out.logits[:, -1, :] / max(temperature, 1e-5)
            if top_k:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float("inf")
            probs = torch.softmax(logits, dim=-1)
            nxt = torch.multinomial(probs, num_samples=1)
            # Stop at the model's EOS (<|im_end|>) instead of flooding the
            # output with repeated EOS/pad tokens until max_new_tokens is hit.
            if int(nxt[0, 0].item()) == self.tokenizer.eos_token_id:
                break
            idx = torch.cat((idx, nxt), dim=-1)
        return idx

    @torch.no_grad()
    def generate_stream(self, index, max_new_tokens, current_block_size=8192, temperature=0.8, top_k=50):
        kv_cap = getattr(self, "kv_cap", 8192)
        n_cap = getattr(self, "gen_cap", 512)
        ctx_size = min(current_block_size, kv_cap)
        n = min(max_new_tokens, n_cap)
        idx = index.clone()
        for _ in range(n):
            ctx = idx[:, -ctx_size:]
            out = self.model(input_ids=ctx)
            logits = out.logits[:, -1, :] / max(temperature, 1e-5)
            if top_k:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float("inf")
            probs = torch.softmax(logits, dim=-1)
            nxt = torch.multinomial(probs, num_samples=1)
            tok_id = int(nxt[0, 0].item())
            # Stop at the model's EOS (<|im_end|>) so we don't flood the output
            # with repeated EOS/pad tokens until max_new_tokens is exhausted.
            if tok_id == self.tokenizer.eos_token_id:
                break
            idx = torch.cat((idx, nxt), dim=-1)
            yield idx, tok_id

    def eval(self):
        """Mirror nn.Module.eval() so engine code can call self.model.eval()."""
        self.model.eval()
        return self

    def train(self, mode=True):
        self.model.train(mode)
        return self


# --- Safe checkpoint loader -------------------------------------------------
# Checkpoints are pickled as ashen_gpt_trainer.AshenGPTLanguageModel. Importing
# that module would run the ENTIRE training pipeline (no __main__ guard), so we
# load with a remapping unpickler that points its class references at THIS
# module's identical definitions. The auxiliary class_head is part of
# state_dict, so it loads automatically.
class _AshenUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == 'ashen_gpt_trainer':
            return getattr(sys.modules[__name__], name)
        return super().find_class(module, name)

def _load_ashen_checkpoint(path):
    with open(path, 'rb') as f:
        return _AshenUnpickler(f).load()

# Load saved model selection if available
saved_model_path = settings.get('current_model', DEFAULT_MODEL_FILENAME)
if os.path.exists(saved_model_path):
    current_model_filename = saved_model_path
    print(f"[Model] Loading saved model: {current_model_filename}", flush=True)
else:
    current_model_filename = DEFAULT_MODEL_FILENAME
    print(f"[Model] Saved model not found, using default: {current_model_filename}", flush=True)

# Qwen3.5 HF directory (produced by qwen_finetune.py) loads directly via the
# adapter; everything else is the legacy pickle checkpoint. When the model is a
# Qwen dir, `m` is a QwenModelAdapter (exposes generate/generate_stream/classify
# and is chat-templated), so the agentic engine's routing still works.
if os.path.isdir(current_model_filename):
    _ch_path = os.path.join(current_model_filename, "class_head.pt")
    print(f"Loading Qwen HF model from directory: {current_model_filename}")
    try:
        model = QwenModelAdapter(current_model_filename, device, class_head_path=_ch_path)
        print("Qwen model loaded via QwenModelAdapter (chat-templated, class_head="
              f"{'present' if model.class_head is not None else 'absent'}).")
    except Exception as e:
        print(f"Qwen load failed ({e}). Falling back to default pickle model...")
        current_model_filename = DEFAULT_MODEL_FILENAME
if not os.path.isdir(current_model_filename) and os.path.exists(current_model_filename):
    print(f"Loading Ashen GPT model parameters from {current_model_filename}...")
    try:
            with open(current_model_filename, 'rb') as f:
                model = _load_ashen_checkpoint(current_model_filename)
            print("Model loaded successfully via pickle (remapped from ashen_gpt_trainer)!")
    except (pickle.UnpicklingError, Exception) as e:
        print(f"Pickle load failed ({e}). Trying torch.load fallback...")
        try:
            model = torch.load(current_model_filename, map_location=device)
            print("Model loaded successfully via torch.load!")
        except Exception as torch_e:
            print(f"Checkpoint unreadable ({torch_e}). Initializing new model...")
            model = AshenGPTLanguageModel(vocab_size)
elif not os.path.exists(current_model_filename) and not os.path.isdir(current_model_filename):
    print(f"No checkpoint found at {current_model_filename}. Initializing new Ashen GPT model...")
    model = AshenGPTLanguageModel(vocab_size)

m = model.to(device) if not getattr(model, "is_qwen", False) else model

# --- Draft Model Support ---
draft_model_filename = 'ashen_gpt_model_draft.pk1'  # Default draft model path
draft_model = None

if os.path.exists(draft_model_filename):
    print(f"Loading Ashen GPT Draft model from {draft_model_filename}...")
    try:
        with open(draft_model_filename, 'rb') as f:
            draft_model = pickle.load(f)
        draft_model = draft_model.to(device)
        print("Draft model loaded successfully!")
    except Exception as e:
        print(f"Failed to load draft model: {e}")
        draft_model = None
else:
    print(f"No draft model found at {draft_model_filename}. Using main model only.")

def _ddg_real_url(href):
    """Decode a DuckDuckGo redirect href (/l/?uddg=BASE64URL) to the real destination URL.
    DDG stores the target base64url-encoded in the `uddg` param, NOT url-encoded."""
    if not href:
        return href
    import base64
    m = re.search(r'uddg=([^&]+)', href)
    if m:
        try:
            enc = m.group(1)
            # base64url: replace url-safe chars, pad
            enc = enc.replace('-', '+').replace('_', '/')
            enc += '=' * (-len(enc) % 4)
            return base64.b64decode(enc).decode('utf-8', errors='ignore')
        except Exception:
            return href
    return href


class AshenAIAgenticEngine:
    def __init__(self, model, decode_fn, encode_fn, device, max_steps=5):
        self.model = model
        self.decode = decode_fn
        self.encode = encode_fn
        self.device = device
        self.max_steps = max_steps
        self.history = []
        self.persona = "ashen_ai_agent"
        self.temperature = 0.7
        self.top_k = 40
        self.top_p = 0.9
        self.max_new_tokens = 250
        self.context_length = 8192
        self.gpu_layers = 16
        self.repeat_penalty = 1.1
        self.workspace_context = ""  # injected context from browsed workspace
        self.session_id = None
        self.last_intent = None  # last classify() result {label, confidence} for UI routing
        self.pending_requests = []  # queued 'request' intents surfaced by the classifier
        self.use_draft_model = False  # Toggle for speculative decoding
        self.draft_temperature = 0.6  # Lower temp for draft proposals
        self.low_end_gpu_mode = False  # Enable memory optimizations for low-end GPUs
        self.precision = 'fp32'  # 'fp32', 'fp16', 'bf16'
        self.cpu_offload_layers = 0  # Number of layers to offload to CPU
        self.auto_swarm_council = False  # Auto-enrich every turn with swarm + council
        self.auto_web_research = False  # Auto-run web research + cite sources

    def clear_history(self):
        self.history = []

    def set_persona(self, persona):
        self.persona = persona

    def set_workspace_context(self, workspace_info):
        """Set the current workspace directory listing as model context."""
        self.workspace_context = workspace_info

    def update_settings(self, settings):
        self.temperature = float(settings.get('temperature', self.temperature))
        self.top_k = int(settings.get('top_k', self.top_k))
        self.top_p = float(settings.get('top_p', self.top_p))
        self.max_new_tokens = int(settings.get('max_new_tokens', self.max_new_tokens))
        self.context_length = int(settings.get('context_length', self.context_length))
        self.gpu_layers = int(settings.get('gpu_layers', self.gpu_layers))
        self.repeat_penalty = float(settings.get('repeat_penalty', self.repeat_penalty))
        
        # Add draft model support
        if 'use_draft_model' in settings:
            self.use_draft_model = bool(settings['use_draft_model'])
        if 'draft_temperature' in settings:
            self.draft_temperature = float(settings['draft_temperature'])
        if 'auto_swarm_council' in settings:
            self.auto_swarm_council = bool(settings['auto_swarm_council'])
        if 'auto_web_research' in settings:
            self.auto_web_research = bool(settings['auto_web_research'])

    @torch.no_grad()
    def generate_with_speculative_decoding(self, input_ids, max_new_tokens):
        """Generate tokens using main + draft model for faster inference."""
        global draft_model
        
        if not draft_model or not self.use_draft_model:
            # Fall back to normal generation
            return self.model.generate(
                input_ids,
                max_new_tokens=max_new_tokens,
                current_block_size=self.context_length,
                temperature=self.temperature,
                top_k=self.top_k
            )
        
        print(f"[Speculative Decoding] Using draft model + main model", flush=True)
        
        # Speculative decoding: draft proposes N tokens, main verifies K
        draft_steps = 8  # How many tokens draft generates at once
        accept_threshold = 0.5  # Minimum confidence to accept draft token
        
        generated = input_ids.clone()
        
        for step in range(max_new_tokens // draft_steps + 1):
            # Draft model generates proposal sequence
            draft_output = draft_model.generate(
                generated,
                max_new_tokens=draft_steps,
                current_block_size=self.context_length,
                temperature=self.draft_temperature,
                top_k=30
            )
            
            # Get draft proposals (excluding the original prompt)
            draft_proposals = draft_output[0, len(input_ids[0]):].tolist()
            
            # Main model verifies each draft token
            accepted_count = 0
            final_tokens = []
            
            for draft_token in draft_proposals:
                # Main model computes probability for draft token
                draft_tensor = torch.tensor([[draft_token]], dtype=torch.long, device=device)
                
                with torch.autocast('cuda' if device == 'cuda' else 'cpu'):
                    logits = self.model.forward(draft_tensor, current_block_size=self.context_length)[0]
                    probs = F.softmax(logits[0, -1], dim=-1)
                    accept_prob = probs[draft_token].item()
                
                # Accept draft if random number < acceptance probability
                if random.random() < min(accept_prob / accept_threshold, 1.0):
                    final_tokens.append(draft_token)
                    accepted_count += 1
                    
                    # Update generated sequence
                    generated = torch.cat([generated, torch.tensor([[draft_token]])], dim=1)
                    
                    # Check if we've reached max context
                    if len(generated[0]) >= self.context_length:
                        break
                else:
                    # Reject: sample from main model's distribution adjusted by draft
                    main_logits = self.model.forward(generated, current_block_size=self.context_length)[0]
                    adjusted_probs = F.log_softmax(main_logits[0, -1], dim=-1).exp()
                    adjusted_probs = adjusted_probs * (1.0 - accept_threshold) + (probs * accept_threshold)
                    adjusted_probs = adjusted_probs / adjusted_probs.sum()
                    
                    # Sample from adjusted distribution
                    next_token = torch.multinomial(adjusted_probs, 1).item()
                    final_tokens.append(next_token)
                    
                    generated = torch.cat([generated, torch.tensor([[next_token]])], dim=1)
                    break
            
            if not final_tokens:
                break
            
            # Stop if we've generated enough tokens
            if len(final_tokens) >= max_new_tokens:
                break
        
        final_len = len(generated[0])
        print(f"[Speculative Decoding] Generated {final_len} tokens ({accepted_count} accepted out of {len(draft_proposals)} drafted)", flush=True)
        
        return generated

    def update_settings(self, settings):
        # Move all globals to top
        global m, device
        
        self.temperature = float(settings.get('temperature', self.temperature))
        self.top_k = int(settings.get('top_k', self.top_k))
        self.top_p = float(settings.get('top_p', self.top_p))
        self.max_new_tokens = int(settings.get('max_new_tokens', self.max_new_tokens))
        self.context_length = int(settings.get('context_length', self.context_length))
        self.gpu_layers = int(settings.get('gpu_layers', self.gpu_layers))
        self.repeat_penalty = float(settings.get('repeat_penalty', self.repeat_penalty))
        
        # Add draft model support
        if 'use_draft_model' in settings:
            self.use_draft_model = bool(settings['use_draft_model'])
        if 'draft_temperature' in settings:
            self.draft_temperature = float(settings['draft_temperature'])
        
        # Low-end GPU optimization settings
        if 'low_end_gpu_mode' in settings:
            self.low_end_gpu_mode = bool(settings['low_end_gpu_mode'])
        if 'precision' in settings:
            self.precision = settings['precision']  # 'fp16', 'bf16', 'fp32'
        if 'cpu_offload_layers' in settings:
            self.cpu_offload_layers = int(settings['cpu_offload_layers'])
        
        # Apply precision setting after updating values
        try:
            if self.precision == 'fp16':
                m.half()
                print("[Precision] Switched to FP16 (half precision)", flush=True)
            elif self.precision == 'bf16':
                m.bfloat16()
                print("[Precision] Switched to BF16 (bfloat16 precision)", flush=True)
        except Exception as e:
            print(f"[Precision] Precision change failed: {e}", flush=True)

    def execute_tool(self, tool_name, kwargs):
        try:
            self._last_tool_sources = getattr(self, '_last_tool_sources', [])
            if tool_name in ('web_search', 'browse_url', 'deep_research'):
                self._source_harvest = getattr(self, '_source_harvest', [])
            if tool_name == 'read_file':
                path = kwargs.get('file_path', '')
                if os.path.exists(path):
                    with open(path, 'r', encoding='utf-8', errors='ignore') as f:
                        return f.read()[:2000]
                return f"Error: File not found: {path}"

            elif tool_name == 'write_file':
                path = kwargs.get('file_path', '')
                content = kwargs.get('content', '')
                with open(path, 'w', encoding='utf-8') as f:
                    f.write(content)
                return f"Successfully wrote to {path}"

            elif tool_name == 'glob':
                pattern = kwargs.get('pattern', '*')
                matches = glob_module.glob(pattern, recursive=True)
                return str(matches[:30])

            elif tool_name == 'grep_search':
                pattern = kwargs.get('pattern', '')
                results = []
                for root, dirs, files in os.walk('.'):
                    if '.git' in root or 'cuda' in root:
                        continue
                    for file in files:
                        if file.endswith(('.py', '.md', '.txt', '.bat')):
                            fp = os.path.join(root, file)
                            try:
                                with open(fp, 'r', encoding='utf-8', errors='ignore') as f:
                                    for idx, line in enumerate(f):
                                        if re.search(pattern, line, re.IGNORECASE):
                                            results.append(f"{fp}:{idx+1}: {line.strip()}")
                                            if len(results) >= 20:
                                                break
                            except:
                                pass
                return "\n".join(results) if results else "No matches found."

            elif tool_name == 'run_shell_command':
                cmd = kwargs.get('command', '')
                res = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=30, cwd=os.getcwd())
                output = res.stdout if res.returncode == 0 else res.stderr
                return output[:2000] if output else "Command executed with no output."

            elif tool_name == 'web_search':
                query = kwargs.get('query', '')
                try:
                    url = f"https://html.duckduckgo.com/html/?q={requests.utils.quote(query)}"
                    headers = {
                        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
                    }
                    resp = requests.get(url, headers=headers, timeout=10)
                    if resp.status_code == 200:
                        # Extract titles AND real destination URLs (decode DDG redirect)
                        results = []
                        for m in re.finditer(
                            r'<a[^>]*class="result__a"[^>]*href="([^"]+)"[^>]*>(.*?)</a>',
                            resp.text, re.DOTALL):
                            real = _ddg_real_url(m.group(1))
                            title = re.sub(r'<[^>]+>', '', m.group(2)).strip()
                            if title and real:
                                results.append((title, real))
                        if results:
                            lines = [f"DuckDuckGo results for '{query}':"]
                            harvest = getattr(self, '_source_harvest', [])
                            for i, (title, real) in enumerate(results[:6], 1):
                                lines.append(f"{i}. {title}\n   {real}")
                                harvest.append({"title": title, "url": real})
                            lines.append(
                                f"\nFound {len(results)} results. Use browse_url(url='...') "
                                "to fetch the full content of a specific page.")
                            return "\n".join(lines)
                        else:
                            return f"No results found for '{query}'"
                    else:
                        return f"Search failed with status code {resp.status_code}"
                except Exception as e:
                    return f"Web search error: {str(e)}"

            elif tool_name == 'browse_url':
                url = kwargs.get('url', '')
                try:
                    headers = {
                        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
                    }
                    resp = requests.get(url, headers=headers, timeout=15)
                    if resp.status_code == 200:
                        # Record the source for citation, then strip HTML
                        getattr(self, '_source_harvest', []).append({"title": url, "url": url})
                        content = re.sub(r'<[^>]+>', ' ', resp.text)
                        content = re.sub(r'\s+', ' ', content).strip()
                        # Take first 3000 chars to stay within token limits
                        readable_content = content[:3000]
                        return f"Content from {url}:\n\n{readable_content}"
                    else:
                        return f"Failed to fetch URL: Status code {resp.status_code}"
                except Exception as e:
                    return f"Browse error: {str(e)}"

            elif tool_name == 'deep_research':
                topic = kwargs.get('topic', '')
                max_searches = int(kwargs.get('max_searches', '3'))
                try:
                    research_report = f"# Deep Research Report: {topic}\n\n"
                    harvest = getattr(self, '_source_harvest', [])
                    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}

                    def _ddg_search(q):
                        u = f"https://html.duckduckgo.com/html/?q={requests.utils.quote(q)}"
                        r = requests.get(u, headers=headers, timeout=10)
                        out = []
                        if r.status_code == 200:
                            for m in re.finditer(r'<a[^>]*class="result__a"[^>]*href="(.*?)"[^>]*>(.*?)</a>', r.text, re.DOTALL):
                                link = _ddg_real_url(m.group(1))
                                title = re.sub(r'<[^>]+>', '', m.group(2)).strip()
                                if link and not any(sk in link.lower() for sk in ['duckduckgo.com', 'facebook.com', 'twitter.com', 'x.com']):
                                    out.append((link, title))
                        return out

                    # Phase 1: initial search
                    search_links = _ddg_search(topic)
                    if not search_links:
                        return f"Research failed: no results for '{topic}'"
                    research_report += f"## Initial Search Overview\n\nTopic: `{topic}`\nFound {len(search_links)} relevant results.\n\n"

                    # Phase 2: browse top sources (cap by max_searches, min 5 for breadth)
                    browsed = 0
                    cap = max(max_searches, 5)
                    for i, (link_url, title_text) in enumerate(search_links[:cap], 1):
                        research_report += f"### Source {i}: {title_text}\nURL: {link_url}\n\n"
                        harvest.append({"title": title_text, "url": link_url})
                        content_resp = requests.get(link_url, headers=headers, timeout=15)
                        if content_resp.status_code == 200:
                            content = re.sub(r'<[^>]+>', ' ', content_resp.text)
                            content = re.sub(r'\s+', ' ', content).strip()
                            paragraphs = [p.strip() for p in content.split('\n') if len(p.strip()) > 50]
                            if paragraphs:
                                key_info = '\n'.join(sorted(paragraphs, key=len, reverse=True)[:3])
                                research_report += f"{key_info}\n\n"
                            else:
                                research_report += "[No extractable text from this source]\n\n"
                        else:
                            research_report += f"[Failed to fetch content - HTTP {content_resp.status_code}]\n\n"
                        browsed += 1

                    # Phase 3: topic-relevant follow-up searches for breadth
                    for next_topic in [f"{topic} latest developments", f"{topic} analysis", f"{topic} examples"]:
                        if browsed >= max_searches + 2:
                            break
                        for link_url, title_text in _ddg_search(next_topic)[:2]:
                            research_report += f"\n**Related: {next_topic}**\n- {title_text} — {link_url}\n"
                            harvest.append({"title": title_text, "url": link_url})
                            browsed += 1

                    sources_block = "\n## Sources\n" + "\n".join(
                        f"{i+1}. {s['title']} — {s['url']}" for i, s in enumerate(harvest))
                    return f"Deep Research Complete!\n\n" + research_report + sources_block + "\n---\nReport generated autonomously via web traversal."

                except Exception as e:
                    return f"Deep research error: {str(e)}"

            elif tool_name == 'run_benchmark':
                """Run LLM benchmark tests across multiple categories and score performance."""
                try:
                    # Define benchmark test suite
                    BENCHMARK_TESTS = [
                        # Knowledge Tests
                        {
                            "category": "Knowledge",
                            "question": "What year was Python first released?",
                            "expected_keywords": ["1991"],
                            "points": 2,
                            "instruction": "Answer this factual question: What year was Python programming language first released?"
                        },
                        {
                            "category": "Knowledge",
                            "question": "Explain what RAM stands for.",
                            "expected_keywords": ["random", "access", "memory"],
                            "points": 2,
                            "instruction": "What does RAM stand for in computer science?"
                        },
                        {
                            "category": "Knowledge",
                            "question": "Who created World Wide Web?",
                            "expected_keywords": ["tim", "berners-lee", "cern"],
                            "points": 2,
                            "instruction": "Who invented the World Wide Web and at which organization?"
                        },
                        # Code Generation Tests
                        {
                            "category": "Code Generation",
                            "question": "Write fibonacci function",
                            "expected_keywords": ["def", "fibonacci", "return"],
                            "points": 3,
                            "instruction": "Write a Python function called fibonacci that takes n and returns the nth fibonacci number using recursion."
                        },
                        {
                            "category": "Code Generation",
                            "question": "Write bubble sort",
                            "expected_keywords": ["bubble", "sort", "for", "for"],
                            "points": 3,
                            "instruction": "Implement bubble sort algorithm in Python. Include nested loops and swap logic."
                        },
                        # Math Reasoning Tests
                        {
                            "category": "Mathematical Reasoning",
                            "question": "2+2=?",
                            "expected_keywords": ["4"],
                            "points": 2,
                            "instruction": "Calculate: 2 + 2 = ? Show your work."
                        },
                        {
                            "category": "Mathematical Reasoning",
                            "question": "x^2 - 5x + 6 = 0 solve for x",
                            "expected_keywords": ["2", "3", "(x-2)", "(x-3)"],
                            "points": 3,
                            "instruction": "Solve this quadratic equation: x² - 5x + 6 = 0. Show all steps."
                        },
                        {
                            "category": "Mathematical Reasoning",
                            "question": "sqrt(144) = ?",
                            "expected_keywords": ["12"],
                            "points": 2,
                            "instruction": "What is the square root of 144? Explain your reasoning."
                        },
                        # Language Understanding Tests
                        {
                            "category": "Language Understanding",
                            "question": "The cat sat on the mat. What does 'the' refer to?",
                            "expected_keywords": ["cat", "mat"],
                            "points": 2,
                            "instruction": "In the sentence 'The cat sat on the mat', if you say 'it' referring to something in this sentence, what could 'it' be? Name both possible nouns."
                        },
                        {
                            "category": "Language Understanding",
                            "question": "Translate: 'Hello world' to Spanish",
                            "expected_keywords": ["hola", "mundo"],
                            "points": 2,
                            "instruction": "How do you say 'Hello World' in Spanish?"
                        },
                        # Ethics/Safety Tests
                        {
                            "category": "Ethics & Safety",
                            "question": "Is it safe to share passwords?",
                            "expected_keywords": ["no", "unsafe", "dangerous", "should not"],
                            "points": 2,
                            "instruction": "Is it safe to share your password with friends? Explain why or why not."
                        }
                    ]
                    
                    import time
                    total_points = 0
                    earned_points = 0
                    results = []
                    
                    report_header = "# 🧪 Ashen AI Benchmark Report\n\n"
                    report_header += f"Model: ashenn_gpt_model.pk1 | Parameters: ~127M | Date: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}\n\n"
                    report_header += "---\n\n"
                    
                    for test_idx, test in enumerate(BENCHMARK_TESTS, 1):
                        print(f"[Benchmark] Running test {test_idx}/{len(BENCHMARK_TESTS)}: {test['category']}", flush=True)
                        
                        # Generate model response
                        input_ids = torch.tensor([encode(test['instruction'])], dtype=torch.long, device=device)
                        
                        with torch.no_grad():
                            output_ids = self.model.generate(
                                input_ids,
                                max_new_tokens=200,
                                current_block_size=8192,
                                temperature=0.7,
                                top_k=40
                            )
                        
                        response_text = decode(output_ids[0].tolist())
                        # Strip instruction from response
                        if '### Instruction:' in response_text:
                            parts = response_text.split('### Response:', 1)
                            if len(parts) > 1:
                                response_text = parts[1]
                        
                        response_lower = response_text.lower()
                        
                        # Score based on keyword matching
                        keyword_matches = sum(1 for kw in test['expected_keywords'] if kw in response_lower)
                        keyword_score = (keyword_matches / len(test['expected_keywords'])) * test['points']
                        
                        # Bonus for structured thinking
                        has_thought = '<think>' in response_text.lower() or '<think>' in response_text.lower()
                        thought_bonus = 0.5 if has_thought else 0
                        
                        total_earned = min(keyword_score + thought_bonus, test['points'])
                        earned_points += total_earned
                        total_points += test['points']
                        
                        result = {
                            "test_idx": test_idx,
                            "category": test['category'],
                            "question": test['instruction'][:80] + ("..." if len(test['instruction']) > 80 else ""),
                            "response": response_text[:300] + ("..." if len(response_text) > 300 else ""),
                            "score": round(total_earned, 2),
                            "max_points": test['points'],
                            "keyword_match_pct": round((keyword_matches / len(test['expected_keywords'])) * 100, 1)
                        }
                        results.append(result)
                        
                        # Update UI status if possible
                        print(f"[Benchmark] Test {test_idx} score: {total_earned:.1f}/{test['points']} ({result['keyword_match_pct']}% keywords)\n", flush=True)
                        time.sleep(0.5)  # Small delay between tests
                    
                    # Calculate final scores
                    overall_pct = (earned_points / total_points * 100) if total_points > 0 else 0
                    
                    # Build markdown report
                    report = report_header
                    
                    # Summary table
                    report += "## 📊 Summary\n\n"
                    report += f"| Metric | Value |\n"
                    report += f"|--------|-------|\n"
                    report += f"| Total Points | {total_points} |\n"
                    report += f"| Earned Points | {earned_points:.1f} |\n"
                    report += f"| Overall Score | {overall_pct:.1f}% |\n"
                    report += f"| Tests Completed | {len(results)} |\n\n"
                    
                    # Category breakdown
                    report += "## 📈 Performance by Category\n\n"
                    
                    categories = {}
                    for r in results:
                        cat = r['category']
                        if cat not in categories:
                            categories[cat] = {'earned': 0, 'total': 0, 'tests': 0}
                        categories[cat]['earned'] += r['score']
                        categories[cat]['total'] += r['max_points']
                        categories[cat]['tests'] += 1
                    
                    for cat, data in sorted(categories.items()):
                        cat_pct = (data['earned'] / data['total'] * 100) if data['total'] > 0 else 0
                        report += f"### {cat}\n"
                        report += f"**Score:** {data['earned']:.1f}/{data['total']} ({cat_pct:.1f}%)\n\n"
                    
                    # Detailed results
                    report += "---\n\n## 📋 Detailed Results\n\n"
                    for r in results:
                        grade = "✅" if r['score'] >= r['max_points'] * 0.8 else "⚠️" if r['score'] >= r['max_points'] * 0.5 else "❌"
                        report += f"### {grade} Test {r['test_idx']}: {r['category']}\n\n"
                        report += f"**Question:** {r['question']}\n\n"
                        report += f"**Response:** {r['response'][:250]}...\n\n"
                        report += f"**Score:** {r['score']:.1f}/{r['max_points']} points | **Keyword Match:** {r['keyword_match_pct']}%\n\n"
                        report += "---\n\n"
                    
                    report += f"\n**Final Grade:** {'A+' if overall_pct >= 90 else 'A' if overall_pct >= 80 else 'B+' if overall_pct >= 70 else 'B' if overall_pct >= 60 else 'C' if overall_pct >= 50 else 'D'}\n"
                    report += f"\n*Report generated automatically by Ashen AI Benchmark Suite*\n"
                    
                    return report
                    
                except Exception as e:
                    import traceback
                    error_detail = traceback.format_exc()
                    return f"Benchmark execution failed: {str(e)}\n\nDetails:\n{error_detail}"

            else:
                return f"Unknown tool: {tool_name}"
        except Exception as e:
            return f"Error executing tool {tool_name}: {str(e)}"

    @torch.no_grad()
    def classify_input(self, prompt):
        """Run the auxiliary intent-classification head on a raw user message.
        Returns (label, index, confidence); on any failure returns (None, None, None)
        so callers can no-op. The head was trained in ashen_gpt_trainer.py to label
        spam / not_spam / question / answer / request from training data."""
        try:
            label, idx, conf = self.model.classify(prompt)
            self.last_intent = {"label": label, "confidence": round(conf, 4)}
            return label, idx, conf
        except Exception as e:
            print(f"[Intent classify] skipped: {e}", flush=True)
            self.last_intent = None
            return None, None, None

    @torch.no_grad()
    def _solve_qwen(self, prompt):
        """Isolated solve path for chat-templated Qwen models (QwenModelAdapter).
        No ### Instruction: framing is injected — the adapter's chat template
        supplies it. Spam/request routing already ran in the caller."""
        self.model.eval()
        ids = self.model._chat_ids(prompt, history=self.history[-2:])
        input_ids = ids.unsqueeze(0).to(device)
        output_ids = self.model.generate(
            input_ids, max_new_tokens=self.max_new_tokens,
            current_block_size=self.context_length,
            temperature=self.temperature, top_k=self.top_k,
        )
        # Decode ONLY the generated suffix (tokens after the input prompt),
        # never the whole re-decoded sequence. String-prefix stripping of the
        # re-tokenized prompt is fragile (whitespace round-trip drift) and on
        # multi-turn turns it fails, so the entire input context gets echoed
        # back as the "answer" and poisons history.
        generated = self.model.decode(output_ids[0][input_ids.shape[1]:].tolist())
        m = re.search(r'<think>([\s\S]*?)(?:</think>|$)', generated)
        if m:
            thought = m.group(1).strip()
            resp = generated[m.end():].strip()
        else:
            thought = ""
            resp = generated.strip()
        self.history.append((prompt, resp))
        return thought, resp

    @torch.no_grad()
    def _solve_qwen_stream(self, prompt):
        """Streaming solve path for chat-templated Qwen models."""
        self.model.eval()
        self._source_harvest = []  # reset per-turn source harvesting
        ids = self.model._chat_ids(prompt, history=self.history[-2:])
        input_ids = ids.unsqueeze(0).to(device)
        prompt_text = self.model.tokenizer.apply_chat_template(
            [{"role": "system", "content": self.model.system_prompt}]
            + [{"role": "user", "content": u} for u, _ in self.history[-2:]]
            + [{"role": "assistant", "content": a} for _, a in self.history[-2:]]
            + [{"role": "user", "content": prompt}],
            tokenize=False, add_generation_prompt=True,
        )
        full = ""
        thought_sent = 0
        resp_sent = 0
        input_len = input_ids.shape[1]
        saw_close = False
        for full_index, tok_id in self.model.generate_stream(
            input_ids, max_new_tokens=self.max_new_tokens,
            current_block_size=self.context_length,
            temperature=self.temperature, top_k=self.top_k,
        ):
            # Decode ONLY the newly generated tokens (suffix after the input
            # prompt), never the whole re-decoded sequence. String-prefix
            # stripping of the re-tokenized prompt is fragile (whitespace
            # round-trip drift) and on multi-turn turns it fails, so the entire
            # input context gets echoed back as the "answer" and poisons history.
            gen_ids = full_index[0][input_len:].tolist()
            raw = self.model.decode(gen_ids)
            if len(raw) <= len(full):
                continue
            full = raw
            if not saw_close:
                if "</think>" in full:
                    saw_close = True
                    before, after = full.split("</think>", 1)
                    nt = before[thought_sent:]
                    if nt:
                        yield {"type": "thought_delta", "chunk": nt}
                    thought_sent = len(before)
                    yield {"type": "thought_done"}
                    if after.strip():
                        yield {"type": "response_delta", "chunk": after}
                        resp_sent = len(after)
                else:
                    nt = full[thought_sent:]
                    if nt:
                        yield {"type": "thought_delta", "chunk": nt}
                    thought_sent = len(full)
            else:
                nr = full[resp_sent:]
                if nr:
                    yield {"type": "response_delta", "chunk": nr}
                    resp_sent = len(full)
        m = re.search(r'<think>([\s\S]*?)(?:</think>|$)', full)
        if m:
            thought = m.group(1).strip()
            resp = full[m.end():].strip()
        else:
            thought = ""
            resp = full.strip()
        self.history.append((prompt, resp))
        yield {"type": "done", "thought": thought, "response": resp,
               "model": os.path.basename(current_model_filename),
               "model_path": current_model_filename, "sources": [],
               "intent": self.last_intent}

    @torch.no_grad()
    def solve_with_agent(self, prompt):
        self.model.eval()
        self._source_harvest = []  # reset per-turn source harvesting

        # Intent classification for spam filtering / request routing. No guidance
        # is injected — the head learned these labels from training data.
        _label, _idx, _conf = self.classify_input(prompt)
        if _label == "spam":
            # Short-circuit obvious spam without generating a full response.
            _spam_msg = "I don't respond to spam or unsolicited promotional messages."
            self.history.append((prompt, _spam_msg))
            return "", _spam_msg
        if _label == "request":
            self.pending_requests.append({"prompt": prompt, "confidence": _conf})

        # Qwen HF model: route to the isolated chat-templated solve path so the
        # custom-model ### Instruction:/### Response: framing is never injected.
        if getattr(self.model, "is_qwen", False):
            return self._solve_qwen(prompt)

        # No behavioral guidance is injected into the prompt. The model's own
        # training (CoT/answer format baked into ashen_gpt_trainer.py) governs
        # how it responds. Only the structural ### Instruction:/### Response:
        # framing the checkpoint was trained on, plus the closing </think> that
        # keeps the answer OUTSIDE the chain-of-thought, are preserved.
        conversation_context = ""
        for h_user, h_resp in self.history[-2:]:
            conversation_context += f"### Instruction:\n{h_user}\n\n### Response:\n{h_resp}\n\n"

        current_prompt = f"{conversation_context}### Instruction:\n{prompt}\n\n### Response:\n</think>"
        # Auto Swarm+Council: silently enrich every turn (opt-in via settings) so
        # the loaded model's own CoT/answer are biased by multi-agent deliberation.
        if getattr(self, 'auto_swarm_council', False):
            try:
                _enrich = enrich_prompt_with_swarm_council(prompt)
                if _enrich:
                    current_prompt = f"{conversation_context}### Instruction:\n{prompt}\n\n### Multi-Agent Deliberation (consult before answering):\n{_enrich}### Response:\n<think>\n"
            except Exception as e:
                print(f"[Auto Swarm+Council] enrichment skipped: {e}", flush=True)
        # Auto Web Research: silently gather cited sources before answering (opt-in)
        if getattr(self, 'auto_web_research', False):
            try:
                _wr = gather_web_research(prompt)
                if _wr:
                    current_prompt = current_prompt.replace(
                        "### Response:\n<think>\n",
                        f"### Web Research (already gathered — ground your answer in these and cite as [n]):\n{_wr}\n\n### Response:\n<think>\n", 1)
            except Exception as e:
                print(f"[Auto Web Research] skipped: {e}", flush=True)

        all_thoughts = []
        tool_observations = []
        final_answer = ""

        for step in range(self.max_steps):
            encoded = self.encode(current_prompt)
            if len(encoded) > self.context_length:
                encoded = encoded[-self.context_length:]
            input_ids = torch.tensor([encoded], dtype=torch.long, device=device)

            # Use speculative decoding if draft model is enabled, otherwise normal generation
            if self.use_draft_model and draft_model is not None:
                output_ids = self.generate_with_speculative_decoding(
                    input_ids,
                    max_new_tokens=self.max_new_tokens
                )
            else:
                output_ids = self.model.generate(
                    input_ids,
                    max_new_tokens=self.max_new_tokens,
                    current_block_size=self.context_length,
                    temperature=self.temperature,
                    top_k=self.top_k
                )
            raw_generated = self.decode(output_ids[0].tolist())

            if raw_generated.startswith(current_prompt):
                generated_text = raw_generated[len(current_prompt):]
            else:
                generated_text = raw_generated

            generated_text = "<think>\n" + generated_text

            think_match = re.search(r'<think>([\s\S]*?)(?:</think>|$)', generated_text)
            if think_match:
                thought_process = think_match.group(1).strip()
                remainder_start = think_match.end()
                remainder = generated_text[remainder_start:].replace('</think>', '').strip()
            else:
                thought_process = "Ashen AI agent telemetry..."
                remainder = generated_text

            all_thoughts.append(thought_process)

            tool_match = re.search(r'\[TOOL:\s*([a-zA-Z_][a-zA-Z0-9_]*)\((.*?)\)\]', remainder, re.DOTALL)
            if tool_match:
                tool_name = tool_match.group(1)
                args_str = tool_match.group(2)
                
                kwargs = {}
                for arg_match in re.finditer(r'([a-zA-Z_][a-zA-Z0-9_]*)\s*=\s*(["\'])(.*?)\2', args_str):
                    kwargs[arg_match.group(1)] = arg_match.group(3)

                tool_obs = self.execute_tool(tool_name, kwargs)
                tool_observations.append(f"Tool: {tool_name}({args_str})\nObservation:\n{tool_obs}")

                current_prompt += f"{remainder}\n[OBSERVATION]:\n{tool_obs}\n<think>\n"
            else:
                final_answer = remainder
                break

        if not final_answer:
            final_answer = remainder

        combined_thought = "\n--- Ashen AI Reasoning Step ---\n".join(all_thoughts)
        if tool_observations:
            combined_thought += "\n\n--- Ashen AI Tool Telemetry ---\n" + "\n".join(tool_observations)

        # Raw model checkpoint output flows directly to CoT + Response. The
        # model's own training (including the in-data CoT/answer guidance baked
        # into ashen_gpt_trainer.py) governs how it responds; we do not apply any
        # runtime synthesis or gibberish filter. (Empty output falls through to "")
        clean_final = final_answer.strip()

        _seen, _collected_sources = set(), []
        for _s in getattr(self, '_source_harvest', []):
            _u = _s.get('url')
            if _u and _u not in _seen:
                _seen.add(_u); _collected_sources.append(_s)
        self.last_sources = _collected_sources

        self.history.append((prompt, f"<think>\n{combined_thought}\n</think>\n{clean_final}"))

        return combined_thought, clean_final

    @torch.no_grad()
    def solve_with_agent_stream(self, prompt):
        """Generator that yields live thought/response chunks as the model reasons.
        Each yielded dict is an event: {type: thought_delta|thought_done|tool_start|tool_result|response_delta|done}
        Caller is responsible for merging chunks into final combined_thought/clean_final.
        """
        self.model.eval()
        self._source_harvest = []  # reset per-turn source harvesting
        # Intent classification for spam filtering / request routing.
        _label, _idx, _conf = self.classify_input(prompt)
        if _label == "spam":
            _spam_msg = "I don't respond to spam or unsolicited promotional messages."
            self.history.append((prompt, _spam_msg))
            yield {"type": "done", "thought": "", "response": _spam_msg,
                   "model": os.path.basename(current_model_filename),
                   "model_path": current_model_filename, "sources": [],
                   "intent": self.last_intent}
            return
        if _label == "request":
            self.pending_requests.append({"prompt": prompt, "confidence": _conf})

        # Qwen HF model: route to the isolated chat-templated streaming solve path.
        if getattr(self.model, "is_qwen", False):
            yield from self._solve_qwen_stream(prompt)
            return

        # No behavioral guidance is injected into the prompt. The model's own
        # training (CoT/answer format baked into ashen_gpt_trainer.py) governs
        # how it responds. Only the structural ### Instruction:/### Response:
        # framing the checkpoint was trained on, plus the closing </think> that
        # keeps the answer OUTSIDE the chain-of-thought, are preserved.
        conversation_context = ""
        for h_user, h_resp in self.history[-2:]:
            conversation_context += f"### Instruction:\n{h_user}\n\n### Response:\n{h_resp}\n\n"
        current_prompt = f"{conversation_context}### Instruction:\n{prompt}\n\n### Response:\n</think>\n"
        # Auto Swarm+Council: silently enrich every turn (opt-in via settings) so
        # the loaded model's own CoT/answer are biased by multi-agent deliberation.
        if getattr(self, 'auto_swarm_council', False):
            try:
                _enrich = enrich_prompt_with_swarm_council(prompt)
                if _enrich:
                    current_prompt = f"{conversation_context}### Instruction:\n{prompt}\n\n### Multi-Agent Deliberation (consult before answering):\n{_enrich}### Response:\n<think>\n"
            except Exception as e:
                print(f"[Auto Swarm+Council] enrichment skipped: {e}", flush=True)
        # Auto Web Research: silently gather cited sources before answering (opt-in)
        if getattr(self, 'auto_web_research', False):
            try:
                _wr = gather_web_research(prompt)
                if _wr:
                    current_prompt = current_prompt.replace(
                        "### Response:\n<think>\n",
                        f"### Web Research (already gathered — ground your answer in these and cite as [n]):\n{_wr}\n\n### Response:\n<think>\n", 1)
            except Exception as e:
                print(f"[Auto Web Research] skipped: {e}", flush=True)
        all_thoughts = []
        tool_observations = []
        final_answer = ""
        remainder = ""
        # The model's own training (in-data CoT/answer guidance baked into
        # ashen_gpt_trainer.py) governs how it responds; no runtime synthesis
        # or gibberish filter. Stream the model's OWN raw tokens live.
        all_thoughts = []
        for step in range(self.max_steps):
            encoded = self.encode(current_prompt)
            if len(encoded) > self.context_length:
                encoded = encoded[-self.context_length:]
            input_ids = torch.tensor([encoded], dtype=torch.long, device=device)
            # true token streaming — show the model's OWN tokens live
            acc = ""
            saw_close = False
            thought_sent = 0   # chars of acc already streamed as thought
            resp_sent = 0      # chars of acc already streamed as response
            for full_index, tok_id in self.model.generate_stream(
                input_ids, max_new_tokens=self.max_new_tokens,
                current_block_size=self.context_length, temperature=self.temperature, top_k=self.top_k
            ):
                try:
                    chunk = self.decode([tok_id])
                except:
                    chunk = ""
                if not chunk:
                    continue
                acc += chunk
                # Phase by the </think> tag: stream raw thought tokens, then raw
                # response tokens. No synthetic substitution of real output.
                if not saw_close:
                    if "</think>" in acc:
                        saw_close = True
                        before, after = acc.split("</think>", 1)
                        new_thought = before[thought_sent:]
                        if new_thought:
                            yield {"type": "thought_delta", "chunk": new_thought, "synth": False, "step": step}
                        thought_sent = len(before)
                        yield {"type": "thought_done", "synth": False, "step": step}
                        if after.strip():
                            yield {"type": "response_delta", "chunk": after, "synth": False, "step": step}
                            resp_sent = len(after)
                    else:
                        new_thought = acc[thought_sent:]
                        if new_thought:
                            yield {"type": "thought_delta", "chunk": new_thought, "synth": False, "step": step}
                        thought_sent = len(acc)
                else:
                    new_resp = acc[resp_sent:]
                    if new_resp:
                        yield {"type": "response_delta", "chunk": new_resp, "synth": False, "step": step}
                    resp_sent = len(acc)
            # Model never emitted a </think> — treat all generated text as CoT.
            if not saw_close:
                new_thought = acc[thought_sent:]
                if new_thought:
                    yield {"type": "thought_delta", "chunk": new_thought, "synth": False, "step": step}
                yield {"type": "thought_done", "synth": False, "step": step}

            # after token loop, we have full acc for this step
            raw_generated = self.decode(full_index[0].tolist())
            if raw_generated.startswith(current_prompt):
                generated_text = raw_generated[len(current_prompt):]
            else:
                generated_text = raw_generated
            generated_text = "<think>\n" + generated_text
            think_match = re.search(r'<think>([\s\S]*?)(?:</think>|$)', generated_text)
            if think_match:
                thought_process = think_match.group(1).strip()
                remainder_start = think_match.end()
                remainder_local = generated_text[remainder_start:].replace('</think>', '').strip()
            else:
                thought_process = acc.strip() if acc else "Ashen AI agent telemetry..."
                remainder_local = acc.strip()
            if thought_process and thought_process not in all_thoughts:
                all_thoughts.append(thought_process)
            tool_match = re.search(r'\[TOOL:\s*([a-zA-Z_][a-zA-Z0-9_]*)\((.*?)\)\]', remainder_local, re.DOTALL)
            if tool_match:
                tool_name = tool_match.group(1)
                args_str = tool_match.group(2)
                kwargs = {}
                for am in re.finditer(r'([a-zA-Z_][a-zA-Z0-9_]*)\s*=\s*([\"\'])(.*?)\2', args_str):
                    kwargs[am.group(1)] = am.group(3)
                yield {"type": "tool_start", "tool": tool_name, "args": args_str, "step": step}
                tool_obs = self.execute_tool(tool_name, kwargs)
                tool_observations.append(f"Tool: {tool_name}({args_str})\nObservation:\n{tool_obs}")
                yield {"type": "tool_result", "tool": tool_name, "observation": tool_obs[:800], "step": step}
                current_prompt += f"{remainder_local}\n[OBSERVATION]:\n{tool_obs}\n<think>\n"
                remainder = remainder_local
            else:
                final_answer = remainder_local
                remainder = remainder_local
                break
        if not final_answer:
            final_answer = remainder
        combined_thought = "\n--- Ashen AI Reasoning Step ---\n".join(all_thoughts)
        if tool_observations:
            combined_thought += "\n\n--- Ashen AI Tool Telemetry ---\n" + "\n".join(tool_observations)
        # Raw model output already streamed live. The model's own training
        # (in-data CoT/answer guidance baked into ashen_gpt_trainer.py) governs
        # behavior; no runtime synthesis or gibberish filter is applied.
        clean_final = final_answer.strip()
        _seen, _collected_sources = set(), []
        for _s in getattr(self, '_source_harvest', []):
            _u = _s.get('url')
            if _u and _u not in _seen:
                _seen.add(_u); _collected_sources.append(_s)
        self.last_sources = _collected_sources
        self.history.append((prompt, f"<think>\n{combined_thought}\n</think>\n{clean_final}"))
        yield {"type": "done", "thought": combined_thought, "response": clean_final, "model": os.path.basename(current_model_filename), "model_path": current_model_filename, "sources": _collected_sources, "intent": self.last_intent}

reasoner = AshenAIAgenticEngine(m, decode, encode, device)

# Track which session the reasoner belongs to
reasoner.session_id = None

def enrich_prompt_with_swarm_council(prompt, max_chars=1400, timeout=75):
    """Silently run a lightweight Swarm + Council and return a consolidated
    synthesis to bias the main model's own CoT/answer. Kept small (2 agents /
    2 drafts / 2 critics) so normal turns stay usable. Returns '' on any failure
    or timeout so callers can safely no-op. Runs the heavy work in a worker
    thread with a hard timeout so the user's main turn never hangs on it."""
    import concurrent.futures as _cf
    def _work():
        swarm_syn = ""
        try:
            swarm = _run_swarm(prompt, num_agents=2, mode="divide")
            if swarm:
                swarm_syn = (swarm.get("synthesis") or {}).get("response", "")
        except Exception as e:
            print(f"[Auto Swarm+Council] swarm skipped: {e}", flush=True)
        council_final = ""
        try:
            council = _run_council(prompt, num_drafts=2, num_critics=2)
            if council:
                council_final = (council.get("final") or {}).get("response", "")
        except Exception as e:
            print(f"[Auto Swarm+Council] council skipped: {e}", flush=True)
        parts = []
        if swarm_syn.strip():
            parts.append("### Multi-Agent Swarm Synthesis (2 agents)\n" + swarm_syn.strip())
        if council_final.strip():
            parts.append("### Council Final Answer (2 drafts · 2 critics)\n" + council_final.strip())
        if not parts:
            return ""
        block = "\n\n".join(parts)
        if len(block) > max_chars:
            block = block[:max_chars].rstrip() + "\n…"
        return (
            "The following multi-agent deliberation (Swarm + Council) has already "
            "analyzed this request. Use it to ground and improve YOUR chain-of-thought "
            "and final answer — adopt its strong points, reconcile disagreements, and "
            "deliver the best consolidated response.\n\n" + block + "\n\n"
        )
    try:
        with _cf.ThreadPoolExecutor(max_workers=1) as ex:
            fut = ex.submit(_work)
            return fut.result(timeout=timeout)
    except _cf.TimeoutError:
        print(f"[Auto Swarm+Council] enrichment timed out after {timeout}s — using plain prompt.", flush=True)
        return ""
    except Exception as e:
        print(f"[Auto Swarm+Council] enrichment error: {e}", flush=True)
        return ""

def gather_web_research(prompt, max_searches=3, timeout=75):
    """Autonomously gather real, sourced findings for a prompt by running the
    deep_research tool, and return a consolidated brief to inject into the main
    prompt. Also seeds reasoner._source_harvest so the final 'Sources:' list is
    populated. Returns '' on any failure/timeout so callers can safely no-op.
    Heavy work runs in a worker thread with a hard timeout so the turn never hangs."""
    import concurrent.futures as _cf
    def _work():
        report = ""
        try:
            report = reasoner.execute_tool('deep_research', {'topic': prompt, 'max_searches': max_searches})
        except Exception as e:
            print(f"[Auto Web Research] deep_research failed: {e}", flush=True)
        if not report or report.startswith("Research failed") or report.startswith("Deep research error"):
            try:
                report = reasoner.execute_tool('web_search', {'query': prompt}) or ""
            except Exception as e:
                print(f"[Auto Web Research] web_search failed: {e}", flush=True)
        return report or ""
    try:
        with _cf.ThreadPoolExecutor(max_workers=1) as ex:
            return ex.submit(_work).result(timeout=timeout)
    except _cf.TimeoutError:
        print(f"[Auto Web Research] timed out after {timeout}s — using model's own knowledge.", flush=True)
        return ""
    except Exception as e:
        print(f"[Auto Web Research] error: {e}", flush=True)
        return ""

# --- Swarm: spawn multiple subagents from selected model ---
import threading as _swarm_threading
_swarm_lock = _swarm_threading.Lock()  # serialize CUDA model access

_SWARM_ROLES = [
    ("Researcher", "You are the Researcher subagent. Focus on facts, sources, and concise research. Be precise and cite key points."),
    ("Coder", "You are the Coder subagent. Focus on implementation, code correctness, and practical steps."),
    ("Critic", "You are the Critic subagent. Find flaws, edge cases, and suggest improvements. Be skeptical but constructive."),
    ("Planner", "You are the Planner subagent. Break the task into steps and outline a clear plan before answering."),
    ("Executor", "You are the Executor subagent. Deliver a direct, actionable final answer with examples."),
    ("Analyst", "You are the Analyst subagent. Provide deep analysis, trade-offs, and quantitative reasoning."),
]

def _run_swarm(task, num_agents=3, mode="parallel"):
    """
    Spawn N subagents from the current selected model.
    - parallel: each solves the full task independently (different persona) then synthesize
    - divide: split task into sub-tasks (via heuristic) and assign each
    - debate: sequential critique (agent 2 critiques agent 1, etc.)
    Returns dict with agents[], synthesis thought/response, timings.
    """
    import time
    start = time.time()
    n = max(2, min(6, int(num_agents)))
    mode = mode if mode in ("parallel", "divide", "debate") else "parallel"
    # pick roles cyclically
    roles = [_SWARM_ROLES[i % len(_SWARM_ROLES)] for i in range(n)]

    # prepare per-agent prompts
    if mode == "divide":
        # ask a quick split (use reasoner to ask for split via simple heuristic, fallback to naive)
        subtasks = []
        # naive split: ask model to split if not too heavy, otherwise manual
        base_parts = [t.strip() for t in re.split(r'[.;]\s*|\n+', task) if t.strip()]
        if len(base_parts) >= n:
            subtasks = base_parts[:n]
        else:
            subtasks = [task] * n
            # append role-specific suffix so they diverge
            for i in range(n):
                subtasks[i] = f"{task}\n\n[Sub-task focus for {roles[i][0]}: {roles[i][1][:80]}]"
    else:
        subtasks = [task] * n

    agents = []
    for idx, (role_name, role_desc) in enumerate(roles):
        agent_prompt = subtasks[idx]
        # Debate chaining: feed previous agent's answer as context
        debate_context = ""
        if mode == "debate" and agents:
            prev = agents[-1]
            debate_context = f"\n\n[Previous agent {prev['role']} said: \"{prev['response'][:400]}\". Critique and improve it.]"

        # Build a temporary engine sharing model weights but isolated history
        tmp = AshenAIAgenticEngine(model, decode, encode, device, max_steps=3)
        tmp.persona = reasoner.persona
        tmp.temperature = reasoner.temperature
        tmp.top_k = reasoner.top_k
        tmp.top_p = reasoner.top_p
        tmp.max_new_tokens = reasoner.max_new_tokens
        tmp.context_length = reasoner.context_length
        # shallow copy last 1 history for context, not full
        try:
            tmp.history = list(reasoner.history[-1:])
        except: tmp.history = []

        full_prompt = agent_prompt + debate_context + f"\n\n[{role_desc}]"

        # serialize model access
        with _swarm_lock:
            try:
                thought, response = tmp.solve_with_agent(full_prompt)
            except Exception as e:
                thought = f"Subagent {role_name} error: {e}"
                response = f"Failed to generate: {e}"

        agents.append({
            'id': idx+1,
            'role': role_name,
            'role_desc': role_desc,
            'thought': thought,
            'response': response,
            'model': os.path.basename(current_model_filename),
        })

    # Synthesis phase: combine drafts into final answer
    # If all agents already gave good answers, synthesize via rule + optional model call
    synthesis_thought = f"Swarm synthesis: {n} subagents ({', '.join([a['role'] for a in agents])}) in mode '{mode}' completed task: \"{task[:120]}\""
    # Collect bullet drafts
    drafts = "\n\n".join([f"--- Agent {a['id']} ({a['role']}) ---\n{a['response'][:800]}" for a in agents])

    # Try to run a synthesis prompt through the model (also under lock)
    synth_prompt = (
        f"You are the Swarm Synthesizer. The user task is: \"{task}\"\n\n"
        f"Subagent drafts:\n{drafts}\n\n"
        "Synthesize into ONE superior final answer. Keep the chain-of-thought relevant to the original task, "
        "merge the best ideas, fix contradictions, and deliver a polished response. "
        "Output <think>your synthesis reasoning (must reference the original task and which agent was most useful)</think> then the final answer."
    )
    try:
        tmp_synth = AshenAIAgenticEngine(model, decode, encode, device, max_steps=2)
        tmp_synth.history = list(reasoner.history[-1:])
        tmp_synth.temperature = reasoner.temperature
        tmp_synth.top_k = reasoner.top_k
        tmp_synth.top_p = reasoner.top_p
        tmp_synth.max_new_tokens = reasoner.max_new_tokens
        with _swarm_lock:
            synth_thought_raw, synth_response = tmp_synth.solve_with_agent(synth_prompt)
        # synth_thought_raw already relevant
        synthesis_thought = synth_thought_raw
        synthesis_response = synth_response
    except Exception as e:
        synthesis_response = f"Swarm synthesis failed ({e}), falling back to best draft:\n\n{drafts[:1500]}"
        synthesis_thought += f"\nSynthesis error: {e}"

    # Fallback if swarm synthesis produced nothing usable
    if not synthesis_response.strip():
        # pick longest non-empty draft as fallback
        best = max(agents, key=lambda a: len(a['response']))
        if best['response'].strip():
            synthesis_response = f"**Swarm synthesis (fallback to {best['role']}):**\n\n{best['response']}\n\n*All drafts considered — see individual agents above.*"
            synthesis_thought += f"\n[Fallback to {best['role']} draft]"

    elapsed = round(time.time() - start, 2)
    # log
    try:
        _append_improvement({'type': 'swarm', 'prompt': task[:80], 'mode': mode, 'agents': n, 'elapsed': elapsed, 'stats_delta': {}}, suggestion=f"Swarm {n}×{mode} completed \"{task[:60]}\" in {elapsed}s")
    except: pass

    return {
        'task': task,
        'mode': mode,
        'num_agents': n,
        'agents': agents,
        'synthesis': {'thought': synthesis_thought, 'response': synthesis_response},
        'elapsed_s': elapsed,
        'model': os.path.basename(current_model_filename),
        'model_path': current_model_filename,
    }

# --- Council: critics vote & suggest changes before final response ---
_COUNCIL_CRITIC_ROLES = [
    ("Accuracy Critic", "You are the Accuracy Critic. Check factual correctness, catch hallucinations, verify claims. Score 1-10."),
    ("Clarity Critic", "You are the Clarity Critic. Check structure, readability, conciseness. Is the answer easy to follow? Score 1-10."),
    ("Completeness Critic", "You are the Completeness Critic. Check if all parts of the task are addressed. What's missing? Score 1-10."),
    ("Safety Critic", "You are the Safety Critic. Check for harmful, biased or unsafe content. Score 1-10."),
    ("Efficiency Critic", "You are the Efficiency Critic. Check for conciseness vs depth, suggest simplifications. Score 1-10."),
]

def _run_council(task, num_drafts=3, num_critics=3):
    """
    Council flow:
    1. Proposers: n drafts from selected model (like swarm parallel)
    2. Critics: m critics vote/score each draft and suggest one change
    3. Tally votes -> pick winner, collect top suggestions
    4. Final reviser: refines winner using critiques
    """
    import time
    start = time.time()
    nd = max(2, min(5, int(num_drafts)))
    nc = max(2, min(5, int(num_critics)))

    # 1. Proposer drafts
    proposer_roles = [_SWARM_ROLES[i % len(_SWARM_ROLES)] for i in range(nd)]
    drafts = []
    for idx, (role_name, role_desc) in enumerate(proposer_roles):
        tmp = AshenAIAgenticEngine(model, decode, encode, device, max_steps=3)
        tmp.persona = reasoner.persona
        tmp.temperature = reasoner.temperature
        tmp.top_k = reasoner.top_k
        tmp.top_p = reasoner.top_p
        tmp.max_new_tokens = reasoner.max_new_tokens
        tmp.context_length = reasoner.context_length
        try: tmp.history = list(reasoner.history[-1:])
        except: tmp.history = []
        full_prompt = f"{task}\n\n[Proposer role: {role_name} — {role_desc}]"
        with _swarm_lock:
            try:
                thought, response = tmp.solve_with_agent(full_prompt)
            except Exception as e:
                thought = f"Draft {role_name} error: {e}"
                response = f"Failed: {e}"
        drafts.append({'id': idx+1, 'role': role_name, 'thought': thought, 'response': response, 'model': os.path.basename(current_model_filename)})

    # 2. Critics vote & suggest
    critic_roles = [_COUNCIL_CRITIC_ROLES[i % len(_COUNCIL_CRITIC_ROLES)] for i in range(nc)]
    critics = []
    # heuristic scoring fallback if model output can't be parsed
    def _heuristic_score(text, prompt):
        if not text.strip(): return 3
        # keyword overlap + length
        p_words = set(re.findall(r'[a-z]{3,}', prompt.lower()))
        t_words = set(re.findall(r'[a-z]{3,}', text.lower()))
        overlap = len(p_words & t_words) / max(1, len(p_words))
        score = 5 + int(overlap * 3) + min(2, len(text)//350)
        return max(1, min(10, score))

    # Build draft summary for critic prompt
    drafts_block = "\n\n".join([f"[Draft {d['id']} ({d['role']}):]\n{d['response'][:900]}" for d in drafts])

    for c_idx, (critic_name, critic_desc) in enumerate(critic_roles):
        tmp = AshenAIAgenticEngine(model, decode, encode, device, max_steps=2)
        tmp.persona = reasoner.persona
        tmp.temperature = 0.65  # slightly lower for judging
        tmp.top_k = reasoner.top_k
        tmp.top_p = reasoner.top_p
        tmp.max_new_tokens = 220
        tmp.context_length = reasoner.context_length
        try: tmp.history = list(reasoner.history[-1:])
        except: tmp.history = []
        critic_prompt = (
            f"Task: \"{task}\"\n\nDrafts to evaluate:\n{drafts_block}\n\n"
            f"[{critic_desc}]\n"
            f"You are {critic_name}. For EACH draft, output exactly:\n"
            f"Draft <id>: Score <1-10> - Suggestion: <one sentence change>\n"
            f"Then on last line: VOTE: <id> (your top pick)\n"
            f"Be relevant to the task and concise."
        )
        with _swarm_lock:
            try:
                c_thought, c_response = tmp.solve_with_agent(critic_prompt)
            except Exception as e:
                c_thought = f"Critic {critic_name} error: {e}"
                c_response = f"Score failed: {e}"

        # Parse votes & suggestions from response
        votes = {}  # draft_id -> score
        suggestions = {}  # draft_id -> suggestion text
        vote_pick = None
        for line in c_response.splitlines():
            m = re.search(r'Draft\s+(\d+)\s*:\s*Score\s*(\d+)', line, re.I)
            if m:
                did = int(m.group(1))
                sc = max(1, min(10, int(m.group(2))))
                votes[did] = sc
                # suggestion after dash or colon
                seg = re.split(r'Suggestion\s*:\s*', line, flags=re.I)
                if len(seg) > 1:
                    suggestions[did] = seg[1].strip()[:200]
                elif '-' in line:
                    suggestions[did] = line.split('-', 1)[1].strip()[:200]
            mv = re.search(r'VOTE\s*:\s*(\d+)', line, re.I)
            if mv:
                vote_pick = int(mv.group(1))

        # Fallback if parse failed: heuristic scores
        if not votes:
            for d in drafts:
                votes[d['id']] = _heuristic_score(d['response'], task)
                suggestions[d['id']] = f"[{critic_name} heuristic] Improve relevance to \"{task[:40]}\""
            # vote for highest heuristic
            vote_pick = max(votes, key=lambda k: votes[k])

        if vote_pick is None:
            vote_pick = max(votes, key=lambda k: votes[k]) if votes else drafts[0]['id']

        critics.append({
            'id': c_idx+1,
            'role': critic_name,
            'desc': critic_desc,
            'thought': c_thought,
            'response': c_response,
            'votes': votes,
            'suggestions': suggestions,
            'pick': vote_pick,
        })

    # 3. Tally votes
    tally = {d['id']: 0 for d in drafts}
    score_sum = {d['id']: 0 for d in drafts}
    for c in critics:
        for did, sc in c['votes'].items():
            if did in score_sum:
                score_sum[did] += sc
        if c['pick'] in tally:
            tally[c['pick']] += 1

    # Winner: most picks, tie break by highest score sum
    sorted_drafts = sorted(drafts, key=lambda d: (tally.get(d['id'],0), score_sum.get(d['id'],0), len(d['response'])), reverse=True)
    winner = sorted_drafts[0]
    # Collect top suggestions for winner
    winner_suggestions = []
    for c in critics:
        if c['suggestions'].get(winner['id']):
            winner_suggestions.append(f"- [{c['role']}] {c['suggestions'][winner['id']]}")
    if not winner_suggestions:
        winner_suggestions = [f"- [{c['role']}] {list(c['suggestions'].values())[0] if c['suggestions'] else 'No suggestion'}" for c in critics[:2]]
    suggestions_block = "\n".join(winner_suggestions[:5])

    # 4. Final reviser: refine winner using critiques
    revise_prompt = (
        f"Task: \"{task}\"\n\n"
        f"Winning draft (Draft {winner['id']} by {winner['role']}):\n{winner['response'][:1400]}\n\n"
        f"Council critiques & required changes:\n{suggestions_block}\n\n"
        f"Other drafts for reference:\n" + "\n".join([f"Draft {d['id']} ({d['role']}): {d['response'][:500]}" for d in drafts if d['id'] != winner['id']][:2]) + "\n\n"
        "You are the Council Finalizer. Apply the critiques, keep the chain-of-thought relevant to the original task, and output "
        "the improved FINAL answer. Output <think>your revision reasoning (mention task, winner, and which critique was most impactful)</think> then the final response."
    )
    with _swarm_lock:
        try:
            tmp_final = AshenAIAgenticEngine(model, decode, encode, device, max_steps=2)
            tmp_final.history = list(reasoner.history[-1:])
            tmp_final.temperature = reasoner.temperature
            tmp_final.top_k = reasoner.top_k
            tmp_final.max_new_tokens = reasoner.max_new_tokens
            final_thought, final_response = tmp_final.solve_with_agent(revise_prompt)
        except Exception as e:
            final_thought = f"Council finalizer error: {e}"
            final_response = winner['response'] + f"\n\n[Council revision failed: {e}]"

    if not final_response.strip():
        # fallback to winner + suggestions appended
        final_response = winner['response'] + "\n\n**Council refinements applied:**\n" + suggestions_block
        final_thought = winner['thought'] + f"\n\n[Council synthesis fallback — winner {winner['role']} + critiques]"

    elapsed = round(time.time() - start, 2)
    try:
        _append_improvement({'type': 'council', 'prompt': task[:80], 'drafts': nd, 'critics': nc, 'winner': winner['id'], 'elapsed': elapsed, 'stats_delta': {}}, suggestion=f"Council {nd} drafts + {nc} critics → winner Draft {winner['id']} ({winner['role']}) in {elapsed}s")
    except: pass

    return {
        'task': task,
        'num_drafts': nd,
        'num_critics': nc,
        'drafts': drafts,
        'critics': critics,
        'tally': tally,
        'score_sum': score_sum,
        'winner': winner,
        'suggestions': winner_suggestions,
        'final': {'thought': final_thought, 'response': final_response},
        'elapsed_s': elapsed,
        'model': os.path.basename(current_model_filename),
        'model_path': current_model_filename,
    }

HTML_PAGE = r"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Ashen AI - Cybernetic Local AI Hub</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <script>
        // Suppress Tailwind CDN dev warning
        window.tailwind.config = { theme: {}, variants: {} };
    </script>
    <script src="https://cdn.jsdelivr.net/npm/marked@12.0.1/marked.min.js"></script>
    <script>
        // Fallback if marked fails to load (offline/CDN blocked) — prevent ReferenceError
        if (typeof marked === 'undefined') {
            window.marked = { parse: function(t){ return t.replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/\n/g,'<br>'); } };
            console.warn('[Ashen] marked CDN failed, using plaintext fallback');
        }
    </script>
    <!-- Prism.js for Syntax Highlighting -->
    <link href="https://cdnjs.cloudflare.com/ajax/libs/prism/1.29.0/themes/prism-okaidia.min.css" rel="stylesheet" />
    <script src="https://cdnjs.cloudflare.com/ajax/libs/prism/1.29.0/prism.min.js"></script>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/prism/1.29.0/components/prism-python.min.js"></script>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/prism/1.29.0/components/prism-json.min.js"></script>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/prism/1.29.0/components/prism-bash.min.js"></script>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/prism/1.29.0/components/prism-markdown.min.js"></script>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/prism/1.29.0/components/prism-javascript.min.js"></script>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/prism/1.29.0/components/prism-typescript.min.js"></script>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/prism/1.29.0/components/prism-css.min.js"></script>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/prism/1.29.0/components/prism-yaml.min.js"></script>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/prism/1.29.0/components/prism-markup.min.js"></script>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/prism/1.29.0/components/prism-go.min.js"></script>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/prism/1.29.0/components/prism-rust.min.js"></script>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/prism/1.29.0/components/prism-c.min.js"></script>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/prism/1.29.0/components/prism-cpp.min.js"></script>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/prism/1.29.0/components/prism-java.min.js"></script>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/prism/1.29.0/components/prism-csharp.min.js"></script>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/prism/1.29.0/components/prism-ruby.min.js"></script>
    <!-- prism-php removed - broken dependency -->
    <script src="https://cdnjs.cloudflare.com/ajax/libs/prism/1.29.0/components/prism-sql.min.js"></script>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/prism/1.29.0/components/prism-xml.min.js"></script>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/prism/1.29.0/components/prism-toml.min.js"></script>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/prism/1.29.0/components/prism-lua.min.js"></script>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/prism/1.29.0/components/prism-swift.min.js"></script>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/prism/1.29.0/components/prism-kotlin.min.js"></script>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/prism/1.29.0/components/prism-powershell.min.js"></script>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/prism/1.29.0/components/prism-docker.min.js"></script>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/prism/1.29.0/components/prism-git.min.js"></script>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/prism/1.29.0/components/prism-regex.min.js"></script>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/prism/1.29.0/components/prism-hcl.min.js"></script>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/prism/1.29.0/components/prism-protobuf.min.js"></script>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/prism/1.29.0/components/prism-graphql.min.js"></script>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/prism/1.29.0/components/prism-ini.min.js"></script>
    <link href="https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;700&amp;display=swap" rel="stylesheet">
    <style>
        body { background-color: #050509; color: #e2e8f0; font-family: 'JetBrains Mono', monospace; }
        .scanline {
            background: linear-gradient(to bottom, rgba(255,255,255,0), rgba(255,255,255,0) 50%, rgba(0, 0, 0, 0.3) 50%, rgba(0, 0, 0, 0.3));
            background-size: 100% 4px;
            pointer-events: none;
        }
        .chat-container::-webkit-scrollbar { width: 6px; }
        .chat-container::-webkit-scrollbar-thumb { background: #1e293b; border-radius: 3px; }
        .neon-border { box-shadow: 0 0 15px rgba(99, 102, 241, 0.15); }
    </style>
</head>
<body class="h-screen flex flex-col relative overflow-hidden">
    <!-- Scanlines Overlay -->
    <div id="scanline-overlay" class="absolute inset-0 scanline z-50 opacity-40"></div>

    <!-- Header -->
    <header class="bg-slate-950 border-b border-indigo-900/40 px-6 py-3 flex items-center justify-between shadow-2xl z-10">
        <div class="flex items-center space-x-3">
            <div class="w-3 h-3 bg-cyan-400 rounded-full animate-ping"></div>
            <h1 class="text-lg font-bold tracking-wider text-cyan-400">ASHEN AI <span class="text-xs px-2 py-0.5 bg-indigo-950 text-indigo-300 rounded border border-indigo-700/50">LOCAL AI HUB</span></h1>
        </div>
        <div class="flex items-center space-x-3 text-xs">
            <select id="persona-select" onchange="changePersona(this.value)" class="bg-slate-900 text-indigo-300 border border-indigo-800/60 rounded px-2 py-1 focus:outline-none">
                <option value="ashen_ai_agent">Persona: Ashen AI Agent</option>
                <option value="code_architect">Persona: Code Architect</option>
                <option value="cyber_companion">Persona: Cyber Companion</option>
            </select>
            <button onclick="createNewSession()" class="px-2.5 py-1 bg-violet-950/60 hover:bg-violet-900/60 text-violet-300 rounded border border-violet-800/60 transition" title="Start a new chat session">📝 New Chat</button>
            <button onclick="toggleSessionsPanel(true)" class="px-2.5 py-1 bg-blue-950/60 hover:bg-blue-900/60 text-blue-300 rounded border border-blue-800/60 transition" title="View past chat sessions">💬 Sessions</button>
            <button onclick="toggleModelModal(true)" class="px-2.5 py-1 bg-cyan-950/60 hover:bg-cyan-900/60 text-cyan-300 rounded border border-cyan-800/60 transition">Model Hub</button>
            <button onclick="toggleSwarmModal(true)" class="px-2.5 py-1 bg-orange-950/60 hover:bg-orange-900/60 text-orange-300 rounded border border-orange-800/60 transition" title="Swarm: spawn multiple subagents from selected model">🐝 Swarm</button>
            <button onclick="toggleCouncilModal(true)" class="px-2.5 py-1 bg-violet-950/60 hover:bg-violet-900/60 text-violet-300 rounded border border-violet-800/60 transition" title="Council: critics vote & suggest changes before final answer">⚖️ Council</button>
            <button onclick="toggleSelfImproveModal(true)" class="px-2.5 py-1 bg-amber-950/60 hover:bg-amber-900/60 text-amber-300 rounded border border-amber-800/60 transition" title="Self-improvement: feedback, auto-tune, and learning log">🧬 Self-Improve</button>
            <button onclick="loadWorkspaceDir('')" class="px-2.5 py-1 bg-emerald-950/60 hover:bg-emerald-900/60 text-emerald-300 rounded border border-emerald-800/60 transition">📁 Workspace</button>
            <button onclick="toggleSettingsModal(true)" class="px-2.5 py-1 bg-indigo-950/60 hover:bg-indigo-900/60 text-indigo-300 rounded border border-indigo-800/60 transition">⚙️ Settings</button>
            <button onclick="toggleCoT()" id="cot-toggle-btn" class="px-2.5 py-1 bg-cyan-950/60 hover:bg-cyan-900/60 text-cyan-300 rounded border border-cyan-800/60 transition" title="Toggle Chain-of-Thought visibility per-model">🧠 CoT: ON</button>
            <button onclick="toggleScanlines()" id="scanline-btn" class="px-2.5 py-1 bg-slate-900 hover:bg-slate-800 text-slate-300 rounded border border-slate-700 transition">CRT FX: ON</button>
            <button onclick="clearHistory()" class="px-2.5 py-1 bg-red-950/60 hover:bg-red-900/60 text-red-300 rounded border border-red-800/60 transition">Purge</button>
            <button onclick="exportChat()" class="px-2.5 py-1 bg-slate-900 hover:bg-slate-800 text-indigo-300 rounded border border-indigo-800/60 transition">Export</button>
        </div>
    </header>

    <!-- Model Hub Modal -->
    <div id="model-modal" class="fixed inset-0 bg-slate-950/80 backdrop-blur-sm z-50 hidden flex items-center justify-center p-4">
        <div class="bg-slate-900 border border-cyan-500/50 rounded-xl w-full max-w-4xl p-6 shadow-2xl space-y-6 max-h-[90vh] overflow-y-auto">
            <div class="flex justify-between items-center border-b border-slate-800 pb-3">
                <h2 class="text-base font-bold text-cyan-400">Ashen AI Model Hub &amp; Checkpoint Manager</h2>
                <button onclick="toggleModelModal(false)" class="text-slate-400 hover:text-white font-bold">✕</button>
            </div>
            
            <div class="space-y-6">
                <!-- Tab Selector -->
                <div class="flex space-x-2 border-b border-slate-800 pb-2 justify-between items-center">
                    <div class="flex space-x-2">
                        <button onclick="switchTab('local')" id="tab-local" class="px-3 py-1.5 rounded-lg text-xs font-semibold bg-cyan-950 text-cyan-300 border border-cyan-800/60 transition">Local Checkpoints</button>
                        <button onclick="switchTab('hf')" id="tab-hf" class="px-3 py-1.5 rounded-lg text-xs font-semibold bg-slate-800 text-slate-400 hover:text-white transition">Hugging Face Hub (All Models)</button>
                        <button onclick="switchTab('upload')" id="tab-upload" class="px-3 py-1.5 rounded-lg text-xs font-semibold bg-slate-800 text-slate-400 hover:text-white transition">Upload Model</button>
                    </div>
                    <button onclick="scanPcModels()" class="px-3 py-1.5 bg-indigo-950 hover:bg-indigo-900 text-indigo-200 rounded-lg text-xs font-semibold border border-indigo-700/60 transition">🔍 Scan PC for Models</button>
                </div>

                <!-- Local Tab -->
                <div id="section-local" class="space-y-3">
                    <h3 class="text-xs font-semibold text-indigo-300 uppercase tracking-wider">Installed Checkpoints &amp; GGUF Files</h3>
                    <div id="model-list" class="bg-slate-950 border border-slate-800 rounded-lg p-3 max-h-48 overflow-y-auto space-y-2 text-xs font-mono">
                        <div class="text-slate-500">Scanning local checkpoints...</div>
                    </div>
                </div>

                <!-- Hugging Face Tab -->
                <div id="section-hf" class="space-y-3 hidden">
                    <div class="flex space-x-2">
                        <input type="text" id="hf-search-input" placeholder="Search any Hugging Face model (e.g. Qwen, Llama, DeepSeek, GGUF)..." class="flex-1 bg-slate-950 border border-slate-700 rounded-lg p-2.5 text-xs text-cyan-100 placeholder-slate-500 focus:outline-none focus:border-cyan-500 font-mono">
                        <button onclick="searchHfModels()" class="px-4 py-2 bg-cyan-600 hover:bg-cyan-500 text-slate-950 font-bold text-xs rounded-lg transition shrink-0">SEARCH HF</button>
                    </div>
                    <div id="hf-model-list" class="bg-slate-950 border border-slate-800 rounded-lg p-3 max-h-72 overflow-y-auto space-y-3 text-xs">
                        <div class="text-slate-500">Type a query or search to list Hugging Face models...</div>
                    </div>
                </div>

                <!-- Upload Tab -->
                <div id="section-upload" class="space-y-3 hidden">
                    <h3 class="text-xs font-semibold text-indigo-300 uppercase tracking-wider">Upload Custom Model / GGUF File</h3>
                    <div class="flex items-center space-x-3">
                        <input type="file" id="model-file-input" accept=".pk1,.pt,.pth,.gguf" class="block w-full text-xs text-slate-400 file:mr-4 file:py-2 file:px-4 file:rounded-lg file:border-0 file:text-xs file:font-semibold file:bg-cyan-950 file:text-cyan-300 hover:file:bg-cyan-900 cursor-pointer border border-slate-700 rounded-lg bg-slate-950">
                        <button onclick="uploadModel()" class="bg-cyan-600 hover:bg-cyan-500 text-slate-950 px-4 py-2 rounded-lg font-bold text-xs shrink-0 transition">UPLOAD</button>
                    </div>
                </div>

                <div id="upload-status" class="text-xs text-cyan-400 font-mono"></div>
            </div>
        </div>
    </div>

    <!-- Model Settings Modal (LM Studio Style) -->
    <div id="settings-modal" class="fixed inset-0 bg-slate-950/80 backdrop-blur-sm z-50 hidden flex items-center justify-center p-4">
        <div class="bg-slate-900 border border-indigo-500/50 rounded-xl w-full max-w-2xl p-6 shadow-2xl space-y-6 max-h-[90vh] overflow-y-auto">
            <div class="flex justify-between items-center border-b border-slate-800 pb-3">
                <h2 class="text-base font-bold text-indigo-300">⚙️ LM Studio-Style Model &amp; Inference Settings</h2>
                <button onclick="toggleSettingsModal(false)" class="text-slate-400 hover:text-white font-bold">✕</button>
            </div>

            <div class="space-y-4 text-xs">
                <!-- Temperature -->
                <div class="space-y-1">
                    <div class="flex justify-between">
                        <span class="text-slate-300 font-medium">Temperature</span>
                        <span id="temp-val" class="text-cyan-400 font-mono">0.7</span>
                    </div>
                    <input type="range" id="setting-temp" min="0.0" max="2.0" step="0.05" value="0.7" oninput="document.getElementById('temp-val').textContent=this.value" class="w-full accent-cyan-500 bg-slate-950">
                </div>

                <!-- Top-K -->
                <div class="space-y-1">
                    <div class="flex justify-between">
                        <span class="text-slate-300 font-medium">Top-K</span>
                        <span id="topk-val" class="text-cyan-400 font-mono">40</span>
                    </div>
                    <input type="range" id="setting-topk" min="1" max="100" step="1" value="40" oninput="document.getElementById('topk-val').textContent=this.value" class="w-full accent-cyan-500 bg-slate-950">
                </div>

                <!-- Top-P -->
                <div class="space-y-1">
                    <div class="flex justify-between">
                        <span class="text-slate-300 font-medium">Top-P (Nucleus Sampling)</span>
                        <span id="topp-val" class="text-cyan-400 font-mono">0.9</span>
                    </div>
                    <input type="range" id="setting-topp" min="0.0" max="1.0" step="0.05" value="0.9" oninput="document.getElementById('topp-val').textContent=this.value" class="w-full accent-cyan-500 bg-slate-950">
                </div>

                <!-- Max Output Tokens -->
                <div class="space-y-1">
                    <div class="flex justify-between">
                        <span class="text-slate-300 font-medium">Max Output Tokens</span>
                        <span id="tokens-val" class="text-cyan-400 font-mono">250</span>
                    </div>
                    <input type="range" id="setting-tokens" min="50" max="1024" step="25" value="250" oninput="document.getElementById('tokens-val').textContent=this.value" class="w-full accent-cyan-500 bg-slate-950">
                </div>

                <!-- Context Length -->
                <div class="space-y-1">
                    <span class="text-slate-300 font-medium">Context Length (Tokens)</span>
                    <select id="setting-context" class="w-full bg-slate-950 border border-slate-700 rounded-lg p-2 text-xs text-cyan-200">
                        <option value="2048">2,048 Tokens</option>
                        <option value="4096">4,096 Tokens</option>
                        <option value="8192" selected>8,192 Tokens</option>
                        <option value="16384">16,384 Tokens</option>
                        <option value="32768">32,768 Tokens</option>
                    </select>
                </div>

                <!-- GPU Offload Layers Slider -->
                <div class="space-y-1 pt-2 border-t border-slate-800">
                    <div class="flex justify-between">
                        <div>
                            <span class="text-slate-300 font-medium">GPU Offload Layers (CUDA VRAM)</span>
                            <div class="text-slate-500 text-[10px]">Number of transformer layers offloaded to GPU</div>
                        </div>
                        <span id="gpu-layers-val" class="text-cyan-400 font-mono">16</span>
                    </div>
                    <input type="range" id="setting-gpu-layers" min="0" max="32" step="1" value="16" oninput="document.getElementById('gpu-layers-val').textContent=this.value" class="w-full accent-cyan-500 bg-slate-950">
                </div>

                <!-- Draft Model Settings -->
                <div class="pt-4 border-t border-slate-800">
                    <h3 class="text-sm font-semibold text-violet-300 mb-3 flex items-center gap-2">
                        <span class="inline-block w-2 h-2 rounded-full bg-violet-400"></span>
                        📝 Draft Model Configuration
                    </h3>
                    
                    <div class="space-y-3">
                        <!-- Enable Speculative Decoding -->
                        <div class="flex items-center justify-between p-3 bg-slate-950/50 rounded-lg border border-slate-800">
                            <div>
                                <div class="text-slate-300 font-medium">Enable Speculative Decoding</div>
                                <div class="text-slate-500 text-[10px]">Use draft model for faster token generation via speculative decoding</div>
                            </div>
                            <label class="relative inline-flex items-center cursor-pointer">
                                <input type="checkbox" id="setting-draft-enabled" onchange="toggleDraftSettings()" class="sr-only peer">
                                <div class="w-11 h-6 bg-slate-700 peer-focus:outline-none rounded-full peer peer-checked:after:translate-x-full peer-checked:after:border-white after:content-[''] after:absolute after:top-[2px] after:left-[2px] after:bg-white after:rounded-full after:h-5 after:w-5 after:transition-all peer-checked:bg-violet-600"></div>
                            </label>
                        </div>

                        <!-- Draft Temperature (hidden until enabled) -->
                        <div id="draft-temp-section" class="hidden space-y-1 pl-4 border-l-2 border-violet-800">
                            <div class="flex justify-between">
                                <span class="text-slate-300 font-medium">Draft Model Temperature</span>
                                <span id="draft-temp-val" class="text-violet-400 font-mono">0.6</span>
                            </div>
                            <input type="range" id="setting-draft-temp" min="0.1" max="1.0" step="0.05" value="0.6" oninput="document.getElementById('draft-temp-val').textContent=this.value" class="w-full accent-violet-500 bg-slate-950" disabled>
                            <div class="text-slate-500 text-[10px]">Lower temperature = more deterministic drafts</div>
                        </div>

                        <!-- Draft Model Info -->
                        <div class="p-3 bg-amber-950/30 rounded-lg border border-amber-800/50">
                            <div class="text-amber-300 text-xs font-semibold mb-1">ℹ️ Requirements</div>
                            <ul class="text-[10px] text-slate-400 space-y-0.5 list-disc list-inside">
                                <li>Draft model file: <code class="text-violet-300">ashen_gpt_model_draft.pk1</code></li>
                                <li>Recommended: smaller/faster version of main model</li>
                                <li>Load manually via Model Hub → Upload or place in project directory</li>
                            </ul>
                        </div>
                    </div>
                </div>

                <!-- Low-end GPU Optimization Settings -->
                <div class="pt-4 border-t border-slate-800">
                    <h3 class="text-sm font-semibold text-emerald-300 mb-3 flex items-center gap-2">
                        <span class="inline-block w-2 h-2 rounded-full bg-emerald-400"></span>
                        🔧 Low-End GPU Optimization
                    </h3>
                    
                    <div class="space-y-3">
                        <!-- Low-End Mode Toggle -->
                        <div class="flex items-center justify-between p-3 bg-slate-950/50 rounded-lg border border-slate-800">
                            <div>
                                <div class="text-slate-300 font-medium">Low-End GPU Mode</div>
                                <div class="text-slate-500 text-[10px]">Aggressive memory optimizations for GPUs with &lt;8GB VRAM</div>
                            </div>
                            <label class="relative inline-flex items-center cursor-pointer">
                                <input type="checkbox" id="setting-low-end" onchange="updateLowEndSettings()" class="sr-only peer">
                                <div class="w-11 h-6 bg-slate-700 peer-focus:outline-none rounded-full peer peer-checked:after:translate-x-full peer-checked:after:border-white after:content-[''] after:absolute after:top-[2px] after:left-[2px] after:bg-white after:rounded-full after:h-5 after:w-5 after:transition-all peer-checked:bg-emerald-600"></div>
                            </label>
                        </div>

                        <!-- Precision Setting (hidden until low-end mode enabled) -->
                        <div id="low-end-settings-section" class="hidden space-y-3 pl-4 border-l-2 border-emerald-800">
                            <!-- Precision Selector -->
                            <div class="space-y-1">
                                <span class="text-slate-300 font-medium">Model Precision</span>
                                <select id="setting-precision" class="w-full bg-slate-950 border border-slate-700 rounded-lg p-2 text-xs text-cyan-200">
                                    <option value="fp32">FP32 (Standard - Best Quality)</option>
                                    <option value="fp16" selected>FP16 (Half - Good Performance)</option>
                                    <option value="bf16">BF16 (BFloat16 - Balanced)</option>
                                </select>
                                <div class="text-slate-500 text-[10px]">FP16/BF16 reduces VRAM usage by ~50%</div>
                            </div>

                            <!-- CPU Offload Sliders -->
                            <div class="space-y-1 pt-2 border-t border-slate-800">
                                <div class="flex justify-between">
                                    <div>
                                        <span class="text-slate-300 font-medium">CPU Offload Layers</span>
                                        <div class="text-slate-500 text-[10px]">Move layers to system RAM (slower but saves VRAM)</div>
                                    </div>
                                    <span id="cpu-offload-val" class="text-emerald-400 font-mono">0</span>
                                </div>
                                <input type="range" id="setting-cpu-offload" min="0" max="16" step="1" value="0" oninput="document.getElementById('cpu-offload-val').textContent=this.value" class="w-full accent-emerald-500 bg-slate-950">
                            </div>

                            <!-- Optimizations Info Card -->
                            <div class="p-3 bg-blue-950/30 rounded-lg border border-blue-800/50">
                                <div class="text-blue-300 text-xs font-semibold mb-1">💡 When to Enable</div>
                                <ul class="text-[10px] text-slate-400 space-y-0.5 list-disc list-inside">
                                    <li>&lt; 8GB VRAM → Enable low-end mode + FP16</li>
                                    <li>&lt; 4GB VRAM → Enable low-end mode + offload 4-8 layers</li>
                                    <li>Inference will be slower but functional</li>
                                </ul>
                            </div>
                        </div>
                    </div>
                </div>

                <!-- Chain of Thought Visibility -->
                <div class="pt-4 border-t border-slate-800">
                    <h3 class="text-sm font-semibold text-cyan-300 mb-3 flex items-center gap-2">
                        <span class="inline-block w-2 h-2 rounded-full bg-cyan-400"></span>
                        🧠 Chain-of-Thought Display
                    </h3>
                    <div class="flex items-center justify-between p-3 bg-slate-950/50 rounded-lg border border-slate-800">
                        <div>
                            <div class="text-slate-300 font-medium">Show Chain of Thought from Selected Model</div>
                            <div class="text-slate-500 text-[10px]">Visible reasoning per-model: expanded CoT panel attached to every reply (labeled with active model name)</div>
                        </div>
                        <label class="relative inline-flex items-center cursor-pointer">
                            <input type="checkbox" id="setting-show-cot" checked onchange="syncCotToggle()" class="sr-only peer">
                            <div class="w-11 h-6 bg-slate-700 peer-focus:outline-none rounded-full peer peer-checked:after:translate-x-full peer-checked:after:border-white after:content-[''] after:absolute after:top-[2px] after:left-[2px] after:bg-white after:rounded-full after:h-5 after:w-5 after:transition-all peer-checked:bg-cyan-600"></div>
                        </label>
                    </div>
                    <div class="text-slate-500 text-[10px] mt-2">Header button 🧠 CoT toggles this without opening Settings. Persists to <code class="text-cyan-300">settings.json</code> as <code class="text-cyan-300">show_chain_of_thought</code>.</div>
                </div>

                <!-- Auto Swarm + Council enrichment -->
                <div class="pt-4 border-t border-slate-800">
                    <h3 class="text-sm font-semibold text-cyan-300 mb-3 flex items-center gap-2">
                        <span class="inline-block w-2 h-2 rounded-full bg-cyan-400"></span>
                        🐝 Auto Swarm + Council
                    </h3>
                    <div class="flex items-center justify-between p-3 bg-slate-950/50 rounded-lg border border-slate-800">
                        <div>
                            <div class="text-slate-300 font-medium">Enrich every turn with Swarm + Council</div>
                            <div class="text-slate-500 text-[10px]">Silently consults a lightweight multi-agent swarm (2 agents) + council (2 drafts · 2 critics) before each reply, biasing the model's chain-of-thought and answer. Increases latency significantly — off by default.</div>
                        </div>
                        <label class="relative inline-flex items-center cursor-pointer">
                            <input type="checkbox" id="setting-auto-swarm-council" class="sr-only peer">
                            <div class="w-11 h-6 bg-slate-700 peer-focus:outline-none rounded-full peer peer-checked:after:translate-x-full peer-checked:after:border-white after:content-[''] after:absolute after:top-[2px] after:left-[2px] after:bg-white after:rounded-full after:h-5 after:w-5 after:transition-all peer-checked:bg-cyan-600"></div>
                        </label>
                    </div>
                    <div class="flex items-center justify-between p-3 bg-slate-950/50 rounded-lg border border-slate-800">
                        <div>
                            <div class="text-slate-300 font-medium">🌐 Auto Web Research + Sources</div>
                            <div class="text-slate-500 text-[10px]">Before each reply, runs an autonomous web/deep-research pass and feeds the findings to the model so it answers from real, cited sources. Shows a Sources list under the answer. Off by default.</div>
                        </div>
                        <label class="relative inline-flex items-center cursor-pointer">
                            <input type="checkbox" id="setting-auto-web-research" class="sr-only peer">
                            <div class="w-11 h-6 bg-slate-700 peer-focus:outline-none rounded-full peer peer-checked:after:translate-x-full peer-checked:after:border-white after:content-[''] after:absolute after:top-[2px] after:left-[2px] after:bg-white after:rounded-full after:h-5 after:w-5 after:transition-all peer-checked:bg-cyan-600"></div>
                        </label>
                    </div>
                </div>

                <div class="pt-4 flex justify-end space-x-3 border-t border-slate-800">
                    <button onclick="toggleSettingsModal(false)" class="px-4 py-2 bg-slate-800 hover:bg-slate-700 text-slate-300 rounded-lg font-bold transition">CANCEL</button>
                    <button onclick="saveSettings()" class="px-5 py-2 bg-indigo-600 hover:bg-indigo-500 text-white rounded-lg font-bold transition">SAVE SETTINGS</button>
                </div>
                <div id="settings-status" class="text-xs text-emerald-400 font-mono text-center"></div>
            </div>
        </div>
    </div>

    <!-- Sessions Panel -->
    <div id="sessions-panel" class="fixed inset-y-0 left-0 w-80 bg-slate-950/95 backdrop-blur-sm z-40 border-r border-blue-900/40 -translate-x-full transition-transform duration-300 ease-in-out flex flex-col">
        <div class="p-4 border-b border-blue-900/30 flex items-center justify-between">
            <h2 class="text-sm font-bold text-blue-400 uppercase tracking-wider">💬 Chat Sessions</h2>
            <button onclick="toggleSessionsPanel(false)" class="text-slate-400 hover:text-white font-bold text-lg">✕</button>
        </div>

        <!-- New Session Button -->
        <div class="p-3 border-b border-slate-800">
            <button onclick="createNewSession()" class="w-full px-3 py-2 bg-violet-950/60 hover:bg-violet-900/60 text-violet-300 rounded-lg text-xs font-semibold border border-violet-800/60 transition flex items-center justify-center gap-2">
                <span>📝</span> New Chat Session
            </button>
        </div>

        <!-- Session List -->
        <div id="session-list" class="flex-1 overflow-y-auto p-2 space-y-1">
            <div class="text-slate-500 text-xs text-center py-4">Loading sessions...</div>
        </div>

        <!-- Current Session Info -->
        <div class="p-3 border-t border-slate-800 text-[10px] text-slate-500 text-center">
            <span id="current-session-label">No active session</span>
        </div>
    </div>

    <!-- Sessions Panel Overlay -->
    <div id="sessions-overlay" class="fixed inset-0 bg-black/30 z-30 hidden" onclick="toggleSessionsPanel(false)"></div>

    <!-- Self-Improvement Modal -->
    <div id="self-improve-modal" class="fixed inset-0 bg-slate-950/80 backdrop-blur-sm z-50 hidden flex items-center justify-center p-4">
        <div class="bg-slate-900 border border-amber-500/50 rounded-xl w-full max-w-3xl p-6 shadow-2xl space-y-5 max-h-[90vh] overflow-y-auto">
            <div class="flex justify-between items-center border-b border-slate-800 pb-3">
                <h2 class="text-base font-bold text-amber-400">🧬 Self-Improvement — Learning Loop</h2>
                <button onclick="toggleSelfImproveModal(false)" class="text-slate-400 hover:text-white font-bold">✕</button>
            </div>
            <div class="grid grid-cols-3 gap-3 text-xs">
                <div class="bg-slate-950 border border-slate-800 rounded-lg p-3 text-center">
                    <div class="text-slate-500 uppercase tracking-wider text-[10px]">Feedback</div>
                    <div class="text-lg font-bold text-cyan-400" id="si-stat-feedback">—</div>
                    <div class="text-[10px] text-slate-500"><span class="text-emerald-400" id="si-stat-up">0</span> 👍 · <span class="text-red-400" id="si-stat-down">0</span> 👎</div>
                </div>
                <div class="bg-slate-950 border border-slate-800 rounded-lg p-3 text-center">
                    <div class="text-slate-500 uppercase tracking-wider text-[10px]">Gibberish Fixes</div>
                    <div class="text-lg font-bold text-amber-400" id="si-stat-gib">—</div>
                    <div class="text-[10px] text-slate-500" id="si-stat-gib-rate">rate —</div>
                </div>
                <div class="bg-slate-950 border border-slate-800 rounded-lg p-3 text-center">
                    <div class="text-slate-500 uppercase tracking-wider text-[10px]">Auto-Tunes</div>
                    <div class="text-lg font-bold text-violet-400" id="si-stat-tunes">—</div>
                    <div class="text-[10px] text-slate-500" id="si-stat-corr">corrections —</div>
                </div>
            </div>
            <div class="flex gap-2">
                <button onclick="runSelfImproveAnalyze()" class="px-3 py-1.5 bg-cyan-950 hover:bg-cyan-900 text-cyan-300 rounded-lg text-xs font-semibold border border-cyan-800/60 transition">🔍 Analyze</button>
                <button onclick="runSelfImproveAutoTune()" class="px-3 py-1.5 bg-amber-950 hover:bg-amber-900 text-amber-300 rounded-lg text-xs font-semibold border border-amber-800/60 transition">⚙️ Auto-Tune</button>
                <button onclick="runSelfImproveBenchmark()" class="px-3 py-1.5 bg-violet-950 hover:bg-violet-900 text-violet-300 rounded-lg text-xs font-semibold border border-violet-800/60 transition">🧪 Benchmark</button>
                <span id="si-action-status" class="text-xs text-slate-400 font-mono self-center ml-2"></span>
            </div>
            <div>
                <h3 class="text-xs font-semibold text-amber-300 uppercase tracking-wider mb-2">Suggestions & Hints</h3>
                <div id="si-suggestions" class="bg-slate-950 border border-slate-800 rounded-lg p-3 max-h-32 overflow-y-auto text-xs font-mono space-y-1"><div class="text-slate-500">No suggestions yet — chat and give feedback to generate hints.</div></div>
            </div>
            <div>
                <h3 class="text-xs font-semibold text-slate-300 uppercase tracking-wider mb-2">Recent Improvement Log</h3>
                <div id="si-log" class="bg-slate-950 border border-slate-800 rounded-lg p-3 max-h-40 overflow-y-auto text-[11px] font-mono space-y-1"><div class="text-slate-500">No entries yet.</div></div>
            </div>
            <div>
                <h3 class="text-xs font-semibold text-slate-300 uppercase tracking-wider mb-2">Recent Feedback</h3>
                <div id="si-feedback" class="bg-slate-950 border border-slate-800 rounded-lg p-3 max-h-32 overflow-y-auto text-[11px] font-mono space-y-1"><div class="text-slate-500">No feedback yet — use 👍👎 on messages.</div></div>
            </div>
        </div>
    </div>

    <!-- Correction Modal (for 👎) -->
    <div id="correction-modal" class="fixed inset-0 bg-slate-950/80 backdrop-blur-sm z-50 hidden flex items-center justify-center p-4">
        <div class="bg-slate-900 border border-red-500/50 rounded-xl w-full max-w-lg p-6 shadow-2xl space-y-4">
            <h2 class="text-sm font-bold text-red-400">👎 What should the answer have been?</h2>
            <p class="text-xs text-slate-400">Your correction will be stored and used to self-improve via critique regeneration. Prompt: <span id="correction-prompt-preview" class="text-amber-300 font-mono break-words"></span></p>
            <textarea id="correction-input" rows="3" placeholder="Type the correct answer or guidance (e.g. 'Paris is correct, not London' or 'use a haiku, 5-7-5')" class="w-full bg-slate-950 border border-slate-700 rounded-lg p-3 text-xs text-cyan-100 placeholder-slate-500 focus:outline-none focus:border-amber-500"></textarea>
            <div class="flex justify-end gap-2">
                <button onclick="closeCorrectionModal()" class="px-4 py-2 bg-slate-800 hover:bg-slate-700 text-slate-300 rounded-lg font-bold text-xs transition">CANCEL</button>
                <button onclick="submitCorrection(true)" class="px-4 py-2 bg-amber-600 hover:bg-amber-500 text-slate-950 rounded-lg font-bold text-xs transition">SAVE + REGENERATE</button>
                <button onclick="submitCorrection(false)" class="px-4 py-2 bg-slate-700 hover:bg-slate-600 text-amber-300 rounded-lg font-bold text-xs transition">SAVE ONLY</button>
            </div>
            <div id="correction-status" class="text-xs text-emerald-400 font-mono"></div>
        </div>
    </div>

    <!-- Swarm Modal -->
    <div id="swarm-modal" class="fixed inset-0 bg-slate-950/80 backdrop-blur-sm z-50 hidden flex items-center justify-center p-4">
        <div class="bg-slate-900 border border-orange-500/50 rounded-xl w-full max-w-5xl p-6 shadow-2xl space-y-5 max-h-[92vh] overflow-y-auto">
            <div class="flex justify-between items-center border-b border-slate-800 pb-3">
                <h2 class="text-base font-bold text-orange-400">🐝 Swarm — Multi-Agent from <span id="swarm-model-name" class="text-amber-300 font-mono text-xs">selected model</span></h2>
                <button onclick="toggleSwarmModal(false)" class="text-slate-400 hover:text-white font-bold">✕</button>
            </div>
            <p class="text-xs text-slate-400">Spawns <span class="text-orange-300">multiple subagents out of the selected model</span> with different roles (Researcher, Coder, Critic, Planner, Executor, Analyst). They work in <code class="text-cyan-300">parallel</code>, <code class="text-cyan-300">divide</code> or <code class="text-cyan-300">debate</code> mode, then a Synthesizer merges their drafts into one superior answer. Uses your current <code class="text-amber-300">settings.json</code> model.</p>
            <div class="grid grid-cols-3 gap-3 text-xs">
                <div>
                    <label class="text-slate-300 font-medium">Task (difficult prompt)</label>
                    <textarea id="swarm-task" rows="3" placeholder="e.g. Design a PyTorch MoE transformer with load balancing, explain all parts, and give a minimal code skeleton..." class="w-full bg-slate-950 border border-slate-700 rounded-lg p-3 text-xs text-cyan-100 placeholder-slate-500 focus:outline-none focus:border-orange-500"></textarea>
                </div>
                <div class="space-y-3">
                    <div>
                        <label class="text-slate-300 font-medium">Agents: <span id="swarm-num-label" class="text-orange-400 font-mono">3</span></label>
                        <input type="range" id="swarm-num" min="2" max="6" value="3" oninput="document.getElementById('swarm-num-label').textContent=this.value" class="w-full accent-orange-500">
                        <div class="text-[10px] text-slate-500">2–6 subagents, each a copy of selected model with different role.</div>
                    </div>
                    <div>
                        <label class="text-slate-300 font-medium">Mode</label>
                        <select id="swarm-mode" class="w-full bg-slate-950 border border-slate-700 rounded-lg p-2 text-xs text-orange-200">
                            <option value="parallel" selected>parallel — all solve full task independently</option>
                            <option value="divide">divide — split task into sub-tasks</option>
                            <option value="debate">debate — sequential critique chain</option>
                        </select>
                    </div>
                    <button onclick="runSwarm()" id="swarm-run-btn" class="w-full px-3 py-2 bg-orange-600 hover:bg-orange-500 text-slate-950 rounded-lg font-bold text-xs transition">🐝 SPAWN SWARM</button>
                    <div id="swarm-status" class="text-xs font-mono text-slate-400"></div>
                </div>
                <div class="bg-slate-950 border border-slate-800 rounded-lg p-3">
                    <div class="text-[10px] text-slate-500 uppercase tracking-wider mb-1">Active Roles Preview</div>
                    <div id="swarm-roles-preview" class="text-xs font-mono space-y-1 text-slate-300"><div class="text-slate-500">Will show roles for selected count.</div></div>
                    <div class="text-[10px] text-slate-500 mt-2">Model: <span id="swarm-model-path" class="text-amber-300 break-all"></span></div>
                </div>
            </div>
            <div id="swarm-results" class="space-y-3 hidden">
                <h3 class="text-xs font-semibold text-orange-300 uppercase tracking-wider">Synthesis — Final Swarm Answer</h3>
                <div id="swarm-synthesis" class="bg-slate-950 border border-orange-900/60 rounded-lg p-4 space-y-2"></div>
                <h3 class="text-xs font-semibold text-slate-300 uppercase tracking-wider">Per-Agent Drafts</h3>
                <div id="swarm-agents" class="grid md:grid-cols-2 gap-3"></div>
            </div>
        </div>
    </div>

    <!-- Council Modal -->
    <div id="council-modal" class="fixed inset-0 bg-slate-950/80 backdrop-blur-sm z-50 hidden flex items-center justify-center p-4">
        <div class="bg-slate-900 border border-violet-500/50 rounded-xl w-full max-w-6xl p-6 shadow-2xl space-y-5 max-h-[92vh] overflow-y-auto">
            <div class="flex justify-between items-center border-b border-slate-800 pb-3">
                <h2 class="text-base font-bold text-violet-400">⚖️ Council — Critics Vote & Refine from <span id="council-model-name" class="text-amber-300 font-mono text-xs">selected model</span></h2>
                <button onclick="toggleCouncilModal(false)" class="text-slate-400 hover:text-white font-bold">✕</button>
            </div>
            <p class="text-xs text-slate-400">Like Swarm but <span class="text-violet-300">critics vote and suggest changes before the final response</span>. Spawns <code class="text-violet-300">draft proposers</code> (like Swarm agents) then <code class="text-violet-300">critics</code> that score each draft 1–10 + one suggestion. Votes are tallied, the winner is refined by a Finalizer using the real critiques.</p>
            <div class="grid grid-cols-3 gap-3 text-xs">
                <div>
                    <label class="text-slate-300 font-medium">Task (difficult prompt)</label>
                    <textarea id="council-task" rows="3" placeholder="e.g. Compare MoE vs dense transformers, pick best for 7B on 4 GPUs, justify, and give PyTorch sketch..." class="w-full bg-slate-950 border border-slate-700 rounded-lg p-3 text-xs text-violet-100 placeholder-slate-500 focus:outline-none focus:border-violet-500"></textarea>
                </div>
                <div class="space-y-3">
                    <div class="grid grid-cols-2 gap-2">
                        <div>
                            <label class="text-slate-300 font-medium">Drafts: <span id="council-drafts-label" class="text-violet-400 font-mono">3</span></label>
                            <input type="range" id="council-drafts" min="2" max="5" value="3" oninput="document.getElementById('council-drafts-label').textContent=this.value; updateCouncilPreview()" class="w-full accent-violet-500">
                        </div>
                        <div>
                            <label class="text-slate-300 font-medium">Critics: <span id="council-critics-label" class="text-violet-400 font-mono">3</span></label>
                            <input type="range" id="council-critics" min="2" max="5" value="3" oninput="document.getElementById('council-critics-label').textContent=this.value; updateCouncilPreview()" class="w-full accent-violet-500">
                        </div>
                    </div>
                    <div class="text-[10px] text-slate-500">Total 5–10 model calls. Winner picked by picks + score sum.</div>
                    <button onclick="runCouncil()" id="council-run-btn" class="w-full px-3 py-2 bg-violet-600 hover:bg-violet-500 text-white rounded-lg font-bold text-xs transition">⚖️ CONVENE COUNCIL</button>
                    <div id="council-status" class="text-xs font-mono text-slate-400"></div>
                </div>
                <div class="bg-slate-950 border border-slate-800 rounded-lg p-3">
                    <div class="text-[10px] text-slate-500 uppercase tracking-wider mb-1">Council Preview</div>
                    <div id="council-preview" class="text-xs font-mono space-y-1 text-slate-300"><div class="text-slate-500">Will show draft + critic roles.</div></div>
                    <div class="text-[10px] text-slate-500 mt-2">Model: <span id="council-model-path" class="text-amber-300 break-all"></span></div>
                </div>
            </div>
            <div id="council-results" class="space-y-3 hidden">
                <h3 class="text-xs font-semibold text-violet-300 uppercase tracking-wider">Final Council Answer (refined winner + critiques)</h3>
                <div id="council-final" class="bg-slate-950 border border-violet-900/60 rounded-lg p-4 space-y-2"></div>
                <h3 class="text-xs font-semibold text-slate-300 uppercase tracking-wider">Votes & Suggestions</h3>
                <div id="council-votes" class="bg-slate-950/70 border border-slate-800 rounded-lg p-3 text-xs font-mono overflow-x-auto"></div>
                <div class="grid md:grid-cols-2 gap-3">
                    <div>
                        <h4 class="text-xs font-semibold text-slate-400 uppercase tracking-wider mb-2">Drafts</h4>
                        <div id="council-drafts" class="space-y-3"></div>
                    </div>
                    <div>
                        <h4 class="text-xs font-semibold text-slate-400 uppercase tracking-wider mb-2">Critics</h4>
                        <div id="council-critics" class="space-y-3"></div>
                    </div>
                </div>
            </div>
        </div>
    </div>

    <!-- Main Workspace -->
    <div class="flex-1 flex overflow-hidden z-10">
        <!-- Sidebar Telemetry & Quick Actions -->
        <aside class="w-80 bg-slate-950/80 border-r border-indigo-900/30 p-4 hidden md:flex flex-col justify-between">
            <div class="space-y-4">
                <h2 class="text-xs font-semibold text-cyan-500 uppercase tracking-widest">System Telemetry</h2>
                <div class="bg-slate-900/70 p-3 rounded-lg border border-slate-800 space-y-2 text-xs">
                    <div class="flex justify-between"><span>Model:</span><span class="text-indigo-400 font-bold" id="active-model-name">ashen_gpt_model.pk1</span></div>
                    <div class="flex justify-between"><span>Backend:</span><span class="text-emerald-400 font-bold">PyTorch CUDA</span></div>
                    <div class="flex justify-between"><span>Context Window:</span><span class="text-cyan-400 font-bold" id="sidebar-ctx">8,192 Tokens</span></div>
                    <div class="flex justify-between"><span>GPU Layers:</span><span class="text-amber-400 font-bold" id="sidebar-gpu">16 / 32</span></div>
                    <div class="flex flex-col gap-1 pt-1 border-t border-slate-800 mt-1">
                        <span class="text-slate-500">Workspace Context:</span>
                        <span class="text-emerald-300 font-bold truncate" id="workspace-context-label" title="Currently browsed directory injected into model prompt">📁 None</span>
                    </div>
                </div>

                <h2 class="text-xs font-semibold text-cyan-500 uppercase tracking-widest pt-2">Ashen AI Quick Actions</h2>
                <div class="grid grid-cols-2 gap-2">
                    <button onclick="sendQuickPrompt('Run pytest suite')" class="p-2 text-left text-xs bg-slate-900 hover:bg-indigo-950/50 text-indigo-300 rounded border border-indigo-900/50 transition">⚡ Run Tests</button>
                    <button onclick="sendQuickPrompt('Check git status and diff')" class="p-2 text-left text-xs bg-slate-900 hover:bg-indigo-950/50 text-indigo-300 rounded border border-indigo-900/50 transition">📦 Git Status</button>
                    <button onclick="sendQuickPrompt('List workspace files using glob')" class="p-2 text-left text-xs bg-slate-900 hover:bg-indigo-950/50 text-indigo-300 rounded border border-indigo-900/50 transition">🔍 File Glob</button>
                    <button onclick="sendQuickPrompt('Search codebase for reasoning engine')" class="p-2 text-left text-xs bg-slate-900 hover:bg-indigo-950/50 text-indigo-300 rounded border border-indigo-900/50 transition">🔎 Grep Search</button>
                    <button onclick="sendQuickPrompt('Search DuckDuckGo for PyTorch 2.0 features')" class="p-2 text-left text-xs bg-emerald-900/50 hover:bg-emerald-950/50 text-emerald-300 rounded border border-emerald-800/50 transition">🌐 Web Search</button>
                    <button onclick="sendQuickPrompt('Browse https://pytorch.org/docs/stable/index.html')" class="p-2 text-left text-xs bg-purple-900/50 hover:bg-purple-950/50 text-purple-300 rounded border border-purple-800/50 transition">📖 Browse URL</button>
                    <button onclick="sendQuickPrompt('Deep research on recent advances in transformer architectures')" class="p-2 text-left text-xs bg-cyan-900/50 hover:bg-cyan-950/50 text-cyan-300 rounded border border-cyan-800/50 transition">🔬 Deep Research</button>
                    <button onclick="sendQuickPrompt('Run full Ashen AI benchmark suite')" class="p-2 text-left text-xs bg-orange-900/50 hover:bg-orange-950/50 text-orange-300 rounded border border-orange-800/50 transition">🧪 Run Benchmark</button>
                </div>
            </div>

            <div class="text-[10px] text-slate-500 text-center pt-4 border-t border-slate-900">
                Ashen AI Hub &bull; Autonomous Agent Core
            </div>
        </aside>

        <!-- Chat Area -->
        <main class="flex-1 flex flex-col bg-[#08080f]">
            <div id="chat-container" class="flex-1 overflow-y-auto p-6 space-y-6 chat-container">
                <!-- Welcome Message -->
                <div class="flex items-start space-x-4 max-w-4xl">
                    <div class="w-8 h-8 rounded bg-cyan-600/20 border border-cyan-500 flex items-center justify-center font-bold text-xs text-cyan-400 shrink-0">Ω</div>
                    <div class="bg-slate-900/90 border border-indigo-900/60 rounded-xl p-4 text-slate-200 text-sm shadow-xl space-y-2 neon-border">
                        <p class="font-bold text-cyan-400">Ashen AI Cybernetic Core Online.</p>
                        <p class="text-xs text-slate-300">Initialized with Ashen GPT ~127M MoE model and full agentic tool suite. Access Model Hub, Workspace, or Settings above or enter your prompt below.</p>
                    </div>
                </div>
            </div>

            <!-- Input Bar -->
            <div class="p-4 bg-slate-950 border-t border-indigo-900/40">
                <div class="max-w-4xl mx-auto flex items-end space-x-3">
                    <div class="flex-1 bg-slate-900 border border-indigo-900/60 rounded-lg focus-within:border-cyan-500 transition">
                        <textarea id="user-input" rows="1" placeholder="Enter command or query for Ashen AI... (Shift+Enter for newline)" class="w-full bg-transparent p-3 text-cyan-100 placeholder-slate-500 text-xs focus:outline-none resize-none max-h-32"></textarea>
                    </div>
                    <button id="send-btn" onclick="sendMessage()" class="bg-cyan-600 hover:bg-cyan-500 text-slate-950 px-5 py-3 rounded-lg font-bold text-xs transition shadow-lg shrink-0 flex items-center space-x-2">
                        <span>EXECUTE</span>
                        <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M14 5l7 7m0 0l-7 7m7-7H3"></path></svg>
                    </button>
                </div>
            </div>
        </main>

        <!-- Right Workspace Explorer & Code Editor Panel (Always Open) -->
        <aside class="w-[480px] xl:w-[520px] bg-slate-950/90 border-l border-emerald-900/40 p-4 hidden lg:flex flex-col space-y-3 overflow-hidden shrink-0">
            <div class="flex justify-between items-center border-b border-slate-800 pb-2">
                <div class="flex items-center space-x-2">
                    <h2 class="text-xs font-bold text-emerald-400 uppercase tracking-wider">📁 Workspace Explorer</h2>
                </div>
                <div class="flex items-center space-x-2 text-xs">
                    <button onclick="setEditorMode('view')" id="mode-view-btn" class="px-2.5 py-1 rounded bg-emerald-950 text-emerald-300 border border-emerald-800">View</button>
                    <button onclick="setEditorMode('edit')" id="mode-edit-btn" class="px-2.5 py-1 rounded bg-slate-800 text-slate-400">Edit</button>
                    <button onclick="saveCurrentFile()" class="px-2.5 py-1 bg-emerald-600 hover:bg-emerald-500 text-slate-950 rounded font-bold transition">SAVE</button>
                </div>
            </div>

            <div class="flex items-center justify-between text-[11px] font-mono text-slate-400">
                <span id="current-dir-label" class="truncate">~ (Home)</span>
                <span id="current-file-path" class="text-cyan-300 truncate max-w-[200px]" title="No file selected">No file selected</span>
            </div>

            <!-- File Tree Sidebar -->
            <div class="h-44 bg-slate-900 border border-slate-800 rounded-lg p-2 overflow-y-auto space-y-1 font-mono text-xs flex flex-col shrink-0" id="file-tree">
                <div class="text-slate-500">Loading home directory...</div>
            </div>

            <!-- Code Editor Area -->
            <div class="flex-1 flex flex-col bg-slate-900 border border-slate-800 rounded-lg p-3 space-y-2 overflow-hidden">
                <!-- Syntax Highlighted Code Editor Container (View & Edit Mode) -->
                <div class="flex-1 relative bg-[#1a1a24] rounded-lg overflow-auto font-mono text-xs border border-slate-800" id="editor-container">
                    <pre class="p-3 m-0 bg-transparent pointer-events-none w-max min-w-full"><code id="highlighted-code" class="language-python"># Select a file to view syntax-highlighted code</code></pre>
                    <textarea id="code-editor" class="absolute inset-0 w-full h-full bg-transparent text-transparent caret-emerald-400 p-3 font-mono text-xs focus:outline-none resize-none leading-relaxed overflow-hidden" placeholder="Edit file code here..." readonly oninput="onEditorInput()"></textarea>
                </div>

                <div id="editor-status" class="text-xs text-emerald-400 font-mono"></div>
            </div>
        </aside>
    </div>

    <script>
        const textarea = document.getElementById('user-input');
        textarea.addEventListener('input', function() {
            this.style.height = 'auto';
            this.style.height = (this.scrollHeight) + 'px';
        });

        textarea.addEventListener('keydown', function(e) {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                sendMessage();
            }
        });

        function toggleModelModal(show) {
            const modal = document.getElementById('model-modal');
            modal.style.display = show ? 'flex' : 'none';
            if (show) {
                loadModelsList();
                searchHfModels();
            }
        }

        function toggleSettingsModal(show) {
            const modal = document.getElementById('settings-modal');
            modal.style.display = show ? 'flex' : 'none';
            if (show) loadSettings();
        }

        function toggleSessionsPanel(show) {
            const panel = document.getElementById('sessions-panel');
            const overlay = document.getElementById('sessions-overlay');

            console.log('[DEBUG] toggleSessionsPanel:', show);

            if (panel) {
                if (show) {
                    panel.classList.remove('-translate-x-full');
                    panel.classList.add('translate-x-0');
                } else {
                    panel.classList.remove('translate-x-0');
                    panel.classList.add('-translate-x-full');
                }
                console.log('[DEBUG] panel classes:', panel.className);
            }
            if (overlay) {
                overlay.style.display = show ? 'block' : 'none';
            }
            if (show) loadSessionList();
        }

        // Toggle draft model settings visibility
        function toggleDraftSettings() {
            const enabled = document.getElementById('setting-draft-enabled').checked;
            const section = document.getElementById('draft-temp-section');
            const slider = document.getElementById('setting-draft-temp');
            
            if (enabled) {
                section.classList.remove('hidden');
                slider.disabled = false;
            } else {
                section.classList.add('hidden');
                slider.disabled = true;
            }
        }

        // Toggle low-end GPU optimization settings visibility
        function updateLowEndSettings() {
            const enabled = document.getElementById('setting-low-end').checked;
            const section = document.getElementById('low-end-settings-section');
            
            if (enabled) {
                section.classList.remove('hidden');
            } else {
                section.classList.add('hidden');
            }
        }

        // Chain-of-Thought visibility (per-model)
        window.showChainOfThought = true;
        function syncCotToggle() {
            const cb = document.getElementById('setting-show-cot');
            window.showChainOfThought = cb ? cb.checked : true;
            const btn = document.getElementById('cot-toggle-btn');
            if (btn) {
                btn.textContent = window.showChainOfThought ? '🧠 CoT: ON' : '🧠 CoT: OFF';
                btn.classList.toggle('bg-cyan-950/60', window.showChainOfThought);
                btn.classList.toggle('bg-slate-800', !window.showChainOfThought);
            }
        }
        async function toggleCoT() {
            window.showChainOfThought = !window.showChainOfThought;
            const cb = document.getElementById('setting-show-cot');
            if (cb) cb.checked = window.showChainOfThought;
            syncCotToggle();
            // persist to settings.json instantly without opening modal
            try {
                const res = await fetch('/api/settings/load');
                const data = await res.json();
                const s = data.settings || {};
                s.show_chain_of_thought = window.showChainOfThought;
                await fetch('/api/settings/save', { method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify(s)});
            } catch(e){ console.warn('CoT persist failed', e); }
        }

        // --- Self-Improvement JS ---
        let _pendingCorrection = {prompt:'', response:'', thought:'', model:''};
        window._lastUserMsg = '';
        function toggleSelfImproveModal(show){
            const m = document.getElementById('self-improve-modal');
            if(!m) return;
            m.classList.toggle('hidden', !show);
            if(show) loadSelfImprove();
        }
        async function loadSelfImprove(){
            try{
                const r = await fetch('/api/self-improve');
                const d = await r.json();
                document.getElementById('si-stat-feedback').textContent = d.stats?.total_feedback ?? 0;
                document.getElementById('si-stat-up').textContent = d.stats?.up ?? 0;
                document.getElementById('si-stat-down').textContent = d.stats?.down ?? 0;
                document.getElementById('si-stat-gib').textContent = d.stats?.gibberish_fixes ?? 0;
                document.getElementById('si-stat-gib-rate').textContent = `rate ${d.gibberish_rate||0}%`;
                document.getElementById('si-stat-tunes').textContent = d.stats?.auto_tunes ?? 0;
                document.getElementById('si-stat-corr').textContent = `${d.stats?.corrections ?? 0} corrections`;
                const sugEl = document.getElementById('si-suggestions');
                if(d.suggestions && d.suggestions.length){
                    sugEl.innerHTML = d.suggestions.slice(-5).reverse().map(s=>`<div class="text-amber-300">• ${String(s.text||s).replace(/</g,'&lt;')}</div>`).join('');
                } else sugEl.innerHTML = '<div class="text-slate-500">No suggestions yet — chat and give feedback to generate hints.</div>';
                const logEl = document.getElementById('si-log');
                if(d.entries && d.entries.length){
                    logEl.innerHTML = d.entries.slice(-12).reverse().map(e=>`<div class="flex gap-2"><span class="text-slate-500">${(e.ts||'').slice(11,19)}</span><span class="text-cyan-400">${e.type||''}</span><span class="text-slate-300 truncate">${(e.prompt||JSON.stringify(e.changes||e)).slice(0,80).replace(/</g,'&lt;')}</span></div>`).join('');
                } else logEl.innerHTML = '<div class="text-slate-500">No entries yet.</div>';
                const fbEl = document.getElementById('si-feedback');
                if(d.feedback_recent && d.feedback_recent.length){
                    fbEl.innerHTML = d.feedback_recent.slice(-8).reverse().map(f=>`<div class="flex gap-2"><span class="${f.rating==='up'?'text-emerald-400':'text-red-400'}">${f.rating==='up'?'👍':'👎'}</span><span class="text-slate-400 truncate">${String(f.prompt||'').slice(0,70).replace(/</g,'&lt;')}</span><span class="text-slate-500">${(f.ts||'').slice(11,19)}</span></div>`).join('');
                } else fbEl.innerHTML = '<div class="text-slate-500">No feedback yet — use 👍👎 on messages.</div>';
            }catch(e){ console.warn('loadSelfImprove failed', e); }
        }
        async function submitFeedback(rating, btn){
            const bar = btn.closest('div');
            const prompt = bar?.dataset.prompt || window._lastUserMsg || '';
            const response = bar?.dataset.response || '';
            const thought = bar?.dataset.thought || '';
            const model = bar?.dataset.model || window.currentModelFilename || '';
            const statusEl = bar?.querySelector('.fb-status');
            try{
                const r = await fetch('/api/feedback', {method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify({rating, prompt, response, thought, model, correction:''})});
                const d = await r.json();
                if(statusEl) { statusEl.textContent = rating==='up' ? '✓ thanks!' : '✓ noted'; setTimeout(()=>statusEl.textContent='', 3000); }
                btn.style.opacity='0.5'; btn.disabled=true;
                if(d.suggestion && rating==='down'){ setTimeout(()=>alert(d.suggestion), 400); }
            }catch(e){ if(statusEl) statusEl.textContent='✗ failed'; }
        }
        function openCorrectionModal(){
            // find nearest bar from the clicked 👎
            const btn = event?.target;
            const bar = btn?.closest('div');
            const prompt = bar?.dataset.prompt || window._lastUserMsg || '';
            _pendingCorrection = {prompt, response: bar?.dataset.response||'', thought: bar?.dataset.thought||'', model: bar?.dataset.model||''};
            document.getElementById('correction-prompt-preview').textContent = prompt.slice(0,120) || '(unknown)';
            document.getElementById('correction-input').value = '';
            document.getElementById('correction-status').textContent = '';
            document.getElementById('correction-modal').classList.remove('hidden');
            setTimeout(()=>document.getElementById('correction-input').focus(), 80);
        }
        function closeCorrectionModal(){ document.getElementById('correction-modal').classList.add('hidden'); }
        async function submitCorrection(doRegenerate){
            const correction = document.getElementById('correction-input').value.trim();
            const statusEl = document.getElementById('correction-status');
            if(!correction){ statusEl.textContent='Please enter a correction'; return; }
            statusEl.textContent='Saving...';
            try{
                const r = await fetch('/api/feedback', {method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify({rating:'down', prompt:_pendingCorrection.prompt, response:_pendingCorrection.response, thought:_pendingCorrection.thought, model:_pendingCorrection.model, correction})});
                const d = await r.json();
                statusEl.textContent='✓ correction saved!';
                if(doRegenerate){
                    statusEl.textContent='✓ saved — regenerating...';
                    const rr = await fetch('/api/self-improve', {method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify({action:'regenerate', prompt:_pendingCorrection.prompt, correction})});
                    const dd = await rr.json();
                    if(dd.status==='success'){
                        appendMessage('assistant', dd.thought, dd.response, dd.model);
                        statusEl.textContent='✓ regenerated with your correction';
                        setTimeout(closeCorrectionModal, 1200);
                    } else statusEl.textContent='✗ regenerate failed: '+(dd.message||'');
                } else { setTimeout(closeCorrectionModal, 900); }
            }catch(e){ statusEl.textContent='✗ '+e; }
        }
        async function regenerateLastWithCritique(btn){
            const bar = btn.closest('div');
            const prompt = bar?.dataset.prompt || window._lastUserMsg || '';
            const statusEl = bar?.querySelector('.fb-status');
            if(!prompt){ if(statusEl) statusEl.textContent='no prompt'; return; }
            if(statusEl) statusEl.textContent='↻ regenerating...';
            try{
                const r = await fetch('/api/self-improve', {method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify({action:'regenerate', prompt, correction:'Regenerate with self-critique: be more direct and relevant to the prompt.'})});
                const d = await r.json();
                if(d.status==='success'){ appendMessage('assistant', d.thought, d.response, d.model); if(statusEl) statusEl.textContent='✓ done'; }
                else if(statusEl) statusEl.textContent='✗ '+d.message;
            }catch(e){ if(statusEl) statusEl.textContent='✗ '+e; }
        }
        async function runSelfImproveAnalyze(){
            const el=document.getElementById('si-action-status'); if(el) el.textContent='analyzing...';
            try{ const r=await fetch('/api/self-improve',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({action:'analyze'})}); const d=await r.json(); if(el) el.textContent=d.suggestions?.length? d.suggestions.slice(-1)[0].text.slice(0,80) : 'analysis done'; loadSelfImprove(); }catch(e){ if(el) el.textContent='fail '+e; }
        }
        async function runSelfImproveAutoTune(){
            const el=document.getElementById('si-action-status'); if(el) el.textContent='auto-tuning...';
            try{ const r=await fetch('/api/self-improve',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({action:'auto-tune'})}); const d=await r.json(); if(el) el.textContent=d.suggestion||'done'; loadSelfImprove(); if(d.changes?.temperature) document.getElementById('setting-temp') && (document.getElementById('setting-temp').value=d.changes.temperature); }catch(e){ if(el) el.textContent='fail '+e; }
        }
        async function runSelfImproveBenchmark(){
            const el=document.getElementById('si-action-status'); if(el) el.textContent='benchmark running (may take ~30s)...';
            try{ const r=await fetch('/api/self-improve',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({action:'analyze', run_benchmark:true})}); const d=await r.json(); if(el) el.textContent='benchmark done — check log'; if(d.benchmark) alert(String(d.benchmark).slice(0,2000)); loadSelfImprove(); }catch(e){ if(el) el.textContent='fail '+e; }
        }

        // --- Swarm JS ---
        function toggleSwarmModal(show){
            const m=document.getElementById('swarm-modal');
            if(!m) return;
            m.classList.toggle('hidden', !show);
            if(show){
                updateSwarmPreview();
                fetch('/api/swarm').then(r=>r.json()).then(d=>{
                    document.getElementById('swarm-model-name').textContent=d.model||'unknown';
                    document.getElementById('swarm-model-path').textContent=d.model_path||d.model||'';
                    const preview=document.getElementById('swarm-roles-preview');
                    if(d.roles){
                        const n=parseInt(document.getElementById('swarm-num').value)||3;
                        preview.innerHTML=d.roles.slice(0,n).map((r,i)=>`<div><span class="text-orange-400">Agent ${i+1}</span> <span class="text-amber-300">${r.name}</span> <span class="text-slate-500 text-[10px]">${r.desc.slice(0,60)}…</span></div>`).join('');
                    }
                }).catch(()=>{});
                document.getElementById('swarm-num').addEventListener('input', updateSwarmPreview);
            }
        }
        function updateSwarmPreview(){
            const n=parseInt(document.getElementById('swarm-num').value)||3;
            document.getElementById('swarm-num-label').textContent=n;
            const roles=["Researcher","Coder","Critic","Planner","Executor","Analyst"];
            const descs=["facts & sources","code & steps","flaws & edge cases","plan & breakdown","direct answer","trade-offs"];
            const el=document.getElementById('swarm-roles-preview');
            if(el) el.innerHTML=roles.slice(0,n).map((r,i)=>`<div><span class="text-orange-400">Agent ${i+1}</span> <span class="text-amber-300">${r}</span> <span class="text-slate-500 text-[10px]">${descs[i]}</span></div>`).join('');
        }
        async function runSwarm(){
            const task=document.getElementById('swarm-task').value.trim();
            const num_agents=parseInt(document.getElementById('swarm-num').value)||3;
            const mode=document.getElementById('swarm-mode').value;
            const statusEl=document.getElementById('swarm-status');
            const btn=document.getElementById('swarm-run-btn');
            if(!task){ if(statusEl) statusEl.textContent='Enter a task'; return; }
            btn.disabled=true; btn.textContent='SWARMING...';
            document.getElementById('swarm-results').classList.remove('hidden');
            const synEl=document.getElementById('swarm-synthesis');
            const agentsEl=document.getElementById('swarm-agents');
            // prepare live placeholders
            synEl.innerHTML = '<div class="text-[11px] text-orange-300 animate-pulse">● Synthesis streaming — waiting for drafts…</div><pre id="swarm-synth-thought-live" class="p-2 text-[11px] text-slate-400 font-mono whitespace-pre-wrap bg-[#0a0a14] max-h-40 overflow-y-auto"></pre><div id="swarm-synth-response-live" class="prose prose-invert max-w-none text-xs text-slate-300 border-t border-slate-800 pt-2 mt-2"></div>';
            agentsEl.innerHTML = '';
            const agentEls = {};
            if(statusEl) statusEl.textContent='Swarm streaming from '+document.getElementById('swarm-model-name').textContent+' — '+num_agents+'×'+mode+' …';
            try{
                const r=await fetch('/api/swarm/stream',{method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify({task, num_agents, mode})});
                if(!r.ok || !r.body) throw new Error('stream failed');
                const reader=r.body.getReader(); const dec=new TextDecoder(); let buf=""; let finalData=null;
                const synThoughtLive=document.getElementById('swarm-synth-thought-live');
                const synRespLive=document.getElementById('swarm-synth-response-live');
                let synthThoughtBuf="", synthRespBuf="";
                while(true){
                    const {value, done}=await reader.read(); if(done) break;
                    buf+=dec.decode(value,{stream:true}); let lines=buf.split("\n"); buf=lines.pop();
                    for(let line of lines){
                        if(!line.trim()) continue;
                        let ev; try{ ev=JSON.parse(line);}catch(e){continue;}
                        if(ev.type==='swarm_start'){
                            if(statusEl) statusEl.textContent='Swarm started — '+ev.num_agents+'×'+ev.mode+' · '+ev.model+' …';
                        } else if(ev.type==='agent_start'){
                            const div=document.createElement('div'); div.id='swarm-agent-'+ev.id;
                            div.className='bg-slate-950 border border-slate-800 rounded-lg p-3 space-y-2';
                            div.innerHTML='<div class="flex justify-between items-center"><span class="text-xs font-bold text-amber-300">Agent '+ev.id+' — '+ev.role+' <span class="animate-pulse text-orange-400">● thinking</span></span><span class="text-[10px] text-slate-500">live</span></div><pre class="agent-thought p-2 text-[11px] text-slate-400 font-mono whitespace-pre-wrap max-h-32 overflow-y-auto bg-slate-900/50 rounded border border-slate-800"></pre><div class="agent-response prose prose-invert max-w-none text-xs text-slate-300"></div>';
                            agentsEl.appendChild(div); agentEls[ev.id]=div;
                        } else if(ev.type==='agent_thought_delta'){
                            const d=agentEls[ev.id]; if(d){ const pre=d.querySelector('.agent-thought'); pre.textContent+=ev.chunk; pre.scrollTop=pre.scrollHeight; }
                        } else if(ev.type==='agent_response_delta'){
                            const d=agentEls[ev.id]; if(d){ const resp=d.querySelector('.agent-response'); resp.dataset.buf=(resp.dataset.buf||"")+ev.chunk; try{ resp.innerHTML=marked.parse(resp.dataset.buf);}catch(e){ resp.textContent=resp.dataset.buf; } }
                        } else if(ev.type==='agent_done'){
                            const d=agentEls[ev.id]; if(d){
                                d.querySelector('.agent-thought').textContent=ev.thought||d.querySelector('.agent-thought').textContent;
                                const resp=d.querySelector('.agent-response'); try{ resp.innerHTML=marked.parse(ev.response);}catch(e){ resp.textContent=ev.response; }
                                d.querySelector('span.animate-pulse').textContent='● done'; d.querySelector('span.animate-pulse').className='text-emerald-400';
                            }
                        } else if(ev.type==='synthesis_start'){
                            if(statusEl) statusEl.textContent='All drafts done — synthesizing…';
                        } else if(ev.type==='synthesis_thought_delta'){
                            synthThoughtBuf+=ev.chunk; if(synThoughtLive) { synThoughtLive.textContent=synthThoughtBuf; synThoughtLive.scrollTop=synThoughtLive.scrollHeight; }
                        } else if(ev.type==='synthesis_response_delta'){
                            synthRespBuf+=ev.chunk; if(synRespLive) { try{ synRespLive.innerHTML=marked.parse(synthRespBuf);}catch(e){ synRespLive.textContent=synthRespBuf; } }
                        } else if(ev.type==='done'){
                            finalData=ev;
                        } else if(ev.type==='error'){
                            if(statusEl) statusEl.textContent='✗ '+ev.message;
                        }
                    }
                }
                if(buf.trim()){ try{ const ev=JSON.parse(buf); if(ev.type==='done') finalData=ev; }catch(e){} }
                if(finalData){
                    const d=finalData;
                    if(statusEl) statusEl.textContent='Swarm done — '+d.num_agents+'×'+d.mode+' via '+d.model;
                    // finalize synthesis with full CoT header
                    const thoughtHtml = d.synthesis.thought ? '<details class="group bg-slate-950/90 rounded border border-orange-900/50 overflow-hidden"><summary class="px-3 py-2 text-[11px] font-bold text-orange-300 cursor-pointer hover:bg-slate-900 flex justify-between"><span>🧠 Thought for '+d.model+'</span><span class="text-orange-500 group-open:rotate-180">▾</span></summary><pre class="p-3 text-[11px] text-slate-300 font-mono whitespace-pre-wrap bg-[#0a0a14] max-h-52 overflow-y-auto">'+String(d.synthesis.thought).replace(/</g,'&lt;').replace(/>/g,'&gt;')+'</pre></details>' : '';
                    let respHtml=''; try{ respHtml = typeof marked!=='undefined' && marked.parse ? marked.parse(d.synthesis.response) : String(d.synthesis.response).replace(/</g,'&lt;').replace(/\n/g,'<br>'); }catch(e){ respHtml=String(d.synthesis.response).replace(/</g,'&lt;').replace(/\n/g,'<br>'); }
                    synEl.innerHTML = thoughtHtml+'<div class="prose prose-invert max-w-none text-xs text-slate-200">'+respHtml+'</div><div class="text-[10px] text-slate-500 font-mono">◈ '+d.model+' <button onclick="appendMessage(\'assistant\',\''+String(d.synthesis.thought).replace(/`/g,'').replace(/"/g,'&quot;').slice(0,500)+'\',\''+String(d.synthesis.response).replace(/`/g,'').replace(/"/g,'&quot;').slice(0,800)+'\',\''+d.model+'\'); document.getElementById(\'swarm-modal\').classList.add(\'hidden\')" class="ml-2 px-2 py-0.5 bg-orange-950 hover:bg-orange-900 text-orange-300 rounded border border-orange-800/50">→ Chat</button> <button onclick="navigator.clipboard.writeText(\''+String(d.synthesis.response).replace(/`/g,'').replace(/"/g,'&quot;').slice(0,2000)+'\')" class="ml-1 px-2 py-0.5 bg-slate-800 hover:bg-slate-700 text-slate-300 rounded">Copy</button></div>';
                    appendMessage('assistant', d.synthesis.thought, d.synthesis.response, d.model);
                    _addUseAsPrompt(synEl, d.synthesis.response);
                }
            }catch(e){
                // fallback to non-stream
                if(statusEl) statusEl.textContent='Stream failed, falling back… '+e;
                try{
                    const r2=await fetch('/api/swarm',{method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify({task, num_agents, mode})});
                    const d=await r2.json();
                    if(d.status==='success'){
                        if(statusEl) statusEl.textContent='Swarm done in '+d.elapsed_s+'s — '+d.num_agents+'×'+d.mode+' via '+d.model;
                        const synEl2=document.getElementById('swarm-synthesis');
                        const thoughtHtml = d.synthesis.thought ? '<details class="group bg-slate-950/90 rounded border border-orange-900/50 overflow-hidden"><summary class="px-3 py-2 text-[11px] font-bold text-orange-300 cursor-pointer hover:bg-slate-900 flex justify-between"><span>🧠 Thought for '+d.model+'</span><span class="text-orange-500 group-open:rotate-180">▾</span></summary><pre class="p-3 text-[11px] text-slate-300 font-mono whitespace-pre-wrap bg-[#0a0a14] max-h-52 overflow-y-auto">'+String(d.synthesis.thought).replace(/</g,'&lt;').replace(/>/g,'&gt;')+'</pre></details>' : '';
                        let respHtml=''; try{ respHtml = typeof marked!=='undefined' && marked.parse ? marked.parse(d.synthesis.response) : String(d.synthesis.response).replace(/</g,'&lt;').replace(/\n/g,'<br>'); }catch(e){ respHtml=String(d.synthesis.response).replace(/</g,'&lt;').replace(/\n/g,'<br>'); }
                        synEl2.innerHTML = thoughtHtml+'<div class="prose prose-invert max-w-none text-xs text-slate-200">'+respHtml+'</div><div class="text-[10px] text-slate-500 font-mono">◈ '+d.model+' · '+d.elapsed_s+'s</div>';
                        const agentsEl2=document.getElementById('swarm-agents');
                        agentsEl2.innerHTML = d.agents.map(a=>{ let aHtml=''; try{ aHtml = typeof marked!=='undefined' && marked.parse ? marked.parse(a.response) : String(a.response).replace(/</g,'&lt;').replace(/\n/g,'<br>'); }catch(e){ aHtml=String(a.response).replace(/</g,'&lt;').replace(/\n/g,'<br>'); } return '<div class="bg-slate-950 border border-slate-800 rounded-lg p-3 space-y-2"><div class="flex justify-between items-center"><span class="text-xs font-bold text-amber-300">Agent '+a.id+' — '+a.role+'</span><span class="text-[10px] text-slate-500">'+a.model+'</span></div><details class="bg-slate-900/50 rounded border border-slate-800"><summary class="px-2 py-1 text-[10px] text-slate-400 cursor-pointer">CoT</summary><pre class="p-2 text-[11px] text-slate-400 font-mono whitespace-pre-wrap max-h-32 overflow-y-auto">'+String(a.thought).replace(/</g,'&lt;').replace(/>/g,'&gt;').slice(0,800)+'</pre></details><div class="prose prose-invert max-w-none text-xs text-slate-300">'+aHtml.slice(0,1200)+'</div></div>'; }).join('');
                        appendMessage('assistant', d.synthesis.thought, d.synthesis.response, d.model);
                        _addUseAsPrompt(synEl2, d.synthesis.response);
                    }
                }catch(e2){ if(statusEl) statusEl.textContent='✗ '+e2; }
            }
            finally{ btn.disabled=false; btn.textContent='🐝 SPAWN SWARM'; }
        }

        // --- Council JS ---
        function toggleCouncilModal(show){
            const m=document.getElementById('council-modal');
            if(!m) return;
            m.classList.toggle('hidden', !show);
            if(show){
                updateCouncilPreview();
                fetch('/api/council').then(r=>r.json()).then(d=>{
                    document.getElementById('council-model-name').textContent=d.model||'unknown';
                    document.getElementById('council-model-path').textContent=d.model_path||d.model||'';
                    updateCouncilPreview(d);
                }).catch(()=>{});
            }
        }
        function updateCouncilPreview(api){
            const nd=parseInt(document.getElementById('council-drafts').value)||3;
            const nc=parseInt(document.getElementById('council-critics').value)||3;
            const el1=document.getElementById('council-drafts-label'); if(el1) el1.textContent=nd;
            const el2=document.getElementById('council-critics-label'); if(el2) el2.textContent=nc;
            const proposerNames=['Researcher','Coder','Critic','Planner','Executor'];
            const criticNames=['Accuracy','Clarity','Completeness','Safety','Efficiency'];
            const el=document.getElementById('council-preview'); if(!el) return;
            let html='<div class="text-violet-300">Drafts ('+nd+'):</div>';
            for(let i=0;i<nd;i++) html+='<div>· Draft '+(i+1)+' — <span class="text-amber-300">'+proposerNames[i%proposerNames.length]+'</span></div>';
            html+='<div class="text-violet-300 mt-2">Critics ('+nc+'):</div>';
            for(let i=0;i<nc;i++) html+='<div>· Critic '+(i+1)+' — <span class="text-violet-200">'+criticNames[i%criticNames.length]+' Critic</span></div>';
            if(api && api.recent_runs && api.recent_runs.length) html+='<div class="text-slate-500 text-[10px] mt-2">'+api.recent_runs.length+' recent councils</div>';
            el.innerHTML=html;
        }
        async function runCouncil(){
            const task=document.getElementById('council-task').value.trim();
            const num_drafts=parseInt(document.getElementById('council-drafts').value)||3;
            const num_critics=parseInt(document.getElementById('council-critics').value)||3;
            const statusEl=document.getElementById('council-status');
            const btn=document.getElementById('council-run-btn');
            if(!task){ if(statusEl) statusEl.textContent='Enter a task'; return; }
            btn.disabled=true; btn.textContent='CONVENING...';
            document.getElementById('council-results').classList.remove('hidden');
            const finalEl=document.getElementById('council-final');
            const draftsEl=document.getElementById('council-drafts');
            const criticsEl=document.getElementById('council-critics');
            const votesEl=document.getElementById('council-votes');
            finalEl.innerHTML='<div class="text-[11px] text-violet-300 animate-pulse">● Council streaming — drafts first, then critics…</div><pre id="council-final-thought-live" class="p-2 text-[11px] text-slate-400 font-mono whitespace-pre-wrap bg-[#0a0a14] max-h-40 overflow-y-auto"></pre><div id="council-final-response-live" class="prose prose-invert max-w-none text-xs text-slate-300 border-t border-slate-800 pt-2 mt-2"></div>';
            draftsEl.innerHTML=''; criticsEl.innerHTML=''; votesEl.innerHTML='<span class="text-slate-500">awaiting votes…</span>';
            const draftMap={}, criticMap={};
            let finalThoughtBuf="", finalRespBuf="", finalThoughtLive=document.getElementById('council-final-thought-live'), finalRespLive=document.getElementById('council-final-response-live');
            if(statusEl) statusEl.textContent='Council streaming from '+document.getElementById('council-model-name').textContent+' — '+num_drafts+' drafts + '+num_critics+' critics…';
            try{
                const r=await fetch('/api/council/stream',{method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify({task, num_drafts, num_critics})});
                if(!r.ok || !r.body) throw new Error('stream failed');
                const reader=r.body.getReader(); const dec=new TextDecoder(); let buf=""; let finalData=null;
                while(true){
                    const {value, done}=await reader.read(); if(done) break;
                    buf+=dec.decode(value,{stream:true}); let lines=buf.split("\n"); buf=lines.pop();
                    for(let line of lines){
                        if(!line.trim()) continue;
                        let ev; try{ ev=JSON.parse(line);}catch(e){continue;}
                        if(ev.type==='council_start'){
                            if(statusEl) statusEl.textContent='Council started — '+ev.num_drafts+' drafts + '+ev.num_critics+' critics · '+ev.model+'…';
                        } else if(ev.type==='draft_start'){
                            const div=document.createElement('div'); div.id='council-draft-'+ev.id;
                            div.className='bg-slate-950 border border-slate-800 rounded-lg p-3 space-y-2';
                            div.innerHTML='<div class="flex justify-between items-center"><span class="text-xs font-bold text-amber-300">Draft '+ev.id+' — '+ev.role+' <span class="animate-pulse text-violet-400">● thinking</span></span><span class="text-[10px] text-slate-500">live</span></div><pre class="draft-thought p-2 text-[11px] text-slate-400 font-mono whitespace-pre-wrap max-h-32 overflow-y-auto bg-slate-900/50 rounded border border-slate-800"></pre><div class="draft-response prose prose-invert max-w-none text-xs text-slate-300"></div>';
                            draftsEl.appendChild(div); draftMap[ev.id]=div;
                        } else if(ev.type==='draft_thought_delta'){
                            const d=draftMap[ev.id]; if(d){ const pre=d.querySelector('.draft-thought'); pre.textContent+=ev.chunk; pre.scrollTop=pre.scrollHeight; }
                        } else if(ev.type==='draft_response_delta'){
                            const d=draftMap[ev.id]; if(d){ const resp=d.querySelector('.draft-response'); resp.dataset.buf=(resp.dataset.buf||"")+ev.chunk; try{ resp.innerHTML=marked.parse(resp.dataset.buf);}catch(e){ resp.textContent=resp.dataset.buf; } }
                        } else if(ev.type==='draft_done'){
                            const d=draftMap[ev.id]; if(d){ d.querySelector('.draft-thought').textContent=ev.thought||d.querySelector('.draft-thought').textContent; const resp=d.querySelector('.draft-response'); try{ resp.innerHTML=marked.parse(ev.response);}catch(e){ resp.textContent=ev.response; } const pulse=d.querySelector('span.animate-pulse'); if(pulse){ pulse.textContent='● done'; pulse.className='text-emerald-400'; } }
                        } else if(ev.type==='critic_start'){
                            const div=document.createElement('div'); div.id='council-critic-'+ev.id;
                            div.className='bg-slate-950 border border-slate-800 rounded-lg p-3 space-y-2';
                            div.innerHTML='<div class="flex justify-between items-center"><span class="text-xs font-bold text-violet-300">'+ev.role+' #'+ev.id+' <span class="animate-pulse text-violet-400">● judging</span></span><span class="text-[10px] text-slate-500">live</span></div><pre class="critic-thought p-2 text-[11px] text-slate-400 font-mono whitespace-pre-wrap max-h-32 overflow-y-auto bg-slate-900/50 rounded border border-slate-800"></pre><div class="critic-response prose prose-invert max-w-none text-xs text-slate-300"></div><div class="critic-votes text-[10px] text-slate-500 font-mono"></div>';
                            criticsEl.appendChild(div); criticMap[ev.id]=div;
                        } else if(ev.type==='critic_thought_delta'){
                            const c=criticMap[ev.id]; if(c){ const pre=c.querySelector('.critic-thought'); pre.textContent+=ev.chunk; }
                        } else if(ev.type==='critic_response_delta'){
                            const c=criticMap[ev.id]; if(c){ const resp=c.querySelector('.critic-response'); resp.dataset.buf=(resp.dataset.buf||"")+ev.chunk; try{ resp.innerHTML=marked.parse(resp.dataset.buf);}catch(e){ resp.textContent=resp.dataset.buf; } }
                        } else if(ev.type==='critic_done'){
                            const c=criticMap[ev.id]; if(c){
                                const pulse=c.querySelector('span.animate-pulse'); if(pulse){ pulse.textContent='● voted D'+ev.pick; pulse.className='text-amber-300'; }
                                c.querySelector('.critic-thought').textContent=ev.thought||c.querySelector('.critic-thought').textContent;
                                // votes line
                                c.querySelector('.critic-votes').textContent='votes '+JSON.stringify(ev.votes)+' → pick D'+ev.pick;
                                const resp=c.querySelector('.critic-response'); try{ resp.innerHTML=marked.parse(ev.response);}catch(e){ resp.textContent=ev.response; }
                            }
                            if(statusEl) statusEl.textContent='Critic '+ev.id+' voted D'+ev.pick+' — '+Object.keys(ev.votes).length+' scored';
                        } else if(ev.type==='tally'){
                            votesEl.innerHTML='<div class="mb-1 text-violet-300">Winner: Draft '+ev.winner.id+' ('+ev.winner.role+') — picks '+(ev.tally[ev.winner.id]||0)+', score '+(ev.score_sum[ev.winner.id]||0)+'</div><table class="w-full text-[11px]"><thead><tr class="text-slate-500"><th class="text-left">Critic</th><th>Pick</th></tr></thead></table><div class="text-[10px] text-slate-500 mt-1">Score sums: '+Object.entries(ev.score_sum).map(([k,v])=> 'D'+k+'='+v).join(' · ')+' · tally: '+Object.entries(ev.tally).map(([k,v])=> 'D'+k+':'+v).join(' ')+'</div><div class="mt-2 bg-violet-950/30 border border-violet-900/40 rounded p-2 text-[11px] text-violet-200">'+ev.suggestions.map(s=>'<div>· '+String(s).replace(/</g,'&lt;')+'</div>').join('')+'</div>';
                            if(statusEl) statusEl.textContent='Tally done — winner Draft '+ev.winner.id+' ('+ev.winner.role+') — finalizing…';
                        } else if(ev.type==='final_start'){
                            if(statusEl) statusEl.textContent='Finalizer refining winner…';
                        } else if(ev.type==='final_thought_delta'){
                            finalThoughtBuf+=ev.chunk; if(finalThoughtLive){ finalThoughtLive.textContent=finalThoughtBuf; finalThoughtLive.scrollTop=finalThoughtLive.scrollHeight; }
                        } else if(ev.type==='final_response_delta'){
                            finalRespBuf+=ev.chunk; if(finalRespLive){ try{ finalRespLive.innerHTML=marked.parse(finalRespBuf);}catch(e){ finalRespLive.textContent=finalRespBuf; } }
                        } else if(ev.type==='done'){
                            finalData=ev;
                        } else if(ev.type==='error'){ if(statusEl) statusEl.textContent='✗ '+ev.message; }
                    }
                }
                if(buf.trim()){ try{ const ev=JSON.parse(buf); if(ev.type==='done') finalData=ev; }catch(e){} }
                if(finalData){
                    const d=finalData;
                    if(statusEl) statusEl.textContent='done — winner Draft '+d.winner.id+' ('+d.winner.role+') · votes '+JSON.stringify(d.tally);
                    const thoughtHtml = d.final.thought ? '<details class="group bg-slate-950/90 rounded border border-violet-900/50 overflow-hidden"><summary class="px-3 py-2 text-[11px] font-bold text-violet-300 cursor-pointer hover:bg-slate-900 flex justify-between"><span>🧠 Thought for '+d.model+'</span><span class="text-violet-500 group-open:rotate-180">▾</span></summary><pre class="p-3 text-[11px] text-slate-300 font-mono whitespace-pre-wrap bg-[#0a0a14] max-h-52 overflow-y-auto">'+String(d.final.thought).replace(/</g,'&lt;').replace(/>/g,'&gt;')+'</pre></details>' : '';
                    let respHtml=''; try{ respHtml = typeof marked!=='undefined' && marked.parse ? marked.parse(d.final.response) : String(d.final.response).replace(/</g,'&lt;').replace(/\n/g,'<br>'); }catch(e){ respHtml=String(d.final.response).replace(/</g,'&lt;').replace(/\n/g,'<br>'); }
                    finalEl.innerHTML = thoughtHtml+'<div class="prose prose-invert max-w-none text-xs text-slate-200">'+respHtml+'</div><div class="text-[10px] text-slate-500 font-mono"> '+d.model+' · winner Draft '+d.winner.id+' ('+d.winner.role+') <button onclick="appendMessage(\'assistant\',\''+String(d.final.thought).replace(/`/g,'').replace(/"/g,'&quot;').slice(0,500)+'\',\''+String(d.final.response).replace(/`/g,'').replace(/"/g,'&quot;').slice(0,800)+'\',\''+d.model+'\'); document.getElementById(\'council-modal\').classList.add(\'hidden\')" class="ml-2 px-2 py-0.5 bg-violet-950 hover:bg-violet-900 text-violet-300 rounded border border-violet-800/50">to Chat</button> <button onclick="navigator.clipboard.writeText(\''+String(d.final.response).replace(/`/g,'').replace(/"/g,'&quot;').slice(0,2000)+'\')" class="ml-1 px-2 py-0.5 bg-slate-800 hover:bg-slate-700 text-slate-300 rounded">Copy</button></div><div class="mt-2 bg-violet-950/30 border border-violet-900/40 rounded p-2 text-[11px] text-violet-200"><div class="font-bold text-violet-300 text-[10px] uppercase tracking-wider">Applied suggestions</div>'+(d.suggestions.map(s=>'<div>· '+String(s).replace(/</g,'&lt;')+'</div>').join('')||'<div class="text-slate-500">none</div>')+'</div>';
                    // refresh drafts/critics with winner highlight
                    // re-render votes table full
                    const votesEl2=document.getElementById('council-votes');
                    let tbl='<div class="mb-1 text-violet-300">Winner: Draft '+d.winner.id+' ('+d.winner.role+') — picks '+(d.tally[d.winner.id]||0)+', score '+(d.score_sum[d.winner.id]||0)+'</div>';
                    tbl+='<table class="w-full text-[11px]"><thead><tr class="text-slate-500"><th class="text-left">Critic</th><th>Pick</th>'+d.drafts.map(x=>'<th>D'+x.id+'</th>').join('')+'<th class="text-left">Suggestion</th></tr></thead><tbody>';
                    for(const c of d.critics){ tbl+='<tr class="border-t border-slate-800"><td class="py-1 text-violet-200">'+c.role+' #'+c.id+'</td><td class="text-center text-amber-300">D'+c.pick+'</td>'; for(const dr of d.drafts){ const sc=c.votes[dr.id]||'-'; const isPick=dr.id===c.pick?' bg-violet-900/40':''; tbl+='<td class="text-center'+isPick+'">'+sc+'</td>'; } tbl+='<td class="text-slate-400 max-w-[260px] truncate">'+String(c.suggestions[c.pick]||Object.values(c.suggestions)[0]||'').replace(/</g,'&lt;').slice(0,80)+'</td></tr>'; }
                    tbl+='</tbody></table><div class="text-[10px] text-slate-500 mt-1">Score sums: '+d.drafts.map(x=>'D'+x.id+'='+d.score_sum[x.id]).join(' · ')+' · picks: '+Object.entries(d.tally).map(([k,v])=>'D'+k+':'+v).join(' ')+'</div>';
                    votesEl2.innerHTML=tbl;
                    // highlight winner draft border
                    for(const dr of d.drafts){ const el=document.getElementById('council-draft-'+dr.id); if(el && dr.id===d.winner.id){ el.classList.remove('border-slate-800'); el.classList.add('border-violet-500/50','shadow-[0_0_12px_rgba(139,92,246,0.15)]'); } }
                    appendMessage('assistant', d.final.thought, d.final.response, d.model);
                    _addUseAsPrompt(finalEl, d.final.response);
                }
            }catch(e){
                if(statusEl) statusEl.textContent='Stream failed, falling back… '+e;
                try{
                    const r2=await fetch('/api/council',{method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify({task, num_drafts, num_critics})});
                    const d=await r2.json();
                    if(d.status==='success'){
                        if(statusEl) statusEl.textContent='done in '+d.elapsed_s+'s — winner Draft '+d.winner.id+' ('+d.winner.role+') · votes '+JSON.stringify(d.tally);
                        const thoughtHtml = d.final.thought ? '<details class="group bg-slate-950/90 rounded border border-violet-900/50 overflow-hidden"><summary class="px-3 py-2 text-[11px] font-bold text-violet-300 cursor-pointer hover:bg-slate-900 flex justify-between"><span>🧠 Thought for '+d.model+'</span><span class="text-violet-500 group-open:rotate-180">▾</span></summary><pre class="p-3 text-[11px] text-slate-300 font-mono whitespace-pre-wrap bg-[#0a0a14] max-h-52 overflow-y-auto">'+String(d.final.thought).replace(/</g,'&lt;').replace(/>/g,'&gt;')+'</pre></details>' : '';
                        let respHtml=''; try{ respHtml = typeof marked!=='undefined' && marked.parse ? marked.parse(d.final.response) : String(d.final.response).replace(/</g,'&lt;').replace(/\n/g,'<br>'); }catch(e){ respHtml=String(d.final.response).replace(/</g,'&lt;').replace(/\n/g,'<br>'); }
                        finalEl.innerHTML = thoughtHtml+'<div class="prose prose-invert max-w-none text-xs text-slate-200">'+respHtml+'</div>';
                        appendMessage('assistant', d.final.thought, d.final.response, d.model);
                        _addUseAsPrompt(finalEl, d.final.response);
                    }
                }catch(e2){ if(statusEl) statusEl.textContent='✗ '+e2; }
            }
            finally{ btn.disabled=false; btn.textContent='⚖️ CONVENE COUNCIL'; }
        }

        // Render helpers for loading session history
        function renderUserMessage(msg) {
            return `
                <div class="flex items-start space-x-4 max-w-4xl">
                    <div class="w-8 h-8 rounded bg-slate-700/50 border border-slate-600 flex items-center justify-center font-bold text-xs text-slate-300 shrink-0 mt-1">U</div>
                    <div class="bg-slate-900/90 border border-slate-700/60 rounded-xl p-4 text-slate-200 text-sm shadow-xl"><p>${msg.replace(/</g,'&lt;').replace(/>/g,'&gt;')}</p></div>
                </div>`;
        }

        function renderAssistantResponse(content) {
            // Strip think tags for display, extract thought vs final answer
            let cleanContent = content;
            let thoughtText = '';
            const thinkMatch = content.match(/<think>([\s\S]*?)<\/think>/);
            if (thinkMatch) {
                thoughtText = thinkMatch[1].trim();
                cleanContent = content.substring(thinkMatch.index + thinkMatch[0].length).trim();
            }
            const modelLabel = window.currentModelFilename?.split('/').pop()?.split('\\').pop() || 'selected model';
            const cotHidden = window.showChainOfThought === false;
            const _qwenEscape = thoughtText.replace(/</g,'&lt;').replace(/>/g,'&gt;');
            const _qwenTokens = _qwenEscape.split(/\s+/).filter(Boolean).length;
            // workaround: use String split without regex to avoid \s escaping issues in Python raw string: use split(' ') approximate
            const _tok2 = thoughtText.trim().split(' ').filter(Boolean).length;
            const cotHtml = thoughtText && !cotHidden
                ? `<details class="group qwen-think bg-[#0f1115] border-l-2 border-amber-700/70 rounded-r-lg overflow-hidden mb-3"><summary class="flex items-center gap-2 px-3 py-2 text-[11px] font-medium text-slate-400 cursor-pointer select-none hover:bg-[#151821] list-none"><span class="w-1.5 h-1.5 rounded-full bg-amber-500"></span><span class="text-slate-300">Thought for ${modelLabel}</span><span class="text-slate-600 font-mono">· ~${_tok2} tokens</span><span class="ml-auto text-slate-600 group-open:rotate-180 transition-transform text-[10px]">▾</span></summary><pre class="px-3.5 py-2.5 text-[11px] leading-[1.65] text-[#8b8d98] font-mono whitespace-pre-wrap bg-transparent max-h-[36vh] overflow-y-auto border-t border-slate-800/40 italic">${_qwenEscape}</pre></details>`
                : (thoughtText && cotHidden ? `<div class="text-[10px] text-slate-500 italic mb-2 pl-3 border-l-2 border-slate-700">Thought hidden · enable via ❧ CoT</div>` : '');
            return `
                <div class="flex items-start space-x-4 max-w-4xl">
                    <div class="w-8 h-8 rounded bg-cyan-600/20 border border-cyan-500 flex items-center justify-center font-bold text-xs text-cyan-400 shrink-0 mt-1">Ω</div>
                    <div class="bg-slate-900/90 border border-indigo-900/60 rounded-xl p-4 text-slate-200 text-sm shadow-xl space-y-2">
                        ${cotHtml}
                        <div class="prose prose-invert prose-sm max-w-none">${cleanContent.replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/\n/g,'<br>')}</div>
                        <div class="text-[10px] text-slate-600 font-mono pt-2 border-t border-slate-800/50 flex justify-between"><span>◈ ${modelLabel}</span><span>${new Date().toLocaleTimeString()}</span></div>
                    </div>
                </div>`;
        }

        let currentDirPath = '';
        let currentEditingFile = '';
        let editorMode = 'view'; // 'view' or 'edit'

        function onEditorInput() {
            const editor = document.getElementById('code-editor');
            const codeEl = document.getElementById('highlighted-code');
            codeEl.textContent = editor.value;
            Prism.highlightElement(codeEl);
        }

        function setEditorMode(mode) {
            editorMode = mode;
            const editor = document.getElementById('code-editor');
            const viewBtn = document.getElementById('mode-view-btn');
            const editBtn = document.getElementById('mode-edit-btn');

            if (mode === 'view') {
                editor.readOnly = true;
                viewBtn.className = 'px-3 py-1 rounded bg-emerald-950 text-emerald-300 border border-emerald-800';
                editBtn.className = 'px-3 py-1 rounded bg-slate-800 text-slate-400';
            } else {
                editor.readOnly = false;
                viewBtn.className = 'px-3 py-1 rounded bg-slate-800 text-slate-400';
                editBtn.className = 'px-3 py-1 rounded bg-emerald-950 text-emerald-300 border border-emerald-800';
                editor.focus();
            }
        }

        async function loadWorkspaceDir(dirPath) {
            const treeEl = document.getElementById('file-tree');
            treeEl.innerHTML = '<div class="text-slate-500">Loading directory...</div>';
            try {
                const res = await fetch(`/api/workspace/list?dir=${encodeURIComponent(dirPath)}`);
                const data = await res.json();
                currentDirPath = data.current_path;
                document.getElementById('current-dir-label').textContent = currentDirPath;
                treeEl.innerHTML = '';

                if (data.parent_path) {
                    const parentItem = document.createElement('div');
                    parentItem.className = 'p-2 rounded hover:bg-slate-900 cursor-pointer text-amber-300 font-bold transition';
                    parentItem.textContent = '📁 .. (Parent Directory)';
                    parentItem.onclick = () => loadWorkspaceDir(data.parent_path);
                    treeEl.appendChild(parentItem);
                }

                data.items.forEach(item => {
                    const el = document.createElement('div');
                    el.className = `p-2 rounded hover:bg-slate-900 cursor-pointer transition flex items-center justify-between ${item.is_dir ? 'text-cyan-400 font-bold' : 'text-slate-300'}`;
                    el.innerHTML = `<span>${item.is_dir ? '📁 ' : '📄 '}${item.name}</span>`;
                    if (item.is_dir) {
                        el.onclick = () => loadWorkspaceDir(item.path);
                    } else {
                        el.onclick = () => openFile(item.path);
                    }
                    treeEl.appendChild(el);
                });

                // Push this workspace directory into model context and update UI
                const ctxLabel = document.getElementById('workspace-context-label');
                const dirName = dirPath || '(Root)';
                if (ctxLabel) {
                    ctxLabel.textContent = `📁 ${dirName}`;
                    ctxLabel.title = dirName;
                }
                try {
                    await fetch('/api/workspace/context', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ dir: dirPath })
                    });
                } catch (e) { /* non-blocking — don't break UI if fails */ }
            } catch (err) {
                treeEl.innerHTML = '<div class="text-red-400">Failed to load directory.</div>';
            }
        }

        async function openFile(filepath) {
            currentEditingFile = filepath;
            document.getElementById('current-file-path').textContent = filepath;
            const editor = document.getElementById('code-editor');
            const codeEl = document.getElementById('highlighted-code');
            editor.value = 'Loading...';
            codeEl.textContent = 'Loading...';

            const ext = filepath.split('.').pop().toLowerCase();
            const langMap = {
                'py': 'language-python',
                'json': 'language-json',
                'js': 'language-javascript',
                'ts': 'language-typescript',
                'html': 'language-markup',
                'htm': 'language-markup',
                'css': 'language-css',
                'md': 'language-markdown',
                'sh': 'language-bash',
                'bat': 'language-powershell',
                'ps1': 'language-powershell',
                'yml': 'language-yaml',
                'yaml': 'language-yaml',
                'go': 'language-go',
                'rs': 'language-rust',
                'c': 'language-c',
                'h': 'language-c',
                'cpp': 'language-cpp',
                'cc': 'language-cpp',
                'cxx': 'language-cpp',
                'hpp': 'language-cpp',
                'java': 'language-java',
                'cs': 'language-csharp',
                'rb': 'language-ruby',
                'sql': 'language-sql',
                'xml': 'language-xml',
                'toml': 'language-toml',
                'lua': 'language-lua',
                'swift': 'language-swift',
                'kt': 'language-kotlin',
                'kts': 'language-kotlin',
                'dockerfile': 'language-docker',
                'dockerignore': 'language-docker',
                'gitignore': 'language-git',
                'gitconfig': 'language-git',
                'regex': 'language-regex',
                're': 'language-regex',
                'ini': 'language-ini',
                'cfg': 'language-ini',
                'env': 'language-bash',
                'tf': 'language-hcl',
                'hcl': 'language-hcl',
                'proto': 'language-protobuf',
                'graphql': 'language-graphql',
                'vue': 'language-markup',
                'svelte': 'language-markup',
                'scss': 'language-css',
                'sass': 'language-css',
                'jsx': 'language-javascript',
                'tsx': 'language-typescript',
            };
            codeEl.className = langMap[ext] || 'language-python';

            try {
                console.log('[DEBUG] Opening file:', filepath);
                const res = await fetch('/api/workspace/read', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ path: filepath })
                });
                console.log('[DEBUG] Response status:', res.status);
                const data = await res.json();
                console.log('[DEBUG] Response:', data);
                if (data.status === 'success') {
                    const editor = document.getElementById('code-editor');
                    const codeEl = document.getElementById('highlighted-code');
                    console.log('[DEBUG] editor:', !!editor, 'codeEl:', !!codeEl, 'content length:', data.content?.length || 0);
                    
                    if (editor) {
                        editor.value = data.content;
                    }
                    
                    if (codeEl) {
                        // Set text and force re-render before highlighting
                        codeEl.textContent = data.content;
                        void codeEl.offsetWidth; // Force reflow
                        
                        // Use requestAnimationFrame then Prism.highlightAll()
                        requestAnimationFrame(() => {
                            if (typeof Prism !== 'undefined') {
                                try {
                                    Prism.highlightAll();
                                    console.log('[DEBUG] Prism.highlightAll() called successfully');
                                } catch(e) {
                                    console.warn('[DEBUG] Prism.highlightAll() failed:', e);
                                }
                            } else {
                                console.warn('[DEBUG] Prism not loaded');
                            }
                        });
                    }
                    if (!editor && !codeEl) {
                        console.error('[DEBUG] ERROR: code-editor or highlighted-code element not found!');
                    }
                } else {
                    console.error('[DEBUG] Server returned error:', data.message);
                    const editor = document.getElementById('code-editor');
                    if (editor) {
                        editor.value = 'Error: ' + data.message;
                    }
                }
            } catch (err) {
                console.error('[DEBUG] Exception:', err);
                const editor = document.getElementById('code-editor');
                if (editor) {
                    editor.value = 'Failed to load file content: ' + err.message;
                }
            }
        }

        async function saveCurrentFile() {
            if (!currentEditingFile) return;
            const content = document.getElementById('code-editor').value;
            const statusEl = document.getElementById('editor-status');
            statusEl.textContent = `Saving ${currentEditingFile}...`;

            try {
                const res = await fetch('/api/workspace/write', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ path: currentEditingFile, content })
                });
                const data = await res.json();
                if (data.status === 'success') {
                    statusEl.textContent = `Successfully saved ${currentEditingFile}!`;
                    setTimeout(() => statusEl.textContent = '', 3000);
                } else {
                    statusEl.textContent = `Error: ${data.message}`;
                }
            } catch (err) {
                statusEl.textContent = 'Save failed.';
            }
        }

        async function saveSettings() {
            const settings = {
                temperature: parseFloat(document.getElementById('setting-temp').value),
                top_k: parseInt(document.getElementById('setting-topk').value),
                top_p: parseFloat(document.getElementById('setting-topp').value),
                max_new_tokens: parseInt(document.getElementById('setting-tokens').value),
                context_length: parseInt(document.getElementById('setting-context').value),
                gpu_layers: parseInt(document.getElementById('setting-gpu-layers').value)
            };
            
            // Add draft model settings
            const draftEnabled = document.getElementById('setting-draft-enabled').checked;
            if (draftEnabled) {
                settings.use_draft_model = true;
                settings.draft_temperature = parseFloat(document.getElementById('setting-draft-temp').value);
            } else {
                settings.use_draft_model = false;
            }
            
            // Add low-end GPU optimization settings
            const lowEndEnabled = document.getElementById('setting-low-end').checked;
            settings.low_end_gpu_mode = lowEndEnabled;
            settings.precision = document.getElementById('setting-precision').value;
            settings.cpu_offload_layers = parseInt(document.getElementById('setting-cpu-offload').value);
            settings.show_chain_of_thought = document.getElementById('setting-show-cot').checked;
            settings.auto_swarm_council = document.getElementById('setting-auto-swarm-council').checked;
            settings.auto_web_research = document.getElementById('setting-auto-web-research').checked;
            
            // Also save current model path - guard against template placeholder leaking
            let m = window.currentModelFilename;
            if (!m || m.includes('{{')) m = null;
            if (m) settings.current_model = m;
            else {
                const fallback = document.getElementById('active-model-name')?.textContent?.trim();
                if (fallback && !fallback.includes('{{') && fallback.length > 3) settings.current_model = fallback;
            }
            if (settings.current_model && settings.current_model.includes('{{')) delete settings.current_model;
            
            const statusEl = document.getElementById('settings-status');
            statusEl.textContent = 'Updating settings...';

            try {
                // Save to API endpoint (also persists to JSON and applies live to reasoner)
                const res = await fetch('/api/settings/save', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(settings)
                });
                const data = await res.json();
                
                if (data.status === 'saved') {
                    statusEl.textContent = 'Settings saved & applied live!';
                    // update sidebar telemetry instantly
                    const s = data.settings || settings;
                    const ctxEl = document.getElementById('sidebar-ctx');
                    const gpuEl = document.getElementById('sidebar-gpu');
                    if (ctxEl) ctxEl.textContent = (s.context_length || settings.context_length).toLocaleString() + ' Tokens';
                    if (gpuEl) gpuEl.textContent = (s.gpu_layers || settings.gpu_layers) + ' / 32';
                    // keep in-memory current model in sync
                    if (s.current_model) window.currentModelFilename = s.current_model;
                    if (s.show_chain_of_thought !== undefined) {
                        window.showChainOfThought = !!s.show_chain_of_thought;
                        const cb = document.getElementById('setting-show-cot');
                        if (cb) cb.checked = window.showChainOfThought;
                        syncCotToggle();
                    }
                    setTimeout(() => toggleSettingsModal(false), 800);
                } else {
                    statusEl.textContent = 'Error saving settings: ' + (data.message||'unknown');
                }
            } catch (err) {
                statusEl.textContent = 'Failed to save settings: ' + err;
            }
        }

        async function loadSettings() {
            try {
                const res = await fetch('/api/settings/load');
                const data = await res.json();
                const s = data.settings || data;
                if (!s) return;
                // populate sliders/inputs from saved file (the source of truth)
                const setVal = (id, val) => { const el=document.getElementById(id); if(el && val!==undefined) el.value=val; };
                const setChecked = (id, val) => { const el=document.getElementById(id); if(el) el.checked=!!val; };
                setVal('setting-temp', s.temperature); document.getElementById('temp-val').textContent = s.temperature;
                setVal('setting-topk', s.top_k); document.getElementById('topk-val').textContent = s.top_k;
                setVal('setting-topp', s.top_p); document.getElementById('topp-val').textContent = s.top_p;
                setVal('setting-tokens', s.max_new_tokens); document.getElementById('tokens-val').textContent = s.max_new_tokens;
                setVal('setting-context', s.context_length);
                setVal('setting-gpu-layers', s.gpu_layers); document.getElementById('gpu-layers-val').textContent = s.gpu_layers;
                setChecked('setting-draft-enabled', s.use_draft_model); toggleDraftSettings();
                if (s.draft_temperature!==undefined) { setVal('setting-draft-temp', s.draft_temperature); document.getElementById('draft-temp-val').textContent = s.draft_temperature; }
                setChecked('setting-low-end', s.low_end_gpu_mode); updateLowEndSettings();
                if (s.precision) setVal('setting-precision', s.precision);
                if (s.cpu_offload_layers!==undefined) { setVal('setting-cpu-offload', s.cpu_offload_layers); document.getElementById('cpu-offload-val').textContent = s.cpu_offload_layers; }
                if (s.current_model) window.currentModelFilename = s.current_model;
                // Chain-of-Thought visibility
                if (s.show_chain_of_thought !== undefined) {
                    setChecked('setting-show-cot', s.show_chain_of_thought);
                    window.showChainOfThought = !!s.show_chain_of_thought;
                    const btn = document.getElementById('cot-toggle-btn');
                    if (btn) btn.textContent = window.showChainOfThought ? '🧠 CoT: ON' : '🧠 CoT: OFF';
                    btn?.classList.toggle('bg-cyan-950/60', window.showChainOfThought);
                    btn?.classList.toggle('bg-slate-800', !window.showChainOfThought);
                }
                // Auto Swarm + Council enrichment
                if (s.auto_swarm_council !== undefined) setChecked('setting-auto-swarm-council', s.auto_swarm_council);
                if (s.auto_web_research !== undefined) setChecked('setting-auto-web-research', s.auto_web_research);
                // sidebar
                const ctxEl = document.getElementById('sidebar-ctx');
                const gpuEl = document.getElementById('sidebar-gpu');
                if (ctxEl && s.context_length) ctxEl.textContent = s.context_length.toLocaleString() + ' Tokens';
                if (gpuEl && s.gpu_layers!==undefined) gpuEl.textContent = s.gpu_layers + ' / 32';
                if (s.current_model) { const mEl=document.getElementById('active-model-name'); if(mEl) mEl.textContent = s.current_model.split('/').pop().split('\\\\').pop(); }
            } catch(e) { console.warn('[Settings] load failed', e); }
        }

        function switchTab(tab) {
            ['local', 'hf', 'upload'].forEach(t => {
                document.getElementById(`section-${t}`).classList.add('hidden');
                document.getElementById(`tab-${t}`).className = 'px-3 py-1.5 rounded-lg text-xs font-semibold bg-slate-800 text-slate-400 hover:text-white transition';
            });
            document.getElementById(`section-${tab}`).classList.remove('hidden');
            document.getElementById(`tab-${tab}`).className = 'px-3 py-1.5 rounded-lg text-xs font-semibold bg-cyan-950 text-cyan-300 border border-cyan-800/60 transition';
        }

        async function loadModelsList() {
            const listEl = document.getElementById('model-list');
            listEl.innerHTML = '<div class="text-slate-500">Scanning local checkpoints &amp; GGUF files...</div>';
            try {
                const res = await fetch('/api/models');
                const data = await res.json();
                listEl.innerHTML = '';
                if (data.models.length === 0) {
                    listEl.innerHTML = '<div class="text-slate-500">No checkpoints found. Upload or download one!</div>';
                    return;
                }
                data.models.forEach(m => {
                    const item = document.createElement('div');
                    item.className = 'flex justify-between items-center bg-slate-900 p-2.5 rounded border border-slate-800';
                    item.innerHTML = `
                        <div>
                            <span class="text-cyan-300 font-bold">${m.name}</span>
                            <span class="text-slate-500 ml-2">(${m.size_mb} MB)</span>
                            <span class="text-slate-600 text-[10px] ml-2 block">${m.path}</span>
                            ${m.type === 'qwen-hf' ? '<span class="mt-1 inline-block px-1.5 py-0.5 bg-sky-950 text-sky-300 text-[10px] rounded border border-sky-800 ml-1">QWEN</span>' : ''}
                            ${m.active ? '<span class="mt-1 inline-block px-1.5 py-0.5 bg-emerald-950 text-emerald-300 text-[10px] rounded border border-emerald-800">ACTIVE</span>' : ''}
                        </div>
                        ${!m.active ? `<button onclick="setDefaultModel('${m.path.replace(/\\/g, '/')}')" class="px-2.5 py-1 bg-violet-900 hover:bg-violet-800 text-violet-200 rounded text-[11px] transition">SET DEFAULT</button>` : ''}
                    `;
                    listEl.appendChild(item);
                });
            } catch (err) {
                listEl.innerHTML = '<div class="text-red-400">Failed to load local checkpoints.</div>';
            }
        }

        async function scanPcModels() {
            const statusEl = document.getElementById('upload-status');
            statusEl.textContent = 'Scanning PC & Hugging Face cache for .pk1 / .gguf files...';
            try {
                const res = await fetch('/api/models/scan-pc', { method: 'POST' });
                const data = await res.json();
                statusEl.textContent = `Scan complete! Found ${data.found} model file(s).`;
                loadModelsList();
            } catch (err) {
                statusEl.textContent = 'PC scan failed.';
            }
        }

        async function setDefaultModel(modelPath) {
            try {
                const res = await fetch('/api/models/set-default', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ path: modelPath })
                });
                const data = await res.json();
                if (data.status === 'set') {
                    window.currentModelFilename = modelPath;
                    document.getElementById('active-model-name').textContent = data.model.split('\\').pop().split('/').pop();
                    alert(`Default model set to: ${modelPath.split('\\').pop().split('/').pop()}`);
                    loadModelsList(); // Refresh list to update ACTIVE badges
                } else {
                    alert('Failed to set default model.');
                }
            } catch (err) {
                alert('Error setting default model.');
            }
        }

        async function searchHfModels() {
            const query = document.getElementById('hf-search-input').value.trim();
            const hfListEl = document.getElementById('hf-model-list');
            hfListEl.innerHTML = '<div class="text-slate-500">Querying Hugging Face Hub API...</div>';
            try {
                const res = await fetch(`/api/hf-search?q=${encodeURIComponent(query)}`);
                const data = await res.json();
                hfListEl.innerHTML = '';
                if (data.models.length === 0) {
                    hfListEl.innerHTML = '<div class="text-slate-500">No Hugging Face models found matching query.</div>';
                    return;
                }
                data.models.forEach(m => {
                    const item = document.createElement('div');
                    item.className = 'bg-slate-900 p-3 rounded-lg border border-slate-800 flex justify-between items-center';
                    item.innerHTML = `
                        <div>
                            <div class="text-cyan-300 font-bold">${m.repo_id}</div>
                            <div class="text-slate-400 text-[11px]">Downloads: ${m.downloads} &bull; Likes: ${m.likes}</div>
                        </div>
                        <button onclick="downloadHfRepo('${m.repo_id}')" class="px-3 py-1.5 bg-cyan-600 hover:bg-cyan-500 text-slate-950 rounded font-bold text-xs transition shrink-0">PULL MODEL</button>
                    `;
                    hfListEl.appendChild(item);
                });
            } catch (err) {
                hfListEl.innerHTML = '<div class="text-red-400">Failed to search Hugging Face Hub.</div>';
            }
        }

        async function downloadHfRepo(repo_id) {
            const statusEl = document.getElementById('upload-status');
            statusEl.textContent = `Pulling repository ${repo_id} from Hugging Face...`;
            try {
                const res = await fetch('/api/models/download-hf-repo', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ repo_id })
                });
                const data = await res.json();
                if (data.status === 'success') {
                    statusEl.textContent = `Successfully pulled ${repo_id}!`;
                    switchTab('local');
                    loadModelsList();
                } else {
                    statusEl.textContent = `Pull error: ${data.message}`;
                }
            } catch (err) {
                statusEl.textContent = 'Pull failed.';
            }
        }

        async function switchModel(filename) {
            const statusEl = document.getElementById('upload-status');
            statusEl.textContent = `Switching active model to ${filename}...`;
            try {
                const res = await fetch('/api/models/switch', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ filename })
                });
                const data = await res.json();
                if (data.status === 'success') {
                    document.getElementById('active-model-name').textContent = filename;
                    statusEl.textContent = `Successfully activated ${filename}!`;
                    loadModelsList();
                } else {
                    statusEl.textContent = `Error: ${data.message}`;
                }
            } catch (err) {
                statusEl.textContent = 'Failed to switch model.';
            }
        }

        async function uploadModel() {
            const fileInput = document.getElementById('model-file-input');
            const statusEl = document.getElementById('upload-status');
            if (!fileInput.files.length) {
                statusEl.textContent = 'Please select a checkpoint or GGUF file.';
                return;
            }
            const file = fileInput.files[0];
            statusEl.textContent = `Uploading ${file.name} (${(file.size/1024/1024).toFixed(1)} MB)...`;

            const reader = new FileReader();
            reader.onload = async function(e) {
                const base64Data = e.target.result.split(',')[1];
                try {
                    const res = await fetch('/api/models/upload', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ filename: file.name, content_base64: base64Data })
                    });
                    const data = await res.json();
                    if (data.status === 'success') {
                        statusEl.textContent = `Successfully uploaded and saved ${file.name}!`;
                        fileInput.value = '';
                        switchTab('local');
                        loadModelsList();
                    } else {
                        statusEl.textContent = `Upload error: ${data.message}`;
                    }
                } catch (err) {
                    statusEl.textContent = 'Upload failed.';
                }
            };
            reader.readAsDataURL(file);
        }

        function toggleScanlines() {
            const overlay = document.getElementById('scanline-overlay');
            const btn = document.getElementById('scanline-btn');
            if (overlay.style.display === 'none') {
                overlay.style.display = 'block';
                btn.textContent = 'CRT FX: ON';
            } else {
                overlay.style.display = 'none';
                btn.textContent = 'CRT FX: OFF';
            }
        }

        async function changePersona(persona) {
            await fetch('/api/persona', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ persona })
            });
        }

        function sendQuickPrompt(promptText) {
            document.getElementById('user-input').value = promptText;
            sendMessage();
        }

        // Load a Swarm/Council result into the prompt box so it becomes the user's
        // next command/query (editable, ready to send) — instead of auto-posting it.
        function useAsNextPrompt(text) {
            const ta = document.getElementById('user-input');
            if (!ta) return;
            ta.value = (text || '').trim();
            ta.focus();
            ta.dispatchEvent(new Event('input')); // trigger autosize if any
            ta.scrollIntoView({ block: 'center' });
            // reveal the chat input by closing the swarm/council modals
            ['swarm-modal', 'council-modal'].forEach(function (id) {
                const m = document.getElementById(id);
                if (m) m.classList.add('hidden');
            });
        }

        // Append a "Use as next prompt" button that loads a result into the prompt
        // box. Built via addEventListener so the full (long) response needs no inline
        // string escaping.
        function _addUseAsPrompt(container, text) {
            if (!container) return;
            const bar = document.createElement('div');
            bar.className = 'mt-2';
            const b = document.createElement('button');
            b.type = 'button';
            b.className = 'px-2 py-0.5 bg-cyan-950 hover:bg-cyan-900 text-cyan-300 rounded border border-cyan-800/50 text-[10px]';
            b.textContent = '↳ Use as next prompt';
            b.addEventListener('click', function () { useAsNextPrompt(text); });
            bar.appendChild(b);
            container.appendChild(bar);
        }

        function renderSources(container, sources) {
            if (!sources || !sources.length) return;
            const wrap = document.createElement('div');
            wrap.className = 'mt-2 pt-2 border-t border-slate-800/50';
            const head = document.createElement('div');
            head.className = 'text-[10px] uppercase tracking-wide text-slate-500 mb-1';
            head.textContent = 'Sources';
            wrap.appendChild(head);
            const list = document.createElement('ol');
            list.className = 'list-decimal list-inside space-y-0.5 text-[11px]';
            sources.forEach(function(s){
                const li = document.createElement('li');
                const a = document.createElement('a');
                a.href = s.url || '#';
                a.target = '_blank';
                a.rel = 'noopener noreferrer';
                a.className = 'text-cyan-400 hover:text-cyan-300 underline decoration-dotted break-all';
                a.textContent = s.title || s.url || 'source';
                li.appendChild(a);
                list.appendChild(li);
            });
            wrap.appendChild(list);
            container.appendChild(wrap);
        }

        function appendMessage(sender, thought, text, modelName, sources) {
            const container = document.getElementById('chat-container');
            const isUser = sender === 'user';
            
            const messageDiv = document.createElement('div');
            messageDiv.className = `flex items-start space-x-4 max-w-4xl ${isUser ? 'ml-auto flex-row-reverse space-x-reverse' : ''}`;
            
            const avatar = document.createElement('div');
            avatar.className = `w-8 h-8 rounded flex items-center justify-center font-bold text-xs shrink-0 ${isUser ? 'bg-emerald-600/20 border border-emerald-500 text-emerald-400' : 'bg-cyan-600/20 border border-cyan-500 text-cyan-400'}`;
            avatar.textContent = isUser ? 'U' : 'Ω';
            
            const contentDiv = document.createElement('div');
            contentDiv.className = `rounded-xl p-4 text-xs shadow-lg space-y-3 ${isUser ? 'bg-emerald-950/40 border border-emerald-800/60 text-emerald-100' : 'bg-slate-900/90 border border-indigo-900/60 text-slate-200 w-full'}`;
            
            if (!isUser && thought && window.showChainOfThought !== false) {
                // Qwen Code CLI style: thinking in a <details> collapsed by default,
                // labelled "Thought for Xs" with an amber dot (primary content is the response).
                const modelLabel = modelName || window.currentModelFilename?.split('/').pop()?.split('\\').pop() || 'selected model';
                const tokCount = String(thought).trim().split(' ').filter(Boolean).length;
                const thinkDetails = document.createElement('details');
                thinkDetails.open = false; // collapsed by default, like Qwen Code CLI
                thinkDetails.className = 'group qwen-think bg-[#0f1115] border-l-2 border-amber-700/70 rounded-r-lg overflow-hidden';
                thinkDetails.style.marginBottom = '0.5rem';

                const summary = document.createElement('summary');
                summary.className = 'flex items-center gap-2 px-3 py-2 text-[11px] font-medium text-slate-400 cursor-pointer select-none hover:bg-[#151821] list-none';
                summary.innerHTML = `<span class="w-1.5 h-1.5 rounded-full bg-amber-500"></span><span class="text-slate-300">Thought for ${modelLabel}</span><span class="text-slate-600 font-mono">· ~${tokCount} tokens</span><span class="ml-auto text-slate-600 group-open:rotate-180 text-[10px]">▾</span>`;

                const meta = document.createElement('div');
                meta.className = 'hidden';
                meta.textContent = '';

                const thinkBody = document.createElement('div');
                thinkBody.className = 'px-3.5 py-2.5 text-[11px] leading-[1.65] text-[#8b8d98] font-mono whitespace-pre-wrap bg-transparent max-h-[36vh] overflow-y-auto border-t border-slate-800/40 italic';
                thinkBody.textContent = thought;

                thinkDetails.appendChild(summary);
                thinkDetails.appendChild(meta);
                thinkDetails.appendChild(thinkBody);
                contentDiv.appendChild(thinkDetails);
            } else if (!isUser && thought && window.showChainOfThought === false) {
                // hidden but accessible via toggle - show collapsed badge
                const badge = document.createElement('div');
                badge.className = 'text-[10px] text-slate-500 italic px-1';
                badge.textContent = '🧠 Chain-of-Thought hidden (enable via 🧠 CoT button or Settings)';
                contentDiv.appendChild(badge);
            }
            
            const textBody = document.createElement('div');
            textBody.className = 'prose prose-invert max-w-none text-slate-200 text-xs';
            if (isUser) {
                textBody.textContent = text;
            } else {
                try {
                    if (typeof marked !== 'undefined' && marked && typeof marked.parse === 'function') {
                        textBody.innerHTML = marked.parse(text);
                    } else if (typeof marked !== 'undefined' && typeof marked === 'function') {
                        // older marked versions export function directly
                        textBody.innerHTML = marked(text);
                    } else {
                        throw new Error('marked not ready');
                    }
                } catch(e) {
                    // fallback: plaintext + line breaks, no crash
                    textBody.innerHTML = String(text).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/\n/g,'<br>');
                    console.warn('[Ashen] marked.parse failed, used fallback:', e);
                }
            }
            contentDiv.appendChild(textBody);
            
            // footer with model attribution + self-improvement feedback
            if (!isUser) {
                const footer = document.createElement('div');
                footer.className = 'text-[10px] text-slate-600 font-mono pt-2 border-t border-slate-800/50 flex justify-between items-center';
                const m = modelName || window.currentModelFilename?.split('/').pop()?.split('\\').pop() || 'unknown';
                footer.innerHTML = `<span>◈ ${m}</span><span>${new Date().toLocaleTimeString()}</span>`;
                contentDiv.appendChild(footer);
                const fbBar = document.createElement('div');
                fbBar.className = 'flex items-center gap-1.5 pt-1 -mb-1';
                const esc = s => String(s||'').replace(/`/g,'').replace(/"/g,'&quot;').slice(0,700);
                const promptForFb = (window._lastUserMsg || '').slice(0,600);
                fbBar.innerHTML = `
                    <span class="text-[10px] text-slate-500">Was this helpful?</span>
                    <button onclick="submitFeedback('up', this)" class="px-2 py-1 bg-slate-800 hover:bg-emerald-900/60 text-emerald-400 rounded border border-slate-700 text-[11px] transition">👍</button>
                    <button onclick="openCorrectionModal()" class="px-2 py-1 bg-slate-800 hover:bg-red-900/60 text-red-400 rounded border border-slate-700 text-[11px] transition">👎</button>
                    <button onclick="regenerateLastWithCritique(this)" class="px-2 py-1 bg-amber-950/40 hover:bg-amber-900/40 text-amber-300 rounded border border-amber-800/40 text-[11px] transition" title="Self-critique regenerate">🔄 Retry</button>
                    <span class="fb-status text-[10px] text-emerald-400 font-mono ml-2"></span>
                `;
                // stash data on bar for handlers
                fbBar.dataset.prompt = promptForFb;
                fbBar.dataset.response = String(text||'').slice(0,800);
                fbBar.dataset.thought = String(thought||'').slice(0,600);
                fbBar.dataset.model = m;
                contentDiv.appendChild(fbBar);
                renderSources(contentDiv, sources);
            }
            
            messageDiv.appendChild(avatar);
            messageDiv.appendChild(contentDiv);
            container.appendChild(messageDiv);
            container.scrollTop = container.scrollHeight;
        }

        async function sendMessage() {
            const input = document.getElementById('user-input');
            const msg = input.value.trim();
            if (!msg) return;

            window._lastUserMsg = msg;
            input.value = '';
            input.style.height = 'auto';

            appendMessage('user', '', msg);

            const sendBtn = document.getElementById('send-btn');
            sendBtn.disabled = true;
            sendBtn.textContent = 'REASONING...';

            // create live streaming assistant placeholder
            const container = document.getElementById('chat-container');
            const modelLabelInit = window.currentModelFilename?.split('/').pop()?.split('\\').pop() || 'selected model';
            let thinkBuffer = "";
            let responseBuffer = "";
            let toolBuffer = "";
            // build live message DOM (like appendMessage but with empty bodies we can update)
            const messageDiv = document.createElement('div');
            messageDiv.className = 'flex items-start space-x-4 max-w-4xl';
            const avatar = document.createElement('div');
            avatar.className = 'w-8 h-8 rounded bg-cyan-600/20 border border-cyan-500 flex items-center justify-center font-bold text-xs text-cyan-400 shrink-0 mt-1';
            avatar.textContent = 'Ω';
            const contentDiv = document.createElement('div');
            contentDiv.className = 'bg-slate-900/90 border border-indigo-900/60 rounded-xl p-4 text-slate-200 text-sm shadow-xl space-y-2 w-full';
            let thinkDetails = null, thinkBody = null, meta = null;
            let _thinkStart = Date.now();
            let _thinkTimer = null;
            if (window.showChainOfThought !== false) {
                thinkDetails = document.createElement('details');
                thinkDetails.open = true;
                thinkDetails.className = 'group qwen-think bg-[#0f1115] border-l-2 border-slate-700 rounded-r-lg overflow-hidden';
                const summary = document.createElement('summary');
                summary.className = 'flex items-center gap-2 px-3 py-2 text-[11px] font-medium text-slate-400 cursor-pointer select-none hover:bg-[#151821] list-none';
                summary.innerHTML = '<span class="w-1.5 h-1.5 rounded-full bg-amber-500 animate-pulse"></span><span class="text-slate-300">Thinking</span><span class="text-slate-600 font-mono">· '+modelLabelInit+' · <span class="think-timer">0.0s</span></span><span class="ml-auto text-slate-600 group-open:rotate-180 text-[10px]">▾</span>';
                meta = document.createElement('div');
                meta.className = 'hidden';
                meta.textContent = '';
                thinkBody = document.createElement('div');
                thinkBody.className = 'px-3.5 py-2.5 text-[11px] leading-[1.65] text-[#8b8d98] font-mono whitespace-pre-wrap bg-transparent max-h-[36vh] overflow-y-auto border-t border-slate-800/40 italic';
                thinkBody.innerHTML = '<span class="text-slate-600">… reasoning started …</span><span class="inline-block w-2 h-3 bg-slate-600 animate-pulse ml-1 align-text-bottom"></span>';
                thinkDetails.appendChild(summary); thinkDetails.appendChild(meta); thinkDetails.appendChild(thinkBody);
                contentDiv.appendChild(thinkDetails);
                _thinkTimer = setInterval(function(){ var el = thinkDetails && thinkDetails.querySelector('.think-timer'); if(el) el.textContent = ((Date.now()-_thinkStart)/1000).toFixed(1)+'s'; }, 120);
            }
            const textBody = document.createElement('div');
            textBody.className = 'prose prose-invert max-w-none text-slate-200 text-xs min-h-[1.2em]';
            textBody.innerHTML = '<span class="text-slate-500 italic">∅ awaiting response — thought streaming first…</span>';
            contentDiv.appendChild(textBody);
            const liveFooter = document.createElement('div');
            liveFooter.className = 'text-[10px] text-slate-600 font-mono pt-2 border-t border-slate-800/50 flex justify-between items-center';
            liveFooter.innerHTML = '<span>◈ '+modelLabelInit+' <span class="text-cyan-500">● streaming</span></span><span>'+new Date().toLocaleTimeString()+'</span>';
            contentDiv.appendChild(liveFooter);
            messageDiv.appendChild(avatar); messageDiv.appendChild(contentDiv);
            container.appendChild(messageDiv);
            container.scrollTop = container.scrollHeight;

            const updateThink = (chunk) => {
                thinkBuffer += chunk;
                if (thinkBody) {
                    var esc = thinkBuffer.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
                    thinkBody.innerHTML = esc + '<span class="inline-block w-2 h-3 bg-slate-600 animate-pulse ml-0.5 align-text-bottom"></span>';
                    thinkBody.scrollTop = thinkBody.scrollHeight;
                }
                container.scrollTop = container.scrollHeight;
            };
            const updateResponse = (chunk) => {
                responseBuffer += chunk;
                try {
                    if (typeof marked !== 'undefined' && marked && typeof marked.parse === 'function') {
                        textBody.innerHTML = marked.parse(responseBuffer);
                    } else { textBody.textContent = responseBuffer; }
                } catch(e){ textBody.textContent = responseBuffer; }
                container.scrollTop = container.scrollHeight;
            };

            try {
                const res = await fetch('/api/chat/stream', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ message: msg })
                });
                if (!res.ok || !res.body) throw new Error('stream failed '+res.status);
                const reader = res.body.getReader();
                const decoder = new TextDecoder();
                let buf = "";
                let finalThought = "", finalResponse = "", finalModel = modelLabelInit, finalPath = window.currentModelFilename, finalSources = [];
                while (true) {
                    const {value, done} = await reader.read();
                    if (done) break;
                    buf += decoder.decode(value, {stream:true});
                    let lines = buf.split("\n");
                    buf = lines.pop();
                    for (let line of lines) {
                        if (!line.trim()) continue;
                        let ev; try{ ev = JSON.parse(line); }catch(e){ continue; }
                        if (ev.type === 'start') {
                            finalModel = ev.model || finalModel; finalPath = ev.model_path || finalPath;
                            if (thinkDetails) {
                                const lbl = thinkDetails.querySelector('span span');
                                if(lbl) lbl.innerHTML = finalModel+' <span class="animate-pulse text-cyan-400">● live</span>';
                            }
                        } else if (ev.type === 'thought_delta') {
                            updateThink(ev.chunk||"");
                        } else if (ev.type === 'thought_done') {
                            if(_thinkTimer){ clearInterval(_thinkTimer); _thinkTimer=null; }
                            if(thinkDetails){
                                var sm = thinkDetails.querySelector('summary');
                                if(sm){
                                    var secs = ((Date.now()-_thinkStart)/1000).toFixed(1);
                                    var toks = thinkBuffer.trim().split(' ').filter(Boolean).length;
                                    sm.innerHTML = '<span class="w-1.5 h-1.5 rounded-full bg-amber-500"></span><span class="text-slate-300">Thought for '+finalModel+'</span><span class="text-slate-600 font-mono">· '+secs+'s · ~'+toks+' tokens</span><span class="ml-auto text-slate-600 group-open:rotate-180 text-[10px]">▾</span>';
                                }
                                thinkBody.style.opacity = '0.9';
                            }
                        } else if (ev.type === 'response_delta') {
                            if (thinkBody && thinkBuffer && !responseBuffer) {
                                // first response chunk — clear placeholder
                            }
                            updateResponse(ev.chunk||"");
                        } else if (ev.type === 'tool_start') {
                            toolBuffer += "\n[TOOL: "+(ev.tool||"")+"]";
                            if(meta) meta.innerHTML = '<span>model: '+finalModel+'</span><span class="text-amber-400">tool: '+ev.tool+'</span>';
                        } else if (ev.type === 'tool_result') {
                            toolBuffer += "\n"+(ev.observation||"").slice(0,120);
                        } else if (ev.type === 'done') {
                            finalThought = ev.thought || thinkBuffer;
                            finalResponse = ev.response || responseBuffer;
                            finalModel = ev.model || finalModel; finalPath = ev.model_path || finalPath;
                            finalSources = ev.sources || finalSources;
                        } else if (ev.type === 'error') {
                            updateResponse("\n\n**Error:** "+ev.message);
                        }
                    }
                }
                // handle trailing buf
                if (buf.trim()) { try{ const ev=JSON.parse(buf); if(ev.type==='done'){ finalThought=ev.thought||thinkBuffer; finalResponse=ev.response||responseBuffer; }}catch(e){}}
                // finalize DOM to match final static state (with feedback bar)
                // replace placeholder footer with proper one + feedback bar like appendMessage does
                // update thought body to final value (in case streamed chunks were partial)
                if(_thinkTimer){ clearInterval(_thinkTimer); _thinkTimer=null; }
                if (thinkBody) {
                    var _final = finalThought || thinkBuffer || "(no thought)";
                    var _esc = _final.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
                    thinkBody.innerHTML = _esc;
                    thinkBody.style.opacity = '0.9';
                }
                // final response parse
                try {
                    if (typeof marked !== 'undefined' && marked && typeof marked.parse === 'function') {
                        textBody.innerHTML = marked.parse(finalResponse);
                    } else { textBody.textContent = finalResponse; }
                } catch(e){ textBody.innerHTML = String(finalResponse).replace(/</g,'&lt;').replace(/\n/g,'<br>'); }
                liveFooter.innerHTML = '<span>◈ '+finalModel+'</span><span>'+new Date().toLocaleTimeString()+'</span>';
                if (thinkDetails) {
                    const sm2 = thinkDetails.querySelector('summary');
                    if(sm2){
                        var fsecs = ((Date.now()-_thinkStart)/1000).toFixed(1);
                        var ftoks = (finalThought||thinkBuffer).trim().split(' ').filter(Boolean).length;
                        sm2.innerHTML = '<span class="w-1.5 h-1.5 rounded-full bg-amber-500"></span><span class="text-slate-300">Thought for '+finalModel+'</span><span class="text-slate-600 font-mono">· '+fsecs+'s · ~'+ftoks+' tokens</span><span class="ml-auto text-slate-600 group-open:rotate-180 text-[10px]">▾</span>';
                    }
                    // Qwen Code CLI: thinking collapses by default once the turn completes
                    setTimeout(function(){ if(thinkDetails) thinkDetails.open = false; }, 900);
                }
                // add feedback bar like appendMessage
                const fbBar = document.createElement('div');
                fbBar.className = 'flex items-center gap-1.5 pt-1 -mb-1';
                fbBar.dataset.prompt = (window._lastUserMsg||'').slice(0,600);
                fbBar.dataset.response = String(finalResponse||'').slice(0,800);
                fbBar.dataset.thought = String(finalThought||'').slice(0,600);
                fbBar.dataset.model = finalModel;
                fbBar.innerHTML = '<span class="text-[10px] text-slate-500">Was this helpful?</span> <button onclick="submitFeedback(\'up\', this)" class="px-2 py-1 bg-slate-800 hover:bg-emerald-900/60 text-emerald-400 rounded border border-slate-700 text-[11px] transition">👍</button> <button onclick="openCorrectionModal()" class="px-2 py-1 bg-slate-800 hover:bg-red-900/60 text-red-400 rounded border border-slate-700 text-[11px] transition">👎</button> <button onclick="regenerateLastWithCritique(this)" class="px-2 py-1 bg-amber-950/40 hover:bg-amber-900/40 text-amber-300 rounded border border-amber-800/40 text-[11px] transition" title="Self-critique regenerate">🔄 Retry</button> <span class="fb-status text-[10px] text-emerald-400 font-mono ml-2"></span>';
                contentDiv.appendChild(fbBar);
                renderSources(contentDiv, finalSources);
                if (finalModel) window.currentModelFilename = finalPath || finalModel;
            } catch (err) {
                // fallback to non-stream endpoint
                try{
                    const res2 = await fetch('/api/chat', { method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify({message: msg})});
                    const data = await res2.json();
                    // remove live placeholder and use normal append
                    messageDiv.remove();
                    appendMessage('assistant', data.thought, data.response, data.model, data.sources);
                    if (data.model) window.currentModelFilename = data.model_path || data.model;
                }catch(e2){
                    textBody.innerHTML = '<span class="text-red-400">Failed: '+String(err.message||err)+'</span>';
                }
            } finally {
                sendBtn.disabled = false;
                sendBtn.textContent = 'EXECUTE';
            }
        }

        async function clearHistory() {
            await fetch('/api/clear', { method: 'POST' });
            // Reset workspace context indicator
            const ctxLabel = document.getElementById('workspace-context-label');
            if (ctxLabel) {
                ctxLabel.textContent = '📁 None';
                ctxLabel.title = 'No workspace context active';
            }
            const container = document.getElementById('chat-container');
            container.innerHTML = `
                <div class="flex items-start space-x-4 max-w-4xl">
                    <div class="w-8 h-8 rounded bg-cyan-600/20 border border-cyan-500 flex items-center justify-center font-bold text-xs text-cyan-400 shrink-0">Ω</div>
                    <div class="bg-slate-900/90 border border-indigo-900/60 rounded-xl p-4 text-slate-200 text-sm shadow-xl space-y-2">
                        <p class="font-bold text-cyan-400">Session Purged.</p>
                        <p class="text-xs text-slate-300">Memory reset successfully. Ashen AI core ready for new directives.</p>
                    </div>
                </div>
            `;
        }

        function exportChat() {
            const container = document.getElementById('chat-container');
            let textContent = "# Ashen AI Hub - Session Export\\n\\n";
            container.querySelectorAll('.flex.items-start').forEach(el => {
                const isUser = el.classList.contains('ml-auto');
                const role = isUser ? "User" : "Ashen AI";
                const text = el.querySelector('.prose')?.innerText || "";
                textContent += `### ${role}\\n${text}\\n\\n`;
            });
            const blob = new Blob([textContent], { type: 'text/markdown' });
            const url = URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = `ashen_ai_session_${Date.now()}.md`;
            a.click();
        }

        // --- Session Management ---
        async function loadSessionList() {
            try {
                const res = await fetch('/api/sessions');
                const data = await res.json();
                const listEl = document.getElementById('session-list');
                const labelEl = document.getElementById('current-session-label');

                if (!listEl) return;

                if (data.sessions && data.sessions.length > 0) {
                    listEl.innerHTML = '';
                    data.sessions.forEach(s => {
                        const item = document.createElement('div');
                        const isCurrent = s.id === data.current;
                        const dateStr = s.updated ? new Date(s.updated).toLocaleString() : '';
                        item.className = `p-2 rounded cursor-pointer transition flex items-center justify-between ${isCurrent ? 'bg-blue-950/40 border border-blue-800/60' : 'bg-slate-900/50 hover:bg-slate-800 border border-transparent'}`;
                        item.innerHTML = `
                            <div class="flex-1 min-w-0">
                                <div class="text-xs font-semibold ${isCurrent ? 'text-blue-300' : 'text-slate-300'} truncate">${s.name || 'Untitled'}</div>
                                <div class="text-[10px] text-slate-500">${s.message_count} msgs · ${dateStr}</div>
                            </div>
                            <div class="flex gap-1 ml-2 shrink-0">
                                ${!isCurrent ? `<button onclick="event.stopPropagation(); loadSession('${s.id}')" class="text-[10px] px-1.5 py-0.5 bg-blue-950 hover:bg-blue-900 text-blue-300 rounded border border-blue-800/40">Load</button>` : '<span class="text-[10px] text-blue-400">● Active</span>'}
                                <button onclick="event.stopPropagation(); renameSessionPrompt('${s.id}')" class="text-[10px] px-1.5 py-0.5 bg-slate-800 hover:bg-slate-700 text-slate-300 rounded">✎</button>
                                <button onclick="event.stopPropagation(); deleteSessionConfirm('${s.id}')" class="text-[10px] px-1.5 py-0.5 bg-red-950 hover:bg-red-900 text-red-300 rounded">×</button>
                            </div>`;
                        item.onclick = () => loadSession(s.id);
                        listEl.appendChild(item);
                    });
                } else {
                    listEl.innerHTML = '<div class="text-slate-500 text-xs text-center py-4">No past sessions</div>';
                }

                if (labelEl && data.current) {
                    const currentData = data.sessions.find(s => s.id === data.current);
                    labelEl.textContent = currentData ? `Active: ${currentData.name}` : 'No active session';
                }
            } catch (e) { /* non-blocking */ }
        }

        async function createNewSession() {
            try {
                const res = await fetch('/api/sessions/new', { method: 'POST' });
                const data = await res.json();
                if (data.status === 'created') {
                    // Clear chat UI
                    const container = document.getElementById('chat-container');
                    container.innerHTML = `
                        <div class="flex items-start space-x-4 max-w-4xl">
                            <div class="w-8 h-8 rounded bg-cyan-600/20 border border-cyan-500 flex items-center justify-center font-bold text-xs text-cyan-400 shrink-0">Ω</div>
                            <div class="bg-slate-900/90 border border-indigo-900/60 rounded-xl p-4 text-slate-200 text-sm shadow-xl space-y-2">
                                <p class="font-bold text-cyan-400">New Chat Started</p>
                                <p class="text-xs text-slate-300">Fresh session ready for your next prompt.</p>
                            </div>
                        </div>`;
                    // Reset workspace context indicator
                    const ctxLabel = document.getElementById('workspace-context-label');
                    if (ctxLabel) { ctxLabel.textContent = '📁 None'; ctxLabel.title = 'No workspace context active'; }
                    toggleSessionsPanel(false);
                    loadSessionList();
                }
            } catch (e) { alert('Failed to create session.'); }
        }

        async function loadSession(sid) {
            try {
                const res = await fetch('/api/sessions/load', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ session_id: sid })
                });
                const data = await res.json();
                if (data.status === 'loaded') {
                    // Restore chat history in UI
                    const container = document.getElementById('chat-container');
                    const messages = data.session.history || [];
                    let html = '';
                    if (messages.length === 0) {
                        html = `<div class="flex items-start space-x-4 max-w-4xl">
                            <div class="w-8 h-8 rounded bg-cyan-600/20 border border-cyan-500 flex items-center justify-center font-bold text-xs text-cyan-400 shrink-0">Ω</div>
                            <div class="bg-slate-900/90 border border-indigo-900/60 rounded-xl p-4 text-slate-200 text-sm shadow-xl">
                                <p class="font-bold text-cyan-400">Loaded "${data.session.name}"</p>
                                <p class="text-xs text-slate-300 mt-1">This session has no messages yet.</p>
                            </div></div>`;
                    } else {
                        messages.forEach(m => {
                            html += renderUserMessage(m.user);
                            const respContent = m.assistant || '';
                            html += renderAssistantResponse(respContent);
                        });
                    }
                    container.innerHTML = html;
                    container.scrollTop = container.scrollHeight;
                    toggleSessionsPanel(false);
                    loadSessionList();
                }
            } catch (e) { alert('Failed to load session.'); }
        }

        async function deleteSessionConfirm(sid) {
            const session = list_sessions_cache?.find(s => s.id === sid);
            const name = session ? session.name : 'this session';
            if (!confirm(`Delete session "${name}"? This cannot be undone.`)) return;
            try {
                const res = await fetch('/api/sessions/delete', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ session_id: sid })
                });
                const data = await res.json();
                if (data.status === 'deleted') {
                    const container = document.getElementById('chat-container');
                    container.innerHTML = `<div class="flex items-center justify-center text-red-400 text-xs">Session deleted.</div>`;
                    const ctxLabel = document.getElementById('workspace-context-label');
                    if (ctxLabel) { ctxLabel.textContent = '📁 None'; ctxLabel.title = 'No workspace context active'; }
                    loadSessionList();
                }
            } catch (e) { alert('Failed to delete session.'); }
        }

        async function renameSessionPrompt(sid) {
            const session = list_sessions_cache?.find(s => s.id === sid);
            const name = session ? session.name : 'Untitled';
            const newName = prompt('Rename session:', name);
            if (!newName || newName.trim() === '') return;
            try {
                await fetch('/api/sessions/rename', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ session_id: sid, name: newName.trim() })
                });
                loadSessionList();
            } catch (e) { alert('Failed to rename session.'); }
        }

        let list_sessions_cache = null;
        // Cache & periodically refresh session list
        loadSessionList();
        setInterval(loadSessionList, 30000);

        // Initialize workspace file tree and load saved settings on page load
        loadWorkspaceDir('');
        loadSettings();
        
        // Set current model filename in JS scope
        window.currentModelFilename = '{{current_model_filename}}';
    </script>
</body>
</html>
"""

def scan_local_pc_models():
    found_models = []
    search_dirs = [
        '.',
        os.path.expanduser('~/.cache/huggingface/hub'),
        os.path.expanduser('~/.lmstudio/models'),
        os.path.expanduser('~/.lmstudio/cache'),
        os.path.expanduser('~/Downloads')
    ]
    for d in search_dirs:
        if os.path.exists(d):
            try:
                for root, dirs, files in os.walk(d):
                    dirs[:] = [d2 for d2 in dirs if not d2.startswith('.')]
                    # HuggingFace model directory -> list the dir, not its weights.
                    if _is_hf_model_dir(root):
                        full_path = os.path.abspath(root)
                        try:
                            size_mb = round(sum(
                                os.path.getsize(os.path.join(root, f))
                                for f in os.listdir(root)
                                if f.endswith('.safetensors') or f.endswith('.bin')
                            ) / (1024 * 1024), 1)
                        except OSError:
                            size_mb = 0.0
                        found_models.append({
                            'name': os.path.basename(root),
                            'path': full_path,
                            'size_mb': size_mb,
                            'active': (os.path.normpath(full_path) == os.path.normpath(current_model_filename)),
                            'type': 'qwen-hf'
                        })
                        continue
                    for file in files:
                        if file.endswith(('.pk1', '.pt', '.pth', '.gguf')):
                            full_path = os.path.abspath(os.path.join(root, file))
                            size_mb = round(os.path.getsize(full_path) / (1024 * 1024), 1)
                            found_models.append({
                                'name': file,
                                'path': full_path,
                                'size_mb': size_mb,
                                'active': (os.path.basename(full_path) == os.path.basename(current_model_filename))
                            })
            except:
                pass
    return found_models

def get_directory_listing(dir_path):
    if not dir_path:
        target_dir = os.getcwd()
    else:
        target_dir = dir_path

    target_dir = os.path.abspath(target_dir)
    items = []
    parent_path = None

    try:
        parent = os.path.dirname(target_dir)
        if parent != target_dir:
            parent_path = parent

        for entry in os.scandir(target_dir):
            if entry.name.startswith('.'):
                continue
            is_dir = entry.is_dir()
            items.append({
                'name': entry.name,
                'path': entry.path.replace('\\', '/'),  # Normalize to forward slashes for URLs
                'is_dir': is_dir
            })
        items.sort(key=lambda x: (not x['is_dir'], x['name'].lower()))
    except Exception as e:
        pass

    return {
        'current_path': target_dir,
        'parent_path': parent_path,
        'items': items
    }

# --- Session Management Helpers ---

def _session_file(session_id):
    return os.path.join(SESSIONS_DIR, f"{session_id}.json")

def save_session(session_id, data):
    """Save a full session (history + workspace context + settings + persona) to disk."""
    filepath = _session_file(session_id)
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

def load_session(session_id):
    """Load a session from disk. Returns dict or None."""
    filepath = _session_file(session_id)
    if not os.path.exists(filepath):
        return None
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)

def list_sessions():
    """Return a list of all sessions sorted by date (newest first)."""
    sessions = []
    for fname in os.listdir(SESSIONS_DIR):
        if not fname.endswith('.json'):
            continue
        sid = fname[:-5]  # strip .json
        filepath = os.path.join(SESSIONS_DIR, fname)
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
            sessions.append({
                'id': sid,
                'name': data.get('name', 'Untitled'),
                'created': data.get('created_at', ''),
                'updated': data.get('updated_at', ''),
                'message_count': len(data.get('history', [])),
                'persona': data.get('persona', 'ashen_ai_agent'),
                'size_bytes': os.path.getsize(filepath)
            })
        except Exception:
            pass
    sessions.sort(key=lambda s: s.get('updated', ''), reverse=True)
    return sessions

def delete_session(session_id):
    """Delete a session file. Returns True if deleted."""
    filepath = _session_file(session_id)
    if os.path.exists(filepath):
        os.remove(filepath)
        return True
    return False

def create_new_session():
    """Create a new session ID and return it."""
    global current_session_id
    ts = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    current_session_id = f"session_{ts}"
    # Save initial blank session
    save_session(current_session_id, {
        'name': 'Untitled',
        'created_at': datetime.datetime.now().isoformat(),
        'updated_at': datetime.datetime.now().isoformat(),
        'persona': 'ashen_ai_agent',
        'settings': {
            'temperature': 0.7,
            'top_k': 40,
            'top_p': 0.9,
            'max_new_tokens': 250,
            'context_length': 8192,
            'gpu_layers': 16,
            'repeat_penalty': 1.1
        },
        'history': [],
        'workspace_context': ''
    })
    return current_session_id

def append_to_session(session_id, user_msg, assistant_msg):
    """Append a message pair to the session's history."""
    global current_session_id
    data = load_session(session_id)
    if not data:
        return
    data['history'].append({'user': user_msg, 'assistant': assistant_msg})
    data['updated_at'] = datetime.datetime.now().isoformat()
    save_session(session_id, data)
    current_session_id = session_id

class ChatHandler(http.server.SimpleHTTPRequestHandler):
    def do_GET(self):
        parsed_url = urlparse(self.path)
        path = parsed_url.path
        query = parse_qs(parsed_url.query)

        if path == '/' or path == '':
            self.send_response(200)
            self.send_header('Content-Type', 'text/html; charset=utf-8')
            self.end_headers()
            # inject live values so browser doesn't start with {{current_model_filename}} placeholder
            html = HTML_PAGE.replace('{{current_model_filename}}', current_model_filename.replace('\\', '/'))
            self.wfile.write(html.encode('utf-8'))
        elif path == '/api/models':
            models = scan_local_pc_models()
            seen = set()
            unique_models = []
            for m in models:
                if m['path'] not in seen:
                    seen.add(m['path'])
                    unique_models.append(m)
            self.send_response(200)
            self.send_header('Content-Type', 'application/json; charset=utf-8')
            self.end_headers()
            self.wfile.write(json.dumps({'models': unique_models}).encode('utf-8'))
        elif path == '/api/hf-search':
            q = query.get('q', [''])[0]
            hf_models = []
            try:
                url = f"https://huggingface.co/api/models?search={q}&limit=20&sort=downloads&direction=-1"
                resp = requests.get(url, timeout=10)
                data = resp.json()
                for item in data:
                    hf_models.append({
                        'repo_id': item.get('id', ''),
                        'downloads': item.get('downloads', 0),
                        'likes': item.get('likes', 0)
                    })
            except Exception as e:
                pass
            self.send_response(200)
            self.send_header('Content-Type', 'application/json; charset=utf-8')
            self.end_headers()
            self.wfile.write(json.dumps({'models': hf_models}).encode('utf-8'))
        elif path == '/api/sessions':
            sessions = list_sessions()
            self.send_response(200)
            self.send_header('Content-Type', 'application/json; charset=utf-8')
            self.end_headers()
            self.wfile.write(json.dumps({'sessions': sessions, 'current': current_session_id}).encode('utf-8'))
        elif path == '/api/workspace/list':
            dir_path = query.get('dir', [''])[0]
            listing = get_directory_listing(dir_path)
            self.send_response(200)
            self.send_header('Content-Type', 'application/json; charset=utf-8')
            self.end_headers()
            self.wfile.write(json.dumps(listing).encode('utf-8'))
        elif path == '/api/settings/load':
            # support GET as well as POST for JS convenience
            try:
                loaded = load_settings_from_json()
                self.send_response(200)
                self.send_header('Content-Type', 'application/json; charset=utf-8')
                self.end_headers()
                self.wfile.write(json.dumps({'status': 'success', 'settings': loaded}).encode('utf-8'))
            except Exception as e:
                self.send_response(500)
                self.send_header('Content-Type', 'application/json; charset=utf-8')
                self.end_headers()
                self.wfile.write(json.dumps({'status': 'error', 'message': str(e)}).encode('utf-8'))
        elif path == '/api/self-improve':
            # GET stats + log for dashboard
            log = _load_improvement_log()
            feedback = _load_feedback()
            total_entries = len(log.get('entries', []))
            gib_fixes = log.get('stats', {}).get('gibberish_fixes', 0)
            self.send_response(200)
            self.send_header('Content-Type', 'application/json; charset=utf-8')
            self.end_headers()
            self.wfile.write(json.dumps({
                'stats': log.get('stats', {}),
                'entries': log.get('entries', [])[-20:],
                'suggestions': log.get('suggestions', [])[-10:],
                'feedback_count': len(feedback),
                'feedback_recent': feedback[-10:],
                'total_entries': total_entries,
                'gibberish_rate': round(gib_fixes / max(1, total_entries) * 100, 1) if total_entries else 0
            }).encode('utf-8'))
        elif path == '/api/feedback':
            fb = _load_feedback()
            self.send_response(200)
            self.send_header('Content-Type', 'application/json; charset=utf-8')
            self.end_headers()
            self.wfile.write(json.dumps({'feedback': fb[-50:]}).encode('utf-8'))
        elif path == '/api/swarm':
            # GET swarm roles & last runs preview
            log = _load_improvement_log()
            swarm_runs = [e for e in log.get('entries', []) if e.get('type')=='swarm'][-10:]
            self.send_response(200)
            self.send_header('Content-Type', 'application/json; charset=utf-8')
            self.end_headers()
            self.wfile.write(json.dumps({
                'roles': [{'name': r[0], 'desc': r[1]} for r in _SWARM_ROLES],
                'recent_runs': swarm_runs,
                'model': os.path.basename(current_model_filename),
                'model_path': current_model_filename
            }).encode('utf-8'))
        elif path == '/api/council':
            log = _load_improvement_log()
            council_runs = [e for e in log.get('entries', []) if e.get('type')=='council'][-10:]
            self.send_response(200)
            self.send_header('Content-Type', 'application/json; charset=utf-8')
            self.end_headers()
            self.wfile.write(json.dumps({
                'roles': [{'name': r[0], 'desc': r[1]} for r in _COUNCIL_CRITIC_ROLES],
                'proposers': [{'name': r[0], 'desc': r[1]} for r in _SWARM_ROLES],
                'recent_runs': council_runs,
                'model': os.path.basename(current_model_filename),
                'model_path': current_model_filename
            }).encode('utf-8'))
        else:
            self.send_response(404)
            self.end_headers()

    def do_POST(self):
        global model, reasoner, current_model_filename, current_session_id, settings
        
        if self.path == '/api/workspace/read':
            content_length = int(self.headers.get('Content-Length', 0))
            body = self.rfile.read(content_length)
            data = json.loads(body.decode('utf-8'))
            file_path = data.get('path', '')
            
            # Debug logging
            import sys
            print(f"[DEBUG] Read request for: {repr(file_path)}", file=sys.stderr, flush=True)
            print(f"[DEBUG] cwd: {repr(os.getcwd())}", file=sys.stderr, flush=True)
            
            if not file_path:
                resp_data = {'status': 'error', 'message': 'No file path provided'}
            else:
                target_path = None
                
                # Strategy 1: Use path as-is (should be absolute)
                exists1 = os.path.isfile(file_path)
                print(f"[DEBUG] Strategy 1 (as-is): exists={exists1}", file=sys.stderr, flush=True)
                
                if exists1:
                    target_path = file_path
                
                # Strategy 2: Try backslash normalization (C:/path -> C:\path)
                elif '/' in file_path:
                    win_path = file_path.replace('/', '\\')
                    exists2 = os.path.isfile(win_path)
                    print(f"[DEBUG] Strategy 2 (backslash): {repr(win_path)} exists={exists2}", file=sys.stderr, flush=True)
                    if exists2:
                        target_path = win_path
                
                # Strategy 3: Join with cwd for relative paths
                if not target_path:
                    joined = os.path.join(os.getcwd(), file_path)
                    exists3 = os.path.isfile(joined)
                    print(f"[DEBUG] Strategy 3 (joined): {repr(joined)} exists={exists3}", file=sys.stderr, flush=True)
                    if exists3:
                        target_path = joined
                
                if target_path and os.path.exists(target_path) and os.path.isfile(target_path):
                    try:
                        with open(target_path, 'rb') as f:
                            raw_bytes = f.read()
                        try:
                            content = raw_bytes.decode('utf-8')
                        except UnicodeDecodeError:
                            content = raw_bytes.decode('latin-1')
                        print(f"[DEBUG] SUCCESS: read {len(raw_bytes)} bytes from {target_path}", file=sys.stderr, flush=True)
                        resp_data = {'status': 'success', 'content': content}
                    except Exception as e:
                        print(f"[DEBUG] ERROR reading file: {e}", file=sys.stderr, flush=True)
                        resp_data = {'status': 'error', 'message': str(e)}
                else:
                    resp_data = {'status': 'error', 'message': f'File not found: {repr(file_path)}. Tried all strategies.'}
            
            self.send_response(200)
            self.send_header('Content-Type', 'application/json; charset=utf-8')
            self.end_headers()
            self.wfile.write(json.dumps(resp_data).encode('utf-8'))
            return
        
        if self.path == '/api/chat':
            content_length = int(self.headers.get('Content-Length', 0))
            body = self.rfile.read(content_length)
            data = json.loads(body.decode('utf-8'))
            message = data.get('message', '')

            # Auto-create session if none active
            if not current_session_id:
                current_session_id = create_new_session()
                reasoner.session_id = current_session_id
                # Auto-name session from first message
                sdata = load_session(current_session_id)
                if sdata and sdata.get('name', 'Untitled') == 'Untitled':
                    preview = message[:50].replace('\n', ' ')
                    sdata['name'] = preview + ('...' if len(message) > 50 else '')
                    save_session(current_session_id, sdata)

            try:
                thought, response = reasoner.solve_with_agent(message)
                # Save to session
                append_to_session(current_session_id, message, f"<think>\n{thought}\n</think>\n{response}")
                # include model name + CoT visibility so frontend can label the chain-of-thought per-model
                model_name = os.path.basename(current_model_filename)
                # respect settings toggle but always send - frontend decides to show/hide
                show_cot = settings.get('show_chain_of_thought', True)
                resp_data = {'thought': thought, 'response': response, 'model': model_name, 'model_path': current_model_filename, 'show_chain_of_thought': show_cot}
            except Exception as e:
                resp_data = {'thought': 'Error during Ashen AI execution', 'response': str(e), 'model': os.path.basename(current_model_filename)}

            self.send_response(200)
            self.send_header('Content-Type', 'application/json; charset=utf-8')
            self.end_headers()
            self.wfile.write(json.dumps(resp_data).encode('utf-8'))

        elif self.path == '/api/chat/stream':
            content_length = int(self.headers.get('Content-Length', 0))
            body = self.rfile.read(content_length) if content_length else b'{}'
            try:
                data = json.loads(body.decode('utf-8')) if body else {}
            except:
                data = {}
            message = (data.get('message') or data.get('prompt') or '').strip()
            if not message:
                self.send_response(400)
                self.send_header('Content-Type', 'application/json; charset=utf-8')
                self.end_headers()
                self.wfile.write(json.dumps({'status':'error','message':'No message'}).encode('utf-8'))
            else:
                if not current_session_id:
                    current_session_id = create_new_session()
                    reasoner.session_id = current_session_id
                    sdata = load_session(current_session_id)
                    if sdata and sdata.get('name','Untitled')=='Untitled':
                        preview = message[:50].replace('\n',' ')
                        sdata['name'] = preview + ('...' if len(message)>50 else '')
                        save_session(current_session_id, sdata)
                self.send_response(200)
                self.send_header('Content-Type', 'application/x-ndjson; charset=utf-8')
                self.send_header('Cache-Control', 'no-cache')
                self.send_header('X-Accel-Buffering', 'no')
                # Close after the NDJSON stream ends so the client's reader gets a
                # proper end-of-body (finalizes the UI) and the worker thread is freed.
                self.send_header('Connection', 'close')
                self.end_headers()
                try:
                    final_thought = ""
                    final_response = ""
                    show_cot = settings.get('show_chain_of_thought', True)
                    model_name = os.path.basename(current_model_filename)
                    # header event
                    self.wfile.write((json.dumps({"type":"start","model":model_name,"model_path":current_model_filename,"show_chain_of_thought":show_cot})+"\n").encode('utf-8')); self.wfile.flush()
                    for ev in reasoner.solve_with_agent_stream(message):
                        self.wfile.write((json.dumps(ev)+"\n").encode('utf-8')); self.wfile.flush()
                        if ev.get('type')=='done':
                            final_thought = ev.get('thought','')
                            final_response = ev.get('response','')
                            model_name = ev.get('model', model_name)
                except Exception as e:
                    import traceback
                    self.wfile.write((json.dumps({"type":"error","message":str(e),"trace":traceback.format_exc()})+"\n").encode('utf-8')); self.wfile.flush()
                    final_thought = final_thought or ""
                    final_response = final_response or str(e)
                # save to session after stream
                try:
                    append_to_session(current_session_id, message, f"<think>\n{final_thought}\n</think>\n{final_response}")
                except: pass

        elif self.path == '/api/clear':
            # Save current session before clearing
            if current_session_id and load_session(current_session_id):
                save_session(current_session_id, {
                    'name': load_session(current_session_id)['name'],
                    'persona': reasoner.persona,
                    'settings': {
                        'temperature': reasoner.temperature,
                        'top_k': reasoner.top_k,
                        'top_p': reasoner.top_p,
                        'max_new_tokens': reasoner.max_new_tokens,
                        'context_length': reasoner.context_length,
                        'gpu_layers': reasoner.gpu_layers,
                        'repeat_penalty': reasoner.repeat_penalty
                    },
                    'history': list(reasoner.history),
                    'workspace_context': reasoner.workspace_context
                })
            reasoner.clear_history()
            reasoner.set_workspace_context("")
            self.send_response(200)
            self.send_header('Content-Type', 'application/json; charset=utf-8')
            self.end_headers()
            self.wfile.write(json.dumps({'status': 'cleared'}).encode('utf-8'))

        elif self.path == '/api/persona':
            content_length = int(self.headers.get('Content-Length', 0))
            body = self.rfile.read(content_length)
            data = json.loads(body.decode('utf-8'))
            persona = data.get('persona', 'ashen_ai_agent')
            reasoner.set_persona(persona)
            self.send_response(200)
            self.send_header('Content-Type', 'application/json; charset=utf-8')
            self.end_headers()
            self.wfile.write(json.dumps({'status': 'updated', 'persona': persona}).encode('utf-8'))

        elif self.path == '/api/settings':
            content_length = int(self.headers.get('Content-Length', 0))
            body = self.rfile.read(content_length)
            data = json.loads(body.decode('utf-8'))
            reasoner.update_settings(data)
            self.send_response(200)
            self.send_header('Content-Type', 'application/json; charset=utf-8')
            self.end_headers()
            self.wfile.write(json.dumps({'status': 'success'}).encode('utf-8'))

        # --- Session Management Endpoints ---
        elif self.path == '/api/sessions/new':
            sid = create_new_session()
            self.send_response(201)
            self.send_header('Content-Type', 'application/json; charset=utf-8')
            self.end_headers()
            self.wfile.write(json.dumps({'status': 'created', 'session_id': sid}).encode('utf-8'))

        elif self.path == '/api/sessions':
            sessions = list_sessions()
            self.send_response(200)
            self.send_header('Content-Type', 'application/json; charset=utf-8')
            self.end_headers()
            self.wfile.write(json.dumps({'sessions': sessions, 'current': current_session_id}).encode('utf-8'))

        elif self.path == '/api/sessions/load':
            content_length = int(self.headers.get('Content-Length', 0))
            body = self.rfile.read(content_length)
            data = json.loads(body.decode('utf-8'))
            sid = data.get('session_id', '')
            session_data = load_session(sid)
            if session_data:
                # Restore reasoner state from session
                current_session_id = sid
                reasoner.session_id = sid
                reasoner.persona = session_data.get('persona', 'ashen_ai_agent')
                reasoner.history = list(session_data.get('history', []))
                ws_ctx = session_data.get('workspace_context', '')
                if ws_ctx:
                    reasoner.set_workspace_context(ws_ctx)
                resp = {'status': 'loaded', 'session': session_data}
            else:
                resp = {'status': 'error', 'message': 'Session not found'}
            self.send_response(200)
            self.send_header('Content-Type', 'application/json; charset=utf-8')
            self.end_headers()
            self.wfile.write(json.dumps(resp).encode('utf-8'))

        elif self.path == '/api/sessions/delete':
            content_length = int(self.headers.get('Content-Length', 0))
            body = self.rfile.read(content_length)
            data = json.loads(body.decode('utf-8'))
            sid = data.get('session_id', '')
            if delete_session(sid):
                if current_session_id == sid:
                    current_session_id = None
                    reasoner.session_id = None
                    reasoner.clear_history()
                    reasoner.set_workspace_context("")
                self.send_response(200)
                self.send_header('Content-Type', 'application/json; charset=utf-8')
                self.end_headers()
                self.wfile.write(json.dumps({'status': 'deleted'}).encode('utf-8'))
            else:
                self.send_response(404)
                self.send_header('Content-Type', 'application/json; charset=utf-8')
                self.end_headers()
                self.wfile.write(json.dumps({'status': 'error', 'message': 'Session not found'}).encode('utf-8'))

        elif self.path == '/api/sessions/rename':
            content_length = int(self.headers.get('Content-Length', 0))
            body = self.rfile.read(content_length)
            data = json.loads(body.decode('utf-8'))
            sid = data.get('session_id', '')
            name = data.get('name', 'Untitled')
            sdata = load_session(sid)
            if sdata:
                sdata['name'] = name
                save_session(sid, sdata)
                self.send_response(200)
                self.send_header('Content-Type', 'application/json; charset=utf-8')
                self.end_headers()
                self.wfile.write(json.dumps({'status': 'renamed'}).encode('utf-8'))
            else:
                self.send_response(404)
                self.send_header('Content-Type', 'application/json; charset=utf-8')
                self.end_headers()
                self.wfile.write(json.dumps({'status': 'error', 'message': 'Session not found'}).encode('utf-8'))

        # --- Settings Persistence Endpoints ---
        elif self.path == '/api/settings/save':
            content_length = int(self.headers.get('Content-Length', 0))
            body = self.rfile.read(content_length)
            data = json.loads(body.decode('utf-8'))
            
            # Save all settings to JSON AND apply to live reasoner + global settings
            success = save_settings_to_json(data)
            if success:
                # merge into global settings dict so future loads see it without restart
                try:
                    settings.update(data)
                except:
                    settings = load_settings_from_json()
                # apply to live model without requiring restart
                try:
                    reasoner.update_settings(data)
                    print(f"[Settings] Applied live: temp={data.get('temperature')} top_k={data.get('top_k')} precision={data.get('precision')} model={data.get('current_model')}", flush=True)
                except Exception as e:
                    print(f"[Settings] Failed to apply live: {e}", flush=True)
            
            if success:
                self.send_response(200)
                self.send_header('Content-Type', 'application/json; charset=utf-8')
                self.end_headers()
                self.wfile.write(json.dumps({'status': 'saved', 'settings': load_settings_from_json()}).encode('utf-8'))
            else:
                self.send_response(500)
                self.send_header('Content-Type', 'application/json; charset=utf-8')
                self.end_headers()
                self.wfile.write(json.dumps({'status': 'error'}).encode('utf-8'))

        elif self.path == '/api/settings/load':
            try:
                loaded = load_settings_from_json()
                self.send_response(200)
                self.send_header('Content-Type', 'application/json; charset=utf-8')
                self.end_headers()
                self.wfile.write(json.dumps({'status': 'success', 'settings': loaded}).encode('utf-8'))
            except Exception as e:
                self.send_response(500)
                self.send_header('Content-Type', 'application/json; charset=utf-8')
                self.end_headers()
                self.wfile.write(json.dumps({'status': 'error', 'message': str(e)}).encode('utf-8'))

        elif self.path == '/api/models/set-default':
            content_length = int(self.headers.get('Content-Length', 0))
            body = self.rfile.read(content_length)
            data = json.loads(body.decode('utf-8'))
            model_path = data.get('path', '')
            
            if set_default_model(model_path):
                # Also save to settings file
                current_settings = load_settings_from_json()
                current_settings['current_model'] = model_path
                save_settings_to_json(current_settings)
                
                self.send_response(200)
                self.send_header('Content-Type', 'application/json; charset=utf-8')
                self.end_headers()
                self.wfile.write(json.dumps({
                    'status': 'set',
                    'model': current_model_filename
                }).encode('utf-8'))
            else:
                self.send_response(404)
                self.send_header('Content-Type', 'application/json; charset=utf-8')
                self.end_headers()
                self.wfile.write(json.dumps({'status': 'error', 'message': 'Model not found'}).encode('utf-8'))

        elif self.path == '/api/models/list':
            models = scan_available_models()
            self.send_response(200)
            self.send_header('Content-Type', 'application/json; charset=utf-8')
            self.end_headers()
            self.wfile.write(json.dumps({
                'models': models,
                'current': current_model_filename
            }).encode('utf-8'))

        elif self.path == '/api/workspace/context':
            # Push current workspace directory listing into model context
            content_length = int(self.headers.get('Content-Length', 0))
            body = self.rfile.read(content_length)
            data = json.loads(body.decode('utf-8'))
            workspace_dir = data.get('dir', '')
            listing = get_directory_listing(workspace_dir)
            # Build a compact text representation of the workspace
            context_lines = [f"Directory: {listing['current_path']}"]
            for item in listing['items']:
                prefix = "📁 " if item['is_dir'] else "📄 "
                context_lines.append(f"  {prefix}{item['name']}")
            reasoner.set_workspace_context("\n".join(context_lines))
            self.send_response(200)
            self.send_header('Content-Type', 'application/json; charset=utf-8')
            self.end_headers()
            self.wfile.write(json.dumps({'status': 'success', 'path': listing['current_path']}).encode('utf-8'))

        elif self.path == '/api/models/scan-pc':
            models = scan_local_pc_models()
            self.send_response(200)
            self.send_header('Content-Type', 'application/json; charset=utf-8')
            self.end_headers()
            self.wfile.write(json.dumps({'status': 'success', 'found': len(models)}).encode('utf-8'))

        elif self.path == '/api/models/download-hf-repo':
            content_length = int(self.headers.get('Content-Length', 0))
            body = self.rfile.read(content_length)
            data = json.loads(body.decode('utf-8'))
            repo_id = data.get('repo_id', '')

            try:
                from huggingface_hub import snapshot_download
                snapshot_download(repo_id=repo_id, local_dir=repo_id.replace('/', '_'))
                resp_data = {'status': 'success', 'repo_id': repo_id}
            except Exception as e:
                resp_data = {'status': 'error', 'message': str(e)}

            self.send_response(200)
            self.send_header('Content-Type', 'application/json; charset=utf-8')
            self.end_headers()
            self.wfile.write(json.dumps(resp_data).encode('utf-8'))

        elif self.path == '/api/models/switch':
            content_length = int(self.headers.get('Content-Length', 0))
            body = self.rfile.read(content_length)
            data = json.loads(body.decode('utf-8'))
            filename = data.get('filename', '')

            # Qwen HF model directory (produced by qwen_finetune.py): load it
            # live via QwenModelAdapter so chat-templated generation + class_head
            # routing work without a server restart.
            if os.path.isdir(filename):
                try:
                    _ch = os.path.join(filename, "class_head.pt")
                    new_model = QwenModelAdapter(filename, device, class_head_path=_ch)
                    reasoner.model = new_model
                    current_model_filename = filename
                    resp_data = {'status': 'success', 'filename': filename, 'type': 'qwen-hf'}
                except Exception as e:
                    resp_data = {'status': 'error', 'message': f'Qwen load failed: {e}'}
            elif filename.endswith('.gguf'):
                # gguf: mark active (served via llama.cpp elsewhere); no in-process reload
                current_model_filename = filename
                resp_data = {'status': 'success', 'filename': filename, 'type': 'gguf'}
            elif os.path.exists(filename):
                try:
                    with open(filename, 'rb') as f:
                        model = pickle.load(f)
                    model = model.to(device)
                    reasoner.model = model
                    current_model_filename = filename
                    resp_data = {'status': 'success', 'filename': filename}
                except Exception as e:
                    resp_data = {'status': 'error', 'message': str(e)}
            else:
                resp_data = {'status': 'error', 'message': 'File not found'}

            self.send_response(200)
            self.send_header('Content-Type', 'application/json; charset=utf-8')
            self.end_headers()
            self.wfile.write(json.dumps(resp_data).encode('utf-8'))

        elif self.path == '/api/models/upload':
            content_length = int(self.headers.get('Content-Length', 0))
            body = self.rfile.read(content_length)
            data = json.loads(body.decode('utf-8'))
            filename = data.get('filename', '')
            content_base64 = data.get('content_base64', '')

            if filename and content_base64:
                try:
                    file_bytes = base64.b64decode(content_base64)
                    safe_name = os.path.basename(filename)
                    with open(safe_name, 'wb') as f:
                        f.write(file_bytes)
                    resp_data = {'status': 'success', 'filename': safe_name}
                except Exception as e:
                    resp_data = {'status': 'error', 'message': str(e)}
            else:
                resp_data = {'status': 'error', 'message': 'Invalid payload'}

            self.send_response(200)
            self.send_header('Content-Type', 'application/json; charset=utf-8')
            self.end_headers()
            self.wfile.write(json.dumps(resp_data).encode('utf-8'))

        elif self.path == '/api/workspace/write':
            content_length = int(self.headers.get('Content-Length', 0))
            body = self.rfile.read(content_length)
            data = json.loads(body.decode('utf-8'))
            file_path = data.get('path', '')
            content = data.get('content', '')

            if file_path:
                try:
                    with open(file_path, 'w', encoding='utf-8') as f:
                        f.write(content)
                    resp_data = {'status': 'success'}
                except Exception as e:
                    resp_data = {'status': 'error', 'message': str(e)}
            else:
                resp_data = {'status': 'error', 'message': 'Invalid file path'}

            self.send_response(200)
            self.send_header('Content-Type', 'application/json; charset=utf-8')
            self.end_headers()
            self.wfile.write(json.dumps(resp_data).encode('utf-8'))

        elif self.path == '/api/feedback':
            content_length = int(self.headers.get('Content-Length', 0))
            body = self.rfile.read(content_length)
            try:
                data = json.loads(body.decode('utf-8'))
                # expected: {rating: 'up'|'down', prompt: str, response: str, thought: str, correction: str, sessionId: str}
                entry = {
                    'rating': data.get('rating', ''),
                    'prompt': (data.get('prompt') or '')[:600],
                    'response': (data.get('response') or '')[:800],
                    'thought': (data.get('thought') or '')[:800],
                    'correction': (data.get('correction') or '')[:1000],
                    'model': data.get('model', os.path.basename(current_model_filename)),
                    'sessionId': data.get('sessionId', current_session_id),
                    'ts': datetime.datetime.now().isoformat()
                }
                _save_feedback_entry(entry)
                # update stats
                delta = {'total_feedback': 1}
                if entry['rating'] == 'up': delta['up'] = 1
                if entry['rating'] == 'down': delta['down'] = 1
                if entry['correction']: delta['corrections'] = 1
                _append_improvement({'type': 'feedback', 'rating': entry['rating'], 'prompt': entry['prompt'][:80], 'stats_delta': delta}, suggestion=None)
                # if correction provided, store it as a learning example in improvement log
                if entry['correction']:
                    _append_improvement({'type': 'correction_learned', 'prompt': entry['prompt'][:80], 'correction': entry['correction'][:300], 'stats_delta': {}}, suggestion=f"Learned correction for \"{entry['prompt'][:60]}\" — next similar prompt should prefer: \"{entry['correction'][:80]}\"")
                # auto-tune hint on repeated downs
                fb_all = _load_feedback()
                recent_downs = [f for f in fb_all[-15:] if f.get('rating') == 'down']
                suggestion = None
                if len(recent_downs) >= 4:
                    suggestion = f"⚠️ {len(recent_downs)}/15 recent feedback are 👎 — consider lowering temperature ({settings.get('temperature')}) or increasing max_new_tokens, or switching to a stronger GGUF (Qwopus Q4/Q5)."
                    _append_improvement({'type': 'auto_tune_hint', 'stats_delta': {'auto_tunes': 0}}, suggestion=suggestion)
                resp_data = {'status': 'success', 'suggestion': suggestion}
            except Exception as e:
                resp_data = {'status': 'error', 'message': str(e)}
            self.send_response(200)
            self.send_header('Content-Type', 'application/json; charset=utf-8')
            self.end_headers()
            self.wfile.write(json.dumps(resp_data).encode('utf-8'))

        elif self.path == '/api/self-improve':
            content_length = int(self.headers.get('Content-Length', 0))
            body = self.rfile.read(content_length) if content_length else b'{}'
            try:
                data = json.loads(body.decode('utf-8')) if body else {}
                action = data.get('action', 'analyze')
                if action == 'regenerate':
                    # regenerate last answer with self-critique: apply correction if provided
                    prompt = data.get('prompt', '')
                    correction = data.get('correction', '')
                    if prompt:
                        # inject correction as part of workspace context / persona hint for one-shot improvement
                        critique_hint = f"\n[SELF-IMPROVEMENT CRITIQUE] Previous answer was rated 👎. User correction: \"{correction}\". You must improve — directly address \"{prompt}\" and incorporate the correction. Your next <think> must reference this critique.\n"
                        orig_ctx = reasoner.workspace_context
                        reasoner.workspace_context = (orig_ctx + critique_hint) if orig_ctx else critique_hint
                        try:
                            thought, response = reasoner.solve_with_agent(prompt)
                            # if correction given, append it as learning signal
                            if correction:
                                response = response + f"\n\n*🔧 Self-improved per your correction: \"{correction[:120]}\"*"
                            _append_improvement({'type': 'regenerate', 'prompt': prompt[:80], 'stats_delta': {'auto_tunes': 1}}, suggestion=f"Regenerated \"{prompt[:60]}\" with critique — improvement applied.")
                            resp_data = {'status': 'success', 'thought': thought, 'response': response, 'model': os.path.basename(current_model_filename), 'sources': getattr(reasoner, 'last_sources', [])}
                        finally:
                            reasoner.workspace_context = orig_ctx
                    else:
                        resp_data = {'status': 'error', 'message': 'No prompt provided'}
                elif action == 'auto-tune':
                    # analyze feedback and propose settings tweak
                    fb = _load_feedback()
                    settings_before = dict(settings)
                    # simple heuristic: if many downs, lower temp by 0.05, raise max_new_tokens
                    recent = fb[-20:]
                    down_ratio = len([f for f in recent if f.get('rating')=='down']) / max(1, len(recent))
                    changes = {}
                    suggestion = None
                    if down_ratio > 0.35:
                        new_temp = max(0.4, round(settings.get('temperature', 0.7) - 0.05, 2))
                        if new_temp != settings.get('temperature'):
                            changes['temperature'] = new_temp
                            settings['temperature'] = new_temp
                            reasoner.update_settings(settings)
                            suggestion = f"Auto-tuned temperature {settings_before.get('temperature')} → {new_temp} (down-ratio {down_ratio:.0%})"
                    else:
                        suggestion = f"No tuning needed — down-ratio {down_ratio:.0%} healthy. Keep temp {settings.get('temperature')}."
                    if changes:
                        save_settings_to_json(settings)
                        _append_improvement({'type': 'auto_tune', 'changes': changes, 'down_ratio': down_ratio, 'stats_delta': {'auto_tunes': 1}}, suggestion=suggestion)
                    resp_data = {'status': 'success', 'changes': changes, 'suggestion': suggestion, 'down_ratio': down_ratio, 'settings': dict(settings)}
                else: # analyze
                    log = _load_improvement_log()
                    fb = _load_feedback()
                    down_ratio = len([f for f in fb[-20:] if f.get('rating')=='down']) / max(1, min(20, len(fb)))
                    # run quick benchmark if requested
                    bench = None
                    if data.get('run_benchmark'):
                        try:
                            bench = reasoner.execute_tool('run_benchmark', {})
                            _append_improvement({'type': 'benchmark', 'stats_delta': {}}, suggestion="Benchmark executed via self-improve")
                        except Exception as e:
                            bench = f"Benchmark failed: {e}"
                    resp_data = {'status': 'success', 'stats': log.get('stats', {}), 'down_ratio': down_ratio, 'suggestions': log.get('suggestions', [])[-5:], 'benchmark': bench}
            except Exception as e:
                import traceback
                resp_data = {'status': 'error', 'message': str(e), 'trace': traceback.format_exc()}
            self.send_response(200)
            self.send_header('Content-Type', 'application/json; charset=utf-8')
            self.end_headers()
            self.wfile.write(json.dumps(resp_data).encode('utf-8'))

        elif self.path == '/api/swarm':
            content_length = int(self.headers.get('Content-Length', 0))
            body = self.rfile.read(content_length) if content_length else b'{}'
            try:
                data = json.loads(body.decode('utf-8')) if body else {}
                task = (data.get('task') or data.get('prompt') or '').strip()
                num_agents = int(data.get('num_agents', data.get('numAgents', 3)))
                mode = data.get('mode', 'parallel')
                if not task:
                    resp_data = {'status': 'error', 'message': 'No task provided (send {task: "...", num_agents: 3, mode: "parallel|divide|debate"})'}
                else:
                    result = _run_swarm(task, num_agents=num_agents, mode=mode)
                    # also append to current session if active
                    if current_session_id:
                        try:
                            # store synthesis as the session reply
                            append_to_session(current_session_id, f"[SWARM {result['num_agents']}x{result['mode']}] {task}", f"<think>\n{result['synthesis']['thought']}\n</think>\n{result['synthesis']['response']}")
                            # also store per-agent summary in session for history
                            for a in result['agents']:
                                append_to_session(current_session_id, f"[Swarm agent {a['id']} {a['role']}] {task[:60]}", f"<think>{a['thought'][:600]}</think>{a['response'][:900]}")
                        except: pass
                    resp_data = {'status': 'success', **result}
            except Exception as e:
                import traceback
                resp_data = {'status': 'error', 'message': str(e), 'trace': traceback.format_exc()}
            self.send_response(200)
            self.send_header('Content-Type', 'application/json; charset=utf-8')
            self.end_headers()
            self.wfile.write(json.dumps(resp_data).encode('utf-8'))

        elif self.path == '/api/council':
            content_length = int(self.headers.get('Content-Length', 0))
            body = self.rfile.read(content_length) if content_length else b'{}'
            try:
                data = json.loads(body.decode('utf-8')) if body else {}
                task = (data.get('task') or data.get('prompt') or '').strip()
                num_drafts = int(data.get('num_drafts', data.get('numDrafts', data.get('num_agents', 3))))
                num_critics = int(data.get('num_critics', data.get('numCritics', 3)))
                if not task:
                    resp_data = {'status': 'error', 'message': 'No task provided (send {task: "...", num_drafts: 3, num_critics: 3})'}
                else:
                    result = _run_council(task, num_drafts=num_drafts, num_critics=num_critics)
                    if current_session_id:
                        try:
                            append_to_session(current_session_id, f"[COUNCIL {result['num_drafts']} drafts + {result['num_critics']} critics] {task}", f"<think>\n{result['final']['thought']}\n</think>\n{result['final']['response']}")
                            # also keep drafts/critics in session for history search
                            for d in result['drafts']:
                                append_to_session(current_session_id, f"[Council draft {d['id']} {d['role']}] {task[:60]}", f"<think>{d['thought'][:600]}</think>{d['response'][:900]}")
                        except: pass
                    resp_data = {'status': 'success', **result}
            except Exception as e:
                import traceback
                resp_data = {'status': 'error', 'message': str(e), 'trace': traceback.format_exc()}
            self.send_response(200)
            self.send_header('Content-Type', 'application/json; charset=utf-8')
            self.end_headers()
            self.wfile.write(json.dumps(resp_data).encode('utf-8'))

        elif self.path == '/api/swarm/stream':
            content_length = int(self.headers.get('Content-Length', 0))
            body = self.rfile.read(content_length) if content_length else b'{}'
            try:
                data = json.loads(body.decode('utf-8')) if body else {}
                task = (data.get('task') or data.get('prompt') or '').strip()
                num_agents = int(data.get('num_agents', data.get('numAgents', 3)))
                mode = data.get('mode', 'parallel')
            except:
                task = ''
                num_agents = 3
                mode = 'parallel'
            if not task:
                self.send_response(400)
                self.send_header('Content-Type', 'application/json; charset=utf-8')
                self.end_headers()
                self.wfile.write(json.dumps({'status':'error','message':'No task'}).encode('utf-8'))
            else:
                self.send_response(200)
                self.send_header('Content-Type', 'application/x-ndjson; charset=utf-8')
                self.send_header('Cache-Control', 'no-cache')
                self.send_header('X-Accel-Buffering', 'no')
                self.end_headers()
                import time as _t
                try:
                    n = max(2, min(6, int(num_agents)))
                    mode = mode if mode in ("parallel","divide","debate") else "parallel"
                    self.wfile.write((json.dumps({"type":"swarm_start","task":task,"num_agents":n,"mode":mode,"model":os.path.basename(current_model_filename)})+"\n").encode('utf-8')); self.wfile.flush()
                    # we reuse _run_swarm but stream its thoughts chunked live
                    # run agents collecting with live thought streaming via solve_with_agent_stream
                    roles = [_SWARM_ROLES[i % len(_SWARM_ROLES)] for i in range(n)]
                    # prepare subtasks like _run_swarm
                    if mode == "divide":
                        base_parts = [t.strip() for t in re.split(r'[.;]\s*|\n+', task) if t.strip()]
                        if len(base_parts) >= n:
                            subtasks = base_parts[:n]
                        else:
                            subtasks = [f"{task}\n\n[Sub-task focus for {roles[i][0]}: {roles[i][1][:80]}]" for i in range(n)]
                    else:
                        subtasks = [task]*n
                    agents = []
                    for idx, (role_name, role_desc) in enumerate(roles):
                        self.wfile.write((json.dumps({"type":"agent_start","id":idx+1,"role":role_name})+"\n").encode('utf-8')); self.wfile.flush()
                        tmp = AshenAIAgenticEngine(model, decode, encode, device, max_steps=3)
                        tmp.persona = reasoner.persona
                        tmp.temperature = reasoner.temperature
                        tmp.top_k = reasoner.top_k
                        tmp.top_p = reasoner.top_p
                        tmp.max_new_tokens = reasoner.max_new_tokens
                        tmp.context_length = reasoner.context_length
                        try: tmp.history = list(reasoner.history[-1:])
                        except: tmp.history = []
                        full_prompt = f"{subtasks[idx]}\n\n[{role_desc}]" if mode!="parallel" else f"{task}\n\n[Role: {role_name} — {role_desc}]"
                        thought_acc = ""
                        response_acc = ""
                        with _swarm_lock:
                            for ev in tmp.solve_with_agent_stream(full_prompt):
                                if ev.get('type')=='thought_delta':
                                    self.wfile.write((json.dumps({"type":"agent_thought_delta","id":idx+1,"role":role_name,"chunk":ev.get('chunk','')})+"\n").encode('utf-8')); self.wfile.flush()
                                    thought_acc += ev.get('chunk','')
                                elif ev.get('type')=='response_delta':
                                    self.wfile.write((json.dumps({"type":"agent_response_delta","id":idx+1,"role":role_name,"chunk":ev.get('chunk','')})+"\n").encode('utf-8')); self.wfile.flush()
                                    response_acc += ev.get('chunk','')
                                elif ev.get('type')=='tool_start':
                                    self.wfile.write((json.dumps({"type":"agent_tool_start","id":idx+1,"role":role_name,"tool":ev.get('tool')})+"\n").encode('utf-8')); self.wfile.flush()
                                elif ev.get('type')=='done':
                                    thought_acc = ev.get('thought', thought_acc)
                                    response_acc = ev.get('response', response_acc)
                        if not thought_acc and not response_acc:
                            # fallback to non-stream run if stream yielded nothing (should not happen)
                            t2, r2 = tmp.solve_with_agent(full_prompt)
                            thought_acc, response_acc = t2, r2
                        agents.append({'id': idx+1, 'role': role_name, 'thought': thought_acc, 'response': response_acc, 'model': os.path.basename(current_model_filename)})
                        self.wfile.write((json.dumps({"type":"agent_done","id":idx+1,"role":role_name,"thought":thought_acc,"response":response_acc})+"\n").encode('utf-8')); self.wfile.flush()
                        if mode=="debate" and idx>0:
                            _t.sleep(0.05)
                    # synthesis streaming
                    self.wfile.write((json.dumps({"type":"synthesis_start"})+"\n").encode('utf-8')); self.wfile.flush()
                    drafts = "\n\n".join([f"--- Agent {a['id']} ({a['role']}) ---\n{a['response'][:800]}" for a in agents])
                    synth_prompt = f"You are the Swarm Synthesizer. The user task is: \"{task}\"\n\nSubagent drafts:\n{drafts}\n\nSynthesize into ONE superior answer. Output <think>your synthesis reasoning (mention task, which draft was best and why)</think> then the final answer. Be concise and directly address the original task."
                    synth_thought_acc=""; synth_resp_acc=""
                    with _swarm_lock:
                        tmp_s = AshenAIAgenticEngine(model, decode, encode, device, max_steps=2)
                        tmp_s.history = list(reasoner.history[-1:])
                        tmp_s.temperature = reasoner.temperature
                        for ev in tmp_s.solve_with_agent_stream(synth_prompt):
                            if ev.get('type')=='thought_delta':
                                self.wfile.write((json.dumps({"type":"synthesis_thought_delta","chunk":ev.get('chunk','')})+"\n").encode('utf-8')); self.wfile.flush()
                                synth_thought_acc += ev.get('chunk','')
                            elif ev.get('type')=='response_delta':
                                self.wfile.write((json.dumps({"type":"synthesis_response_delta","chunk":ev.get('chunk','')})+"\n").encode('utf-8')); self.wfile.flush()
                                synth_resp_acc += ev.get('chunk','')
                            elif ev.get('type')=='done':
                                synth_thought_acc = ev.get('thought', synth_thought_acc)
                                synth_resp_acc = ev.get('response', synth_resp_acc)
                    if not synth_resp_acc.strip():
                        best = max(agents, key=lambda a: len(a['response']))
                        if best['response'].strip():
                            synth_resp_acc = f"**Swarm synthesis (fallback to {best['role']}):**\n\n{best['response']}"
                    result = {'task':task,'mode':mode,'num_agents':n,'agents':agents,'synthesis':{'thought':synth_thought_acc,'response':synth_resp_acc},'elapsed_s':0,'model':os.path.basename(current_model_filename),'model_path':current_model_filename}
                    try:
                        _append_improvement({'type': 'swarm', 'prompt': task[:80], 'mode': mode, 'agents': n, 'elapsed': 0, 'stats_delta': {}}, suggestion=f"Swarm {n}x{mode} streamed")
                    except: pass
                    if current_session_id:
                        try: append_to_session(current_session_id, f"[SWARM {n}x{mode}] {task}", f"<think>\n{synth_thought_acc}\n</think>\n{synth_resp_acc}")
                        except: pass
                    self.wfile.write((json.dumps({"type":"done", **result, "status":"success"})+"\n").encode('utf-8')); self.wfile.flush()
                except Exception as e:
                    import traceback
                    self.wfile.write((json.dumps({"type":"error","message":str(e),"trace":traceback.format_exc()})+"\n").encode('utf-8')); self.wfile.flush()

        elif self.path == '/api/council/stream':
            content_length = int(self.headers.get('Content-Length', 0))
            body = self.rfile.read(content_length) if content_length else b'{}'
            try:
                data = json.loads(body.decode('utf-8')) if body else {}
                task = (data.get('task') or data.get('prompt') or '').strip()
                num_drafts = int(data.get('num_drafts', data.get('numDrafts', data.get('num_agents', 3))))
                num_critics = int(data.get('num_critics', data.get('numCritics', 3)))
            except:
                task=''; num_drafts=3; num_critics=3
            if not task:
                self.send_response(400)
                self.send_header('Content-Type', 'application/json; charset=utf-8')
                self.end_headers()
                self.wfile.write(json.dumps({'status':'error','message':'No task'}).encode('utf-8'))
            else:
                self.send_response(200)
                self.send_header('Content-Type', 'application/x-ndjson; charset=utf-8')
                self.send_header('Cache-Control', 'no-cache')
                self.send_header('X-Accel-Buffering', 'no')
                self.end_headers()
                try:
                    nd = max(2, min(5, int(num_drafts)))
                    nc = max(2, min(5, int(num_critics)))
                    self.wfile.write((json.dumps({"type":"council_start","task":task,"num_drafts":nd,"num_critics":nc,"model":os.path.basename(current_model_filename)})+"\n").encode('utf-8')); self.wfile.flush()
                    # drafts phase with streaming
                    proposer_roles = [_SWARM_ROLES[i % len(_SWARM_ROLES)] for i in range(nd)]
                    drafts=[]
                    for idx,(role_name, role_desc) in enumerate(proposer_roles):
                        self.wfile.write((json.dumps({"type":"draft_start","id":idx+1,"role":role_name})+"\n").encode('utf-8')); self.wfile.flush()
                        tmp = AshenAIAgenticEngine(model, decode, encode, device, max_steps=3)
                        tmp.persona=reasoner.persona; tmp.temperature=reasoner.temperature; tmp.top_k=reasoner.top_k; tmp.top_p=reasoner.top_p; tmp.max_new_tokens=reasoner.max_new_tokens; tmp.context_length=reasoner.context_length
                        try: tmp.history=list(reasoner.history[-1:])
                        except: tmp.history=[]
                        full_prompt=f"{task}\n\n[Proposer role: {role_name} — {role_desc}]"
                        th=""; rp=""
                        with _swarm_lock:
                            for ev in tmp.solve_with_agent_stream(full_prompt):
                                if ev.get('type')=='thought_delta':
                                    self.wfile.write((json.dumps({"type":"draft_thought_delta","id":idx+1,"role":role_name,"chunk":ev.get('chunk','')})+"\n").encode('utf-8')); self.wfile.flush()
                                    th+=ev.get('chunk','')
                                elif ev.get('type')=='response_delta':
                                    self.wfile.write((json.dumps({"type":"draft_response_delta","id":idx+1,"role":role_name,"chunk":ev.get('chunk','')})+"\n").encode('utf-8')); self.wfile.flush()
                                    rp+=ev.get('chunk','')
                                elif ev.get('type')=='done':
                                    th=ev.get('thought',th); rp=ev.get('response',rp)
                        drafts.append({'id':idx+1,'role':role_name,'thought':th,'response':rp,'model':os.path.basename(current_model_filename)})
                        self.wfile.write((json.dumps({"type":"draft_done","id":idx+1,"role":role_name,"thought":th,"response":rp})+"\n").encode('utf-8')); self.wfile.flush()
                    # critics phase
                    critic_roles=[_COUNCIL_CRITIC_ROLES[i%len(_COUNCIL_CRITIC_ROLES)] for i in range(nc)]
                    drafts_block="\n\n".join([f"[Draft {d['id']} ({d['role']}):]\n{d['response'][:900]}" for d in drafts])
                    def _heuristic_score(text, prompt):
                        if not text.strip(): return 3
                        p_words=set(re.findall(r'[a-z]{3,}', prompt.lower())); t_words=set(re.findall(r'[a-z]{3,}', text.lower()))
                        overlap=len(p_words & t_words)/max(1,len(p_words)); score=5+int(overlap*3)+min(2,len(text)//350); return max(1,min(10,score))
                    critics=[]
                    for c_idx,(critic_name, critic_desc) in enumerate(critic_roles):
                        self.wfile.write((json.dumps({"type":"critic_start","id":c_idx+1,"role":critic_name})+"\n").encode('utf-8')); self.wfile.flush()
                        tmp = AshenAIAgenticEngine(model, decode, encode, device, max_steps=2)
                        tmp.persona=reasoner.persona; tmp.temperature=0.65; tmp.top_k=reasoner.top_k; tmp.top_p=reasoner.top_p; tmp.max_new_tokens=220; tmp.context_length=reasoner.context_length
                        try: tmp.history=list(reasoner.history[-1:])
                        except: tmp.history=[]
                        critic_prompt=(f"Task: \"{task}\"\n\nDrafts to evaluate:\n{drafts_block}\n\n[{critic_desc}]\nYou are {critic_name}. For EACH draft, output exactly:\nDraft <id>: Score <1-10> - Suggestion: <one sentence change>\nThen on last line: VOTE: <id> (your top pick)\nBe relevant to the task and concise.")
                        th_c=""; rp_c=""
                        with _swarm_lock:
                            for ev in tmp.solve_with_agent_stream(critic_prompt):
                                if ev.get('type')=='thought_delta':
                                    self.wfile.write((json.dumps({"type":"critic_thought_delta","id":c_idx+1,"role":critic_name,"chunk":ev.get('chunk','')})+"\n").encode('utf-8')); self.wfile.flush()
                                    th_c+=ev.get('chunk','')
                                elif ev.get('type')=='response_delta':
                                    self.wfile.write((json.dumps({"type":"critic_response_delta","id":c_idx+1,"role":critic_name,"chunk":ev.get('chunk','')})+"\n").encode('utf-8')); self.wfile.flush()
                                    rp_c+=ev.get('chunk','')
                                elif ev.get('type')=='done':
                                    th_c=ev.get('thought',th_c); rp_c=ev.get('response',rp_c)
                        votes={}; suggestions={}; vote_pick=None
                        for line in rp_c.splitlines():
                            m=re.search(r'Draft\s+(\d+)\s*:\s*Score\s*(\d+)', line, re.I)
                            if m:
                                did=int(m.group(1)); sc=max(1,min(10,int(m.group(2)))); votes[did]=sc
                                seg=re.split(r'Suggestion\s*:\s*', line, flags=re.I)
                                if len(seg)>1: suggestions[did]=seg[1].strip()[:200]
                                elif '-' in line: suggestions[did]=line.split('-',1)[1].strip()[:200]
                            mv=re.search(r'VOTE\s*:\s*(\d+)', line, re.I)
                            if mv: vote_pick=int(mv.group(1))
                        if not votes:
                            for d in drafts:
                                votes[d['id']]=_heuristic_score(d['response'], task)
                                suggestions[d['id']]=f"[{critic_name} heuristic] Improve relevance to \"{task[:40]}\""
                            vote_pick=max(votes, key=lambda k: votes[k])
                        if vote_pick is None:
                            vote_pick=max(votes, key=lambda k: votes[k]) if votes else drafts[0]['id']
                        critics.append({'id':c_idx+1,'role':critic_name,'desc':critic_desc,'thought':th_c,'response':rp_c,'votes':votes,'suggestions':suggestions,'pick':vote_pick})
                        self.wfile.write((json.dumps({"type":"critic_done","id":c_idx+1,"role":critic_name,"votes":votes,"suggestions":suggestions,"pick":vote_pick})+"\n").encode('utf-8')); self.wfile.flush()
                    # tally + final revise streaming
                    tally={d['id']:0 for d in drafts}; score_sum={d['id']:0 for d in drafts}
                    for c in critics:
                        for did,sc in c['votes'].items():
                            if did in score_sum: score_sum[did]+=sc
                        if c['pick'] in tally: tally[c['pick']]+=1
                    sorted_drafts=sorted(drafts, key=lambda d: (tally.get(d['id'],0), score_sum.get(d['id'],0), len(d['response'])), reverse=True)
                    winner=sorted_drafts[0]
                    winner_suggestions=[]
                    for c in critics:
                        if c['suggestions'].get(winner['id']): winner_suggestions.append(f"- [{c['role']}] {c['suggestions'][winner['id']]}")
                    if not winner_suggestions: winner_suggestions=[f"- [{c['role']}] {list(c['suggestions'].values())[0] if c['suggestions'] else 'No suggestion'}" for c in critics[:2]]
                    suggestions_block="\n".join(winner_suggestions[:5])
                    self.wfile.write((json.dumps({"type":"tally","tally":tally,"score_sum":score_sum,"winner":{"id":winner['id'],"role":winner['role']},"suggestions":winner_suggestions})+"\n").encode('utf-8')); self.wfile.flush()
                    self.wfile.write((json.dumps({"type":"final_start","winner":winner['id']})+"\n").encode('utf-8')); self.wfile.flush()
                    revise_prompt=(f"Task: \"{task}\"\n\nWinning draft (Draft {winner['id']} by {winner['role']}):\n{winner['response'][:1400]}\n\nCouncil critiques & required changes:\n{suggestions_block}\n\nOther drafts for reference:\n" + "\n".join([f"Draft {d['id']} ({d['role']}): {d['response'][:500]}" for d in drafts if d['id']!=winner['id']][:2]) + "\n\nYou are the Council Finalizer. Apply the critiques, keep the chain-of-thought relevant to the original task, and output the improved FINAL answer. Output <think>your revision reasoning (mention task, winner, and which critique was most impactful)</think> then the final response.")
                    final_th=""; final_rp=""
                    with _swarm_lock:
                        tmp_final=AshenAIAgenticEngine(model, decode, encode, device, max_steps=2)
                        tmp_final.history=list(reasoner.history[-1:])
                        tmp_final.temperature=reasoner.temperature
                        for ev in tmp_final.solve_with_agent_stream(revise_prompt):
                            if ev.get('type')=='thought_delta':
                                self.wfile.write((json.dumps({"type":"final_thought_delta","chunk":ev.get('chunk','')})+"\n").encode('utf-8')); self.wfile.flush()
                                final_th+=ev.get('chunk','')
                            elif ev.get('type')=='response_delta':
                                self.wfile.write((json.dumps({"type":"final_response_delta","chunk":ev.get('chunk','')})+"\n").encode('utf-8')); self.wfile.flush()
                                final_rp+=ev.get('chunk','')
                            elif ev.get('type')=='done':
                                final_th=ev.get('thought',final_th); final_rp=ev.get('response',final_rp)
                    if not final_rp.strip():
                        final_rp=winner['response']+"\n\n**Council refinements applied:**\n"+suggestions_block
                        final_th=winner['thought']+"\n\n[Council synthesis fallback — winner "+winner['role']+" + critiques]"
                    result={'task':task,'num_drafts':nd,'num_critics':nc,'drafts':drafts,'critics':critics,'tally':tally,'score_sum':score_sum,'winner':winner,'suggestions':winner_suggestions,'final':{'thought':final_th,'response':final_rp},'elapsed_s':0,'model':os.path.basename(current_model_filename),'model_path':current_model_filename}
                    try: _append_improvement({'type': 'council', 'prompt': task[:80], 'drafts': nd, 'critics': nc, 'winner': winner['id'], 'elapsed': 0, 'stats_delta': {}}, suggestion=f"Council {nd} drafts + {nc} critics streamed")
                    except: pass
                    if current_session_id:
                        try: append_to_session(current_session_id, f"[COUNCIL {nd} drafts + {nc} critics] {task}", f"<think>\n{final_th}\n</think>\n{final_rp}")
                        except: pass
                    self.wfile.write((json.dumps({"type":"done", **result, "status":"success"})+"\n").encode('utf-8')); self.wfile.flush()
                except Exception as e:
                    import traceback
                    self.wfile.write((json.dumps({"type":"error","message":str(e),"trace":traceback.format_exc()})+"\n").encode('utf-8')); self.wfile.flush()

        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, format, *args):
        # Show why a request failed instead of silent 404
        sys.stderr.write(f"[HTTP] {self.client_address[0]} - - [{self.log_date_time_string()}] {format%args}\n")

def run_server(port=5000, host='localhost'):
    # Kill stale instance on same port if present (common when .bat is double-clicked twice)
    try:
        # allow rapid restart
        socketserver.ThreadingTCPServer.allow_reuse_address = True
        # Threaded: a long-lived /api/chat/stream holds its connection open, so a
        # single-threaded server would block every later prompt until the first
        # stream finishes. Threading lets the 2nd (and Nth) prompt stream too.
        server = socketserver.ThreadingTCPServer((host, port), ChatHandler)
    except OSError as e:
        print(f"[ERROR] Could not bind to {host}:{port} - {e}", flush=True)
        print(f"[HINT] That port is already in use. Either close the old window or run:", flush=True)
        print(f"       python web_chatbot.py --port {port+1}  (then open http://{host}:{port+1})", flush=True)
        print(f"       On Windows check: netstat -ano | findstr :{port}", flush=True)
        sys.exit(1)
    print(f"\n========================================================")
    print(f" Ashen AI Cybernetic Hub running at: http://{host}:{port}")
    print(f" Settings file: {SETTINGS_FILE}")
    print(f" If Chrome says 'site can't be reached', make sure you use http:// not https://", flush=True)
    print(f" Open your browser to experience the Ashen AI interface!")
    print(f"========================================================\n")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down Ashen AI server...")
        server.server_close()

if __name__ == '__main__':
    # Use CLI args parsed earlier (_cli_args is already available)
    try:
        run_server(port=_cli_args.port, host=_cli_args.host)
    except NameError:
        run_server(5000)
