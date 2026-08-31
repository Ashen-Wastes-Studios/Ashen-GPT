# -*- coding: utf-8 -*-
"""
chatbot.py — Ashen GPT agentic CLI chatbot.

Self-contained port of the web_chatbot.py backend (NO import dependency on
web_chatbot.py). It folds in: the legacy custom `.pk1` MoE model, the Qwen3.5
HF `ashen_gpt_model/` backend (via QwenModelAdapter), the agentic reasoning
engine with streaming chain-of-thought, the auxiliary intent-classification
head (spam / not_spam / question / answer / request), Swarm + Council
multi-agent deliberation, autonomous deep web research with source harvesting,
settings.json persistence, a model hub scan, and a self-improvement /
feedback log. The browser/HTML/HTTP layer of web_chatbot.py is intentionally
omitted — this is a terminal CLI.

Run:  cuda\\Scripts\\python.exe chatbot.py
"""

import sys
import io
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

import os
import re
import json
import time
import random
import glob as glob_module
import subprocess
import shutil
import datetime
import argparse
import pickle
import base64
import threading

import torch
import torch.nn as nn
from torch.nn import functional as F
try:
    import tiktoken
except Exception:  # legacy custom model uses tiktoken gpt2 tokenizer
    tiktoken = None
try:
    import requests
except Exception:
    requests = None

HERE = os.path.dirname(os.path.abspath(__file__))

# =====================================================================
#  Paths & session storage
# =====================================================================
SESSIONS_DIR = os.path.join(HERE, 'sessions_cli')
os.makedirs(SESSIONS_DIR, exist_ok=True)
current_session_id = None

# =====================================================================
#  Self-Improvement / feedback storage
# =====================================================================
FEEDBACK_FILE = os.path.join(HERE, 'feedback.json')
IMPROVEMENT_LOG_FILE = os.path.join(HERE, 'self_improvement.json')
_self_improvement_stats = {'total_feedback': 0, 'up': 0, 'down': 0,
                           'corrections': 0, 'gibberish_fixes': 0, 'auto_tunes': 0}


def _load_feedback():
    try:
        if os.path.exists(FEEDBACK_FILE):
            with open(FEEDBACK_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
    except Exception:
        pass
    return []


def _save_feedback_entry(entry):
    data = _load_feedback()
    data.append(entry)
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
    except Exception:
        pass
    return {'entries': [], 'stats': dict(_self_improvement_stats), 'suggestions': []}


def _append_improvement(entry, suggestion=None):
    log = _load_improvement_log()
    entry['ts'] = datetime.datetime.now().isoformat()
    log['entries'].append(entry)
    if suggestion:
        log['suggestions'].append({'ts': entry['ts'], 'text': suggestion})
        log['suggestions'] = log['suggestions'][-50:]
    if len(log['entries']) > 300:
        log['entries'] = log['entries'][-300:]
    for k in _self_improvement_stats:
        if k in entry.get('stats_delta', {}):
            _self_improvement_stats[k] += entry['stats_delta'][k]
    log['stats'] = dict(_self_improvement_stats)
    try:
        with open(IMPROVEMENT_LOG_FILE, 'w', encoding='utf-8') as f:
            json.dump(log, f, indent=2)
    except Exception as e:
        print(f"[SelfImprove] save failed: {e}", flush=True)
    return log


try:
    _existing_log = _load_improvement_log()
    if _existing_log.get('stats'):
        _self_improvement_stats.update(_existing_log['stats'])
except Exception:
    pass

# =====================================================================
#  Settings persistence (settings.json, mirrors web_chatbot.py)
# =====================================================================
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
    "current_model": "ashen_gpt_model.pk1",
}


def _resolve_settings_file():
    default_path = os.path.join(HERE, 'settings.json')
    env_path = os.getenv('SETTINGS_PATH') or os.getenv('ASHEN_SETTINGS')
    if env_path:
        return os.path.abspath(env_path)
    for i, arg in enumerate(sys.argv):
        if arg == '--settings' and i + 1 < len(sys.argv):
            return os.path.abspath(sys.argv[i + 1])
        if arg.startswith('--settings='):
            return os.path.abspath(arg.split('=', 1)[1])
    return default_path


SETTINGS_FILE = _resolve_settings_file()


def load_settings_from_json(path=None):
    target = os.path.abspath(path) if path else SETTINGS_FILE
    if not os.path.exists(target):
        try:
            os.makedirs(os.path.dirname(target) or '.', exist_ok=True)
            with open(target, 'w', encoding='utf-8') as f:
                json.dump(dict(DEFAULT_SETTINGS), f, indent=2)
        except Exception:
            pass
        return dict(DEFAULT_SETTINGS)
    try:
        with open(target, 'r', encoding='utf-8') as f:
            data = json.load(f)
        if not isinstance(data, dict):
            data = {}
    except Exception:
        data = {}
    base = dict(DEFAULT_SETTINGS)
    base.update(data)
    return base


def save_settings_to_json(settings_data, path=None):
    target = os.path.abspath(path) if path else SETTINGS_FILE
    try:
        if os.path.exists(target):
            try:
                with open(target, 'r', encoding='utf-8') as f:
                    existing = json.load(f)
                if not isinstance(existing, dict):
                    existing = {}
            except Exception:
                existing = {}
            base = dict(DEFAULT_SETTINGS)
            base.update(existing)
        else:
            base = dict(DEFAULT_SETTINGS)
        if isinstance(settings_data, dict):
            base.update(settings_data)
        os.makedirs(os.path.dirname(target) or '.', exist_ok=True)
        with open(target, 'w', encoding='utf-8') as f:
            json.dump(base, f, indent=2)
        return True
    except Exception as e:
        print(f"[Settings] Failed to save {target}: {e}", flush=True)
        return False


def parse_cli_args():
    p = argparse.ArgumentParser(description="Ashen AI CLI Chatbot", add_help=False)
    p.add_argument('--settings', type=str, default=SETTINGS_FILE,
                   help='Path to settings.json')
    p.add_argument('--host', type=str, default='localhost', help='(unused in CLI)')
    p.add_argument('--port', type=int, default=0, help='(unused in CLI)')
    p.add_argument('-h', '--help', action='store_true', help='Show help')
    args, _ = p.parse_known_args()
    if args.help:
        p.print_help()
        sys.exit(0)
    return args


# =====================================================================
#  Device & tokenizer
# =====================================================================
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Device: {device}")

# Default tokenizer for the legacy custom model. The Qwen adapter uses its own
# HF tokenizer. We set enc/dec/special tokens here; for Qwen these are only used
# by the custom-model path.
if tiktoken is not None:
    enc = tiktoken.get_encoding("gpt2")
    vocab_size = enc.n_vocab
    encode = lambda s: enc.encode(s, allowed_special={"<|endoftext|>"})
    decode = lambda l: enc.decode(l)
else:
    enc = None
    vocab_size = 50257
    encode = lambda s: [0]
    decode = lambda l: ""

# Legacy custom model hyper-params (used only by AshenGPTLanguageModel)
block_size = 8192
n_embd = 512
n_layer = 8
n_head = 8
dropout = 0.1
num_experts = 4
top_k = 2

DEFAULT_MODEL_FILENAME = 'ashen_gpt_model.pk1'


# =====================================================================
#  Model architecture (legacy custom .pk1 MoE model)
# =====================================================================
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
        loss = None if targets is None else F.cross_entropy(logits.view(B * T, -1), targets.view(B * T))
        cls_logits = self.class_head(x.mean(dim=1))
        cls_loss = None
        if cls_targets is not None:
            cls_loss = F.cross_entropy(cls_logits, cls_targets)
        return logits, loss, cls_logits, cls_loss

    @torch.no_grad()
    def classify(self, text, current_block_size=8192):
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
            tok_id = int(index_next[0, 0].item()) if index_next.dim() == 2 else int(index_next.item())
            yield index, tok_id


# =====================================================================
#  Qwen3.5 HF model adapter
# =====================================================================
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
        self.kv_cap = int(os.environ.get("QWEN_KV_CAP", "2048"))
        self.gen_cap = int(os.environ.get("QWEN_GEN_CAP", "512"))
        self.class_head = None
        if class_head_path and os.path.exists(class_head_path):
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
            msgs, tokenize=False, add_generation_prompt=add_generation_prompt)
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
        self.model.eval()
        return self

    def train(self, mode=True):
        self.model.train(mode)
        return self


# =====================================================================
#  Safe checkpoint loader (legacy .pk1 remapped to this module)
# =====================================================================
class _AshenUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == 'ashen_gpt_trainer':
            return getattr(sys.modules[__name__], name)
        return super().find_class(module, name)


def _load_ashen_checkpoint(path):
    with open(path, 'rb') as f:
        return _AshenUnpickler(f).load()


# =====================================================================
#  Model selection / load (mirrors web_chatbot.py backend)
# =====================================================================
settings = load_settings_from_json(SETTINGS_FILE)
saved_model_path = settings.get('current_model', DEFAULT_MODEL_FILENAME)
current_model_filename = saved_model_path if os.path.exists(saved_model_path) else DEFAULT_MODEL_FILENAME
if os.path.exists(current_model_filename):
    print(f"[Model] Loading saved model: {current_model_filename}")
else:
    print(f"[Model] Saved model not found, using default: {current_model_filename}")

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
    except Exception as e:
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

# Draft model (speculative decoding) — optional
draft_model_filename = 'ashen_gpt_model_draft.pk1'
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
    """Decode a DuckDuckGo redirect href (/l/?uddg=BASE64URL) to the real URL."""
    if not href:
        return href
    mm = re.search(r'uddg=([^&]+)', href)
    if mm:
        try:
            enc = mm.group(1)
            enc = enc.replace('-', '+').replace('_', '/')
            enc += '=' * (-len(enc) % 4)
            return base64.b64decode(enc).decode('utf-8', errors='ignore')
        except Exception:
            return href
    return href


# =====================================================================
#  Agentic reasoning engine
# =====================================================================
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
        self.workspace_context = ""
        self.session_id = None
        self.last_intent = None
        self.pending_requests = []
        self.use_draft_model = False
        self.draft_temperature = 0.6
        self.low_end_gpu_mode = False
        self.precision = 'fp32'
        self.cpu_offload_layers = 0
        self.auto_swarm_council = False
        self.auto_web_research = False
        self._source_harvest = []

    def clear_history(self):
        self.history = []

    def set_persona(self, persona):
        self.persona = persona

    def set_workspace_context(self, workspace_info):
        self.workspace_context = workspace_info

    def update_settings(self, s):
        self.temperature = float(s.get('temperature', self.temperature))
        self.top_k = int(s.get('top_k', self.top_k))
        self.top_p = float(s.get('top_p', self.top_p))
        self.max_new_tokens = int(s.get('max_new_tokens', self.max_new_tokens))
        self.context_length = int(s.get('context_length', self.context_length))
        self.gpu_layers = int(s.get('gpu_layers', self.gpu_layers))
        self.repeat_penalty = float(s.get('repeat_penalty', self.repeat_penalty))
        if 'use_draft_model' in s:
            self.use_draft_model = bool(s['use_draft_model'])
        if 'draft_temperature' in s:
            self.draft_temperature = float(s['draft_temperature'])
        if 'auto_swarm_council' in s:
            self.auto_swarm_council = bool(s['auto_swarm_council'])
        if 'auto_web_research' in s:
            self.auto_web_research = bool(s['auto_web_research'])

    @torch.no_grad()
    def generate_with_speculative_decoding(self, input_ids, max_new_tokens):
        global draft_model
        if not draft_model or not self.use_draft_model:
            return self.model.generate(input_ids, max_new_tokens=max_new_tokens,
                                       current_block_size=self.context_length,
                                       temperature=self.temperature, top_k=self.top_k)
        draft_steps = 8
        accept_threshold = 0.5
        generated = input_ids.clone()
        for step in range(max_new_tokens // draft_steps + 1):
            draft_output = draft_model.generate(generated, max_new_tokens=draft_steps,
                                                 current_block_size=self.context_length,
                                                 temperature=self.draft_temperature, top_k=30)
            draft_proposals = draft_output[0, len(input_ids[0]):].tolist()
            accepted_count = 0
            final_tokens = []
            for draft_token in draft_proposals:
                draft_tensor = torch.tensor([[draft_token]], dtype=torch.long, device=device)
                with torch.autocast('cuda' if device == 'cuda' else 'cpu'):
                    logits = self.model.forward(draft_tensor, current_block_size=self.context_length)[0]
                    probs = F.softmax(logits[0, -1], dim=-1)
                    accept_prob = probs[draft_token].item()
                if random.random() < min(accept_prob / accept_threshold, 1.0):
                    final_tokens.append(draft_token)
                    accepted_count += 1
                    generated = torch.cat([generated, torch.tensor([[draft_token]])], dim=1)
                    if len(generated[0]) >= self.context_length:
                        break
                else:
                    main_logits = self.model.forward(generated, current_block_size=self.context_length)[0]
                    adjusted_probs = F.log_softmax(main_logits[0, -1], dim=-1).exp()
                    adjusted_probs = adjusted_probs * (1.0 - accept_threshold) + (probs * accept_threshold)
                    adjusted_probs = adjusted_probs / adjusted_probs.sum()
                    next_token = torch.multinomial(adjusted_probs, 1).item()
                    final_tokens.append(next_token)
                    generated = torch.cat([generated, torch.tensor([[next_token]])], dim=1)
                    break
            if not final_tokens:
                break
            if len(final_tokens) >= max_new_tokens:
                break
        print(f"[Speculative Decoding] Accepted {accepted_count}/{len(draft_proposals)} tokens",
              flush=True)
        return generated

    def execute_tool(self, tool_name, kwargs):
        try:
            self._source_harvest = getattr(self, '_source_harvest', [])
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
                            except Exception:
                                pass
                return "\n".join(results) if results else "No matches found."

            elif tool_name == 'run_shell_command':
                cmd = kwargs.get('command', '')
                res = subprocess.run(cmd, shell=True, capture_output=True, text=True,
                                      timeout=30, cwd=os.getcwd())
                output = res.stdout if res.returncode == 0 else res.stderr
                return output[:2000] if output else "Command executed with no output."

            elif tool_name == 'web_search':
                query = kwargs.get('query', '')
                if requests is None:
                    return "Web search unavailable (requests not installed)."
                url = f"https://html.duckduckgo.com/html/?q={requests.utils.quote(query)}"
                headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
                try:
                    resp = requests.get(url, headers=headers, timeout=10)
                    if resp.status_code == 200:
                        results = []
                        for mm in re.finditer(
                                r'<a[^>]*class="result__a"[^>]*href="([^"]+)"[^>]*>(.*?)</a>',
                                resp.text, re.DOTALL):
                            real = _ddg_real_url(mm.group(1))
                            title = re.sub(r'<[^>]+>', '', mm.group(2)).strip()
                            if title and real:
                                results.append((title, real))
                        if results:
                            lines = [f"DuckDuckGo results for '{query}':"]
                            harvest = getattr(self, '_source_harvest', [])
                            for i, (title, real) in enumerate(results[:6], 1):
                                lines.append(f"{i}. {title}\n   {real}")
                                harvest.append({"title": title, "url": real})
                            lines.append(f"\nFound {len(results)} results. Use browse_url(url='...') "
                                         "to fetch the full content of a specific page.")
                            return "\n".join(lines)
                        return f"No results found for '{query}'"
                    return f"Search failed with status code {resp.status_code}"
                except Exception as e:
                    return f"Web search error: {str(e)}"

            elif tool_name == 'browse_url':
                url = kwargs.get('url', '')
                if requests is None:
                    return "Browse unavailable (requests not installed)."
                headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
                try:
                    resp = requests.get(url, headers=headers, timeout=15)
                    if resp.status_code == 200:
                        getattr(self, '_source_harvest', []).append({"title": url, "url": url})
                        content = re.sub(r'<[^>]+>', ' ', resp.text)
                        content = re.sub(r'\s+', ' ', content).strip()
                        return f"Content from {url}:\n\n{content[:3000]}"
                    return f"Failed to fetch URL: Status code {resp.status_code}"
                except Exception as e:
                    return f"Browse error: {str(e)}"

            elif tool_name == 'deep_research':
                topic = kwargs.get('topic', '')
                max_searches = int(kwargs.get('max_searches', '3'))
                if requests is None:
                    return "Deep research unavailable (requests not installed)."
                try:
                    research_report = f"# Deep Research Report: {topic}\n\n"
                    harvest = getattr(self, '_source_harvest', [])
                    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}

                    def _ddg_search(q):
                        u = f"https://html.duckduckgo.com/html/?q={requests.utils.quote(q)}"
                        r = requests.get(u, headers=headers, timeout=10)
                        out = []
                        if r.status_code == 200:
                            for mm in re.finditer(r'<a[^>]*class="result__a"[^>]*href="(.*?)"[^>]*>(.*?)</a>',
                                                  r.text, re.DOTALL):
                                link = _ddg_real_url(mm.group(1))
                                title = re.sub(r'<[^>]+>', '', mm.group(2)).strip()
                                if link and not any(sk in link.lower() for sk in
                                                    ['duckduckgo.com', 'facebook.com', 'twitter.com', 'x.com']):
                                    out.append((link, title))
                        return out

                    search_links = _ddg_search(topic)
                    if not search_links:
                        return f"Research failed: no results for '{topic}'"
                    research_report += f"## Initial Search Overview\n\nTopic: `{topic}`\nFound {len(search_links)} relevant results.\n\n"
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
                    for next_topic in [f"{topic} latest developments", f"{topic} analysis", f"{topic} examples"]:
                        if browsed >= max_searches + 2:
                            break
                        for link_url, title_text in _ddg_search(next_topic)[:2]:
                            research_report += f"\n**Related: {next_topic}**\n- {title_text} — {link_url}\n"
                            harvest.append({"title": title_text, "url": link_url})
                            browsed += 1
                    sources_block = "\n## Sources\n" + "\n".join(
                        f"{i+1}. {s['title']} — {s['url']}" for i, s in enumerate(harvest))
                    return (f"Deep Research Complete!\n\n" + research_report + sources_block
                            + "\n---\nReport generated autonomously via web traversal.")
                except Exception as e:
                    return f"Deep research error: {str(e)}"

            else:
                return f"Unknown tool: {tool_name}"
        except Exception as e:
            return f"Error executing tool {tool_name}: {str(e)}"

    @torch.no_grad()
    def classify_input(self, prompt):
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
        self.model.eval()
        ids = self.model._chat_ids(prompt, history=self.history[-2:])
        input_ids = ids.unsqueeze(0).to(device)
        output_ids = self.model.generate(input_ids, max_new_tokens=self.max_new_tokens,
                                          current_block_size=self.context_length,
                                          temperature=self.temperature, top_k=self.top_k)
        # Decode ONLY the generated suffix (tokens after the input prompt),
        # never the whole re-decoded sequence. String-prefix stripping of the
        # re-tokenized prompt is fragile (whitespace round-trip drift) and on
        # multi-turn turns it fails, so the entire input context gets echoed
        # back as the "answer" and poisons history.
        generated = self.model.decode(output_ids[0][input_ids.shape[1]:].tolist())
        mm = re.search(r'<think>([\s\S]*?)(?:</think>|$)', generated)
        if mm:
            thought = mm.group(1).strip()
            resp = generated[mm.end():].strip()
        else:
            thought = ""
            resp = generated.strip()
        self.history.append((prompt, resp))
        return thought, resp

    @torch.no_grad()
    def _solve_qwen_stream(self, prompt):
        self.model.eval()
        self._source_harvest = []
        ids = self.model._chat_ids(prompt, history=self.history[-2:])
        input_ids = ids.unsqueeze(0).to(device)
        prompt_text = self.model.tokenizer.apply_chat_template(
            [{"role": "system", "content": self.model.system_prompt}]
            + [{"role": "user", "content": u} for u, _ in self.history[-2:]]
            + [{"role": "assistant", "content": a} for _, a in self.history[-2:]]
            + [{"role": "user", "content": prompt}],
            tokenize=False, add_generation_prompt=True)
        full = ""
        thought_sent = 0
        resp_sent = 0
        input_len = input_ids.shape[1]
        saw_close = False
        for full_index, tok_id in self.model.generate_stream(
                input_ids, max_new_tokens=self.max_new_tokens,
                current_block_size=self.context_length,
                temperature=self.temperature, top_k=self.top_k):
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
        mm = re.search(r'<think>([\s\S]*?)(?:</think>|$)', full)
        if mm:
            thought = mm.group(1).strip()
            resp = full[mm.end():].strip()
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
        self._source_harvest = []
        _label, _idx, _conf = self.classify_input(prompt)
        if _label == "spam":
            _spam_msg = "I don't respond to spam or unsolicited promotional messages."
            self.history.append((prompt, _spam_msg))
            return "", _spam_msg
        if _label == "request":
            self.pending_requests.append({"prompt": prompt, "confidence": _conf})

        if getattr(self.model, "is_qwen", False):
            return self._solve_qwen(prompt)

        conversation_context = ""
        for h_user, h_resp in self.history[-2:]:
            conversation_context += f"### Instruction:\n{h_user}\n\n### Response:\n{h_resp}\n\n"
        current_prompt = f"{conversation_context}### Instruction:\n{prompt}\n\n### Response:\n</think>"
        if getattr(self, 'auto_swarm_council', False):
            try:
                _enrich = enrich_prompt_with_swarm_council(prompt)
                if _enrich:
                    current_prompt = (f"{conversation_context}### Instruction:\n{prompt}\n\n"
                                       f"### Multi-Agent Deliberation (consult before answering):\n{_enrich}"
                                       f"### Response:\n<think>\n")
            except Exception as e:
                print(f"[Auto Swarm+Council] enrichment skipped: {e}", flush=True)
        if getattr(self, 'auto_web_research', False):
            try:
                _wr = gather_web_research(prompt)
                if _wr:
                    current_prompt = current_prompt.replace(
                        "### Response:\n<think>\n",
                        f"### Web Research (already gathered — ground your answer in these and cite as [n]):\n{_wr}\n\n### Response:\n</think>\n", 1)
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
            if self.use_draft_model and draft_model is not None:
                output_ids = self.generate_with_speculative_decoding(input_ids, self.max_new_tokens)
            else:
                output_ids = self.model.generate(input_ids, max_new_tokens=self.max_new_tokens,
                                                 current_block_size=self.context_length,
                                                 temperature=self.temperature, top_k=self.top_k)
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
                for am in re.finditer(r'([a-zA-Z_][a-zA-Z0-9_]*)\s*=\s*([\"\'])(.*?)\2', args_str):
                    kwargs[am.group(1)] = am.group(3)
                tool_obs = self.execute_tool(tool_name, kwargs)
                tool_observations.append(f"Tool: {tool_name}({args_str})\nObservation:\n{tool_obs}")
                current_prompt += f"{remainder}\n[OBSERVATION]:\n{tool_obs}\n</think>\n"
            else:
                final_answer = remainder
                break
        if not final_answer:
            final_answer = remainder
        combined_thought = "\n--- Ashen AI Reasoning Step ---\n".join(all_thoughts)
        if tool_observations:
            combined_thought += "\n\n--- Ashen AI Tool Telemetry ---\n" + "\n".join(tool_observations)
        clean_final = final_answer.strip()
        _seen, _collected_sources = set(), []
        for _s in getattr(self, '_source_harvest', []):
            _u = _s.get('url')
            if _u and _u not in _seen:
                _seen.add(_u)
                _collected_sources.append(_s)
        self.last_sources = _collected_sources
        self.history.append((prompt, f"<think>\n{combined_thought}\n</think>\n{clean_final}"))
        return combined_thought, clean_final

    @torch.no_grad()
    def solve_with_agent_stream(self, prompt):
        self.model.eval()
        self._source_harvest = []
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
        if getattr(self.model, "is_qwen", False):
            yield from self._solve_qwen_stream(prompt)
            return

        conversation_context = ""
        for h_user, h_resp in self.history[-2:]:
            conversation_context += f"### Instruction:\n{h_user}\n\n### Response:\n{h_resp}\n\n"
        current_prompt = f"{conversation_context}### Instruction:\n{prompt}\n\n### Response:\n</think>\n"
        if getattr(self, 'auto_swarm_council', False):
            try:
                _enrich = enrich_prompt_with_swarm_council(prompt)
                if _enrich:
                    current_prompt = (f"{conversation_context}### Instruction:\n{prompt}\n\n"
                                       f"### Multi-Agent Deliberation (consult before answering):\n{_enrich}"
                                       f"### Response:\n</think>\n")
            except Exception as e:
                print(f"[Auto Swarm+Council] enrichment skipped: {e}", flush=True)
        if getattr(self, 'auto_web_research', False):
            try:
                _wr = gather_web_research(prompt)
                if _wr:
                    current_prompt = current_prompt.replace(
                        "### Response:\n</think>\n",
                        f"### Web Research (already gathered — ground your answer in these and cite as [n]):\n{_wr}\n\n### Response:\n</think>\n", 1)
            except Exception as e:
                print(f"[Auto Web Research] skipped: {e}", flush=True)
        all_thoughts = []
        tool_observations = []
        final_answer = ""
        remainder = ""
        for step in range(self.max_steps):
            encoded = self.encode(current_prompt)
            if len(encoded) > self.context_length:
                encoded = encoded[-self.context_length:]
            input_ids = torch.tensor([encoded], dtype=torch.long, device=device)
            acc = ""
            saw_close = False
            thought_sent = 0
            resp_sent = 0
            for full_index, tok_id in self.model.generate_stream(
                    input_ids, max_new_tokens=self.max_new_tokens,
                    current_block_size=self.context_length, temperature=self.temperature, top_k=self.top_k):
                try:
                    chunk = self.decode([tok_id])
                except Exception:
                    chunk = ""
                if not chunk:
                    continue
                acc += chunk
                if not saw_close:
                    if "</think>" in acc:
                        saw_close = True
                        before, after = acc.split("</think>", 1)
                        new_thought = before[thought_sent:]
                        if new_thought:
                            yield {"type": "thought_delta", "chunk": new_thought}
                        thought_sent = len(before)
                        yield {"type": "thought_done"}
                        if after.strip():
                            yield {"type": "response_delta", "chunk": after}
                            resp_sent = len(after)
                    else:
                        new_thought = acc[thought_sent:]
                        if new_thought:
                            yield {"type": "thought_delta", "chunk": new_thought}
                        thought_sent = len(acc)
                else:
                    new_resp = acc[resp_sent:]
                    if new_resp:
                        yield {"type": "response_delta", "chunk": new_resp}
                        resp_sent = len(acc)
            if not saw_close:
                new_thought = acc[thought_sent:]
                if new_thought:
                    yield {"type": "thought_delta", "chunk": new_thought}
                yield {"type": "thought_done"}
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
                yield {"type": "tool_start", "tool": tool_name, "args": args_str}
                tool_obs = self.execute_tool(tool_name, kwargs)
                tool_observations.append(f"Tool: {tool_name}({args_str})\nObservation:\n{tool_obs}")
                yield {"type": "tool_result", "tool": tool_name, "observation": tool_obs[:800]}
                current_prompt += f"{remainder_local}\n[OBSERVATION]:\n{tool_obs}\n</think>\n"
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
        clean_final = final_answer.strip()
        _seen, _collected_sources = set(), []
        for _s in getattr(self, '_source_harvest', []):
            _u = _s.get('url')
            if _u and _u not in _seen:
                _seen.add(_u)
                _collected_sources.append(_s)
        self.last_sources = _collected_sources
        self.history.append((prompt, f"<think>\n{combined_thought}\n</think>\n{clean_final}"))
        yield {"type": "done", "thought": combined_thought, "response": clean_final,
               "model": os.path.basename(current_model_filename),
               "model_path": current_model_filename, "sources": _collected_sources,
               "intent": self.last_intent}


reasoner = AshenAIAgenticEngine(m, decode, encode, device)
reasoner.session_id = None
# apply persisted settings to the engine
reasoner.update_settings(settings)


# =====================================================================
#  Web research / swarm / council (backend, no UI)
# =====================================================================
def enrich_prompt_with_swarm_council(prompt, max_chars=1400, timeout=75):
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
        return ("The following multi-agent deliberation (Swarm + Council) has already "
                "analyzed this request. Use it to ground and improve YOUR chain-of-thought "
                "and final answer — adopt its strong points, reconcile disagreements, and "
                "deliver the best consolidated response.\n\n" + block + "\n\n")

    try:
        with _cf.ThreadPoolExecutor(max_workers=1) as ex:
            return ex.submit(_work).result(timeout=timeout)
    except _cf.TimeoutError:
        print(f"[Auto Swarm+Council] enrichment timed out after {timeout}s — using plain prompt.", flush=True)
        return ""
    except Exception as e:
        print(f"[Auto Swarm+Council] enrichment error: {e}", flush=True)
        return ""


def gather_web_research(prompt, max_searches=3, timeout=75):
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


_swarm_lock = threading.Lock()
_SWARM_ROLES = [
    ("Researcher", "You are the Researcher subagent. Focus on facts, sources, and concise research. Be precise and cite key points."),
    ("Coder", "You are the Coder subagent. Focus on implementation, code correctness, and practical steps."),
    ("Critic", "You are the Critic subagent. Find flaws, edge cases, and suggest improvements. Be skeptical but constructive."),
    ("Planner", "You are the Planner subagent. Break the task into steps and outline a clear plan before answering."),
    ("Executor", "You are the Executor subagent. Deliver a direct, actionable final answer with examples."),
    ("Analyst", "You are the Analyst subagent. Provide deep analysis, trade-offs, and quantitative reasoning."),
]


def _run_swarm(task, num_agents=3, mode="parallel"):
    import time as _t
    start = _t.time()
    n = max(2, min(6, int(num_agents)))
    mode = mode if mode in ("parallel", "divide", "debate") else "parallel"
    roles = [_SWARM_ROLES[i % len(_SWARM_ROLES)] for i in range(n)]
    if mode == "divide":
        subtasks = []
        base_parts = [t.strip() for t in re.split(r'[.;]\s*|\n+', task) if t.strip()]
        if len(base_parts) >= n:
            subtasks = base_parts[:n]
        else:
            subtasks = [task] * n
            for i in range(n):
                subtasks[i] = f"{task}\n\n[Sub-task focus for {roles[i][0]}: {roles[i][1][:80]}]"
    else:
        subtasks = [task] * n
    agents = []
    for idx, (role_name, role_desc) in enumerate(roles):
        agent_prompt = subtasks[idx]
        debate_context = ""
        if mode == "debate" and agents:
            prev = agents[-1]
            debate_context = f"\n\n[Previous agent {prev['role']} said: \"{prev['response'][:400]}\". Critique and improve it.]"
        tmp = AshenAIAgenticEngine(model, decode, encode, device, max_steps=3)
        tmp.persona = reasoner.persona
        tmp.temperature = reasoner.temperature
        tmp.top_k = reasoner.top_k
        tmp.top_p = reasoner.top_p
        tmp.max_new_tokens = reasoner.max_new_tokens
        tmp.context_length = reasoner.context_length
        try:
            tmp.history = list(reasoner.history[-1:])
        except Exception:
            tmp.history = []
        full_prompt = agent_prompt + debate_context + f"\n\n[{role_desc}]"
        with _swarm_lock:
            try:
                thought, response = tmp.solve_with_agent(full_prompt)
            except Exception as e:
                thought = f"Subagent {role_name} error: {e}"
                response = f"Failed to generate: {e}"
        agents.append({'id': idx + 1, 'role': role_name, 'role_desc': role_desc,
                       'thought': thought, 'response': response,
                       'model': os.path.basename(current_model_filename)})
    synthesis_thought = f"Swarm synthesis: {n} subagents ({', '.join([a['role'] for a in agents])}) in mode '{mode}' completed task: \"{task[:120]}\""
    drafts = "\n\n".join([f"--- Agent {a['id']} ({a['role']}) ---\n{a['response'][:800]}" for a in agents])
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
        synthesis_thought = synth_thought_raw
        synthesis_response = synth_response
    except Exception as e:
        synthesis_response = f"Swarm synthesis failed ({e}), falling back to best draft:\n\n{drafts[:1500]}"
        synthesis_thought += f"\nSynthesis error: {e}"
    if not synthesis_response.strip():
        best = max(agents, key=lambda a: len(a['response']))
        if best['response'].strip():
            synthesis_response = f"**Swarm synthesis (fallback to {best['role']}):**\n\n{best['response']}\n\n*All drafts considered — see individual agents above.*"
            synthesis_thought += f"\n[Fallback to {best['role']} draft]"
    elapsed = round(time.time() - start, 2)
    try:
        _append_improvement({'type': 'swarm', 'prompt': task[:80], 'mode': mode, 'agents': n,
                              'elapsed': elapsed, 'stats_delta': {}},
                             suggestion=f"Swarm {n}x{mode} completed \"{task[:60]}\"")
    except Exception:
        pass
    return {'task': task, 'mode': mode, 'num_agents': n, 'agents': agents,
            'synthesis': {'thought': synthesis_thought, 'response': synthesis_response},
            'elapsed_s': elapsed, 'model': os.path.basename(current_model_filename),
            'model_path': current_model_filename}


_COUNCIL_CRITIC_ROLES = [
    ("Accuracy Critic", "You are the Accuracy Critic. Check factual correctness, catch hallucinations, verify claims. Score 1-10."),
    ("Clarity Critic", "You are the Clarity Critic. Check structure, readability, conciseness. Is the answer easy to follow? Score 1-10."),
    ("Completeness Critic", "You are the Completeness Critic. Check if all parts of the task are addressed. What's missing? Score 1-10."),
    ("Safety Critic", "You are the Safety Critic. Check for harmful, biased or unsafe content. Score 1-10."),
    ("Efficiency Critic", "You are the Efficiency Critic. Check for conciseness vs depth, suggest simplifications. Score 1-10."),
]


def _run_council(task, num_drafts=3, num_critics=3):
    start = time.time()
    nd = max(2, min(5, int(num_drafts)))
    nc = max(2, min(5, int(num_critics)))
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
        try:
            tmp.history = list(reasoner.history[-1:])
        except Exception:
            tmp.history = []
        full_prompt = f"{task}\n\n[Proposer role: {role_name} — {role_desc}]"
        with _swarm_lock:
            try:
                thought, response = tmp.solve_with_agent(full_prompt)
            except Exception as e:
                thought = f"Draft {role_name} error: {e}"
                response = f"Failed: {e}"
        drafts.append({'id': idx + 1, 'role': role_name, 'thought': thought,
                       'response': response, 'model': os.path.basename(current_model_filename)})

    critic_roles = [_COUNCIL_CRITIC_ROLES[i % len(_COUNCIL_CRITIC_ROLES)] for i in range(nc)]
    critics = []

    def _heuristic_score(text, prompt):
        if not text.strip():
            return 3
        p_words = set(re.findall(r'[a-z]{3,}', prompt.lower()))
        t_words = set(re.findall(r'[a-z]{3,}', text.lower()))
        overlap = len(p_words & t_words) / max(1, len(p_words))
        score = 5 + int(overlap * 3) + min(2, len(text) // 350)
        return max(1, min(10, score))

    drafts_block = "\n\n".join([f"[Draft {d['id']} ({d['role']}):]\n{d['response'][:900]}" for d in drafts])
    for c_idx, (critic_name, critic_desc) in enumerate(critic_roles):
        tmp = AshenAIAgenticEngine(model, decode, encode, device, max_steps=2)
        tmp.persona = reasoner.persona
        tmp.temperature = 0.65
        tmp.top_k = reasoner.top_k
        tmp.top_p = reasoner.top_p
        tmp.max_new_tokens = 220
        tmp.context_length = reasoner.context_length
        try:
            tmp.history = list(reasoner.history[-1:])
        except Exception:
            tmp.history = []
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
        votes = {}
        suggestions = {}
        vote_pick = None
        for line in c_response.splitlines():
            vm = re.search(r'Draft\s+(\d+)\s*:\s*Score\s+(\d+)', line, re.I)
            if vm:
                did = int(vm.group(1))
                sc = max(1, min(10, int(vm.group(2))))
                votes[did] = sc
                seg = re.split(r'Suggestion\s*:\s*', line, flags=re.I)
                if len(seg) > 1:
                    suggestions[did] = seg[1].strip()[:200]
                elif '-' in line:
                    suggestions[did] = line.split('-', 1)[1].strip()[:200]
            mv = re.search(r'VOTE\s*:\s*(\d+)', line, re.I)
            if mv:
                vote_pick = int(mv.group(1))
        if not votes:
            for d in drafts:
                votes[d['id']] = _heuristic_score(d['response'], task)
                suggestions[d['id']] = f"[{critic_name} heuristic] Improve relevance to \"{task[:40]}\""
            vote_pick = max(votes, key=lambda k: votes[k])
        if vote_pick is None:
            vote_pick = max(votes, key=lambda k: votes[k]) if votes else drafts[0]['id']
        critics.append({'id': c_idx + 1, 'role': critic_name, 'desc': critic_desc,
                        'thought': c_thought, 'response': c_response, 'votes': votes,
                        'suggestions': suggestions, 'pick': vote_pick})
    tally = {d['id']: 0 for d in drafts}
    score_sum = {d['id']: 0 for d in drafts}
    for c in critics:
        for did, sc in c['votes'].items():
            if did in score_sum:
                score_sum[did] += sc
        if c['pick'] in tally:
            tally[c['pick']] += 1
    sorted_drafts = sorted(drafts, key=lambda d: (tally.get(d['id'], 0), score_sum.get(d['id'], 0), len(d['response'])), reverse=True)
    winner = sorted_drafts[0]
    winner_suggestions = []
    for c in critics:
        if c['suggestions'].get(winner['id']):
            winner_suggestions.append(f"- [{c['role']}] {c['suggestions'][winner['id']]}")
    if not winner_suggestions:
        winner_suggestions = [f"- [{c['role']}] {list(c['suggestions'].values())[0] if c['suggestions'] else 'No suggestion'}" for c in critics[:2]]
    suggestions_block = "\n".join(winner_suggestions[:5])
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
        final_response = winner['response'] + "\n\n**Council refinements applied:**\n" + suggestions_block
        final_thought = winner['thought'] + "\n\n[Council synthesis fallback — winner {winner['role']} + critiques]"
    elapsed = round(time.time() - start, 2)
    try:
        _append_improvement({'type': 'council', 'prompt': task[:80], 'drafts': nd, 'critics': nc,
                              'winner': winner['id'], 'elapsed': elapsed, 'stats_delta': {}},
                             suggestion=f"Council {nd} drafts + {nc} critics → winner Draft {winner['id']} ({winner['role']})")
    except Exception:
        pass
    return {'task': task, 'num_drafts': nd, 'num_critics': nc, 'drafts': drafts, 'critics': critics,
            'tally': tally, 'score_sum': score_sum, 'winner': winner,
            'suggestions': winner_suggestions,
            'final': {'thought': final_thought, 'response': final_response},
            'elapsed_s': elapsed, 'model': os.path.basename(current_model_filename),
            'model_path': current_model_filename}


# =====================================================================
#  Model hub
# =====================================================================
def _is_hf_model_dir(path):
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
    models = []
    for search_dir in ['.', os.path.expanduser('~/.cache/huggingface/hub')]:
        if not os.path.exists(search_dir):
            continue
        for root, dirs, files in os.walk(search_dir):
            dirs[:] = [d for d in dirs if not d.startswith('.')]
            if _is_hf_model_dir(root):
                full_path = os.path.abspath(root).replace('\\', '/')
                try:
                    size_mb = round(sum(os.path.getsize(os.path.join(root, f))
                                        for f in os.listdir(root)
                                        if f.endswith('.safetensors') or f.endswith('.bin')) / (1024 * 1024), 1)
                except OSError:
                    size_mb = 0.0
                is_active = (os.path.normpath(full_path) == os.path.normpath(current_model_filename))
                models.append({'path': full_path, 'name': os.path.basename(root),
                               'size_mb': size_mb, 'active': is_active, 'type': 'qwen-hf'})
                continue
            for file in files:
                if file.endswith(('.pk1', '.pt', '.pth', '.gguf')):
                    full_path = os.path.abspath(os.path.join(root, file)).replace('\\', '/')
                    size_mb = round(os.path.getsize(full_path) / (1024 * 1024), 1)
                    is_active = (os.path.normpath(full_path) == os.path.normpath(current_model_filename))
                    models.append({'path': full_path, 'name': file, 'size_mb': size_mb, 'active': is_active})
    return sorted(models, key=lambda m: m['active'], reverse=True)


def set_default_model(model_path):
    global current_model_filename
    normalized_path = os.path.normpath(model_path)
    if os.path.exists(normalized_path):
        current_model_filename = normalized_path
        return True
    print(f"[Model] Path not found: {normalized_path}")
    return False


def switch_model(model_path):
    """Hot-swap the active model (mirrors /api/models/switch in web_chatbot.py)."""
    global model, m, reasoner
    normalized = os.path.normpath(model_path)
    if not (os.path.exists(normalized) or os.path.isdir(normalized)):
        print(f"[Model] Not found: {normalized}")
        return False
    if os.path.isdir(normalized):
        _ch = os.path.join(normalized, "class_head.pt")
        try:
            model = QwenModelAdapter(normalized, device, class_head_path=_ch)
        except Exception as e:
            print(f"[Model] Qwen load failed ({e})")
            return False
    else:
        try:
            with open(normalized, 'rb') as f:
                model = _load_ashen_checkpoint(normalized)
        except Exception as e:
            print(f"[Model] pickle load failed ({e}); trying torch.load...")
            try:
                model = torch.load(normalized, map_location=device)
            except Exception as te:
                print(f"[Model] load failed ({te})")
                return False
    m = model.to(device) if not getattr(model, "is_qwen", False) else model
    reasoner.model = m
    current_model_filename = normalized
    settings['current_model'] = normalized
    save_settings_to_json({'current_model': normalized})
    print(f"[Model] switched -> {normalized}  ({'Qwen' if getattr(model, 'is_qwen', False) else 'custom'})")
    return True


# =====================================================================
#  Session management helpers
# =====================================================================
def _session_file(session_id):
    return os.path.join(SESSIONS_DIR, f"{session_id}.json")


def save_session(session_id, data):
    filepath = _session_file(session_id)
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def load_session(session_id):
    filepath = _session_file(session_id)
    if not os.path.exists(filepath):
        return None
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)


def list_sessions():
    sessions = []
    for fname in os.listdir(SESSIONS_DIR):
        if not fname.endswith('.json'):
            continue
        sid = fname[:-5]
        try:
            with open(_session_file(sid), 'r', encoding='utf-8') as f:
                data = json.load(f)
            sessions.append({'id': sid, 'name': data.get('name', 'Untitled'),
                             'updated': data.get('updated_at', ''),
                             'message_count': len(data.get('history', [])),
                             'workspace_context': data.get('workspace_context', '')})
        except Exception:
            pass
    sessions.sort(key=lambda s: s.get('updated', ''), reverse=True)
    return sessions


def delete_session(session_id):
    filepath = _session_file(session_id)
    if os.path.exists(filepath):
        os.remove(filepath)
        return True
    return False


def create_new_session(name=None):
    global current_session_id
    ts = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    current_session_id = f"session_{ts}"
    save_session(current_session_id, {'name': name or 'Untitled',
                                      'created_at': datetime.datetime.now().isoformat(),
                                      'updated_at': datetime.datetime.now().isoformat(),
                                      'history': [], 'workspace_context': ''})
    return current_session_id


def append_to_session(session_id, user_msg, assistant_resp):
    global current_session_id
    data = load_session(session_id)
    if not data:
        return
    data['history'].append({'user': user_msg, 'assistant': assistant_resp})
    data['updated_at'] = datetime.datetime.now().isoformat()
    save_session(session_id, data)
    current_session_id = session_id


def rename_session(session_id, name):
    data = load_session(session_id)
    if data:
        data['name'] = name
        save_session(session_id, data)
        return True
    return False


# Working directory for tools
WORKING_DIR = os.getcwd()


def change_working_dir(dir_path):
    global WORKING_DIR
    if not dir_path:
        return False
    if not os.path.isdir(dir_path):
        return False
    WORKING_DIR = os.path.abspath(dir_path)
    return True


# =====================================================================
#  Settings / persona / swarm / council / research CLI commands
# =====================================================================
def cmd_models():
    models = scan_available_models()
    if not models:
        print("No models found.")
        return
    print(f"{'ACTIVE':<6} {'NAME':<28} {'SIZE(MB)':>9}  PATH")
    print("-" * 90)
    for mdl in models:
        active = "●" if mdl.get('active') else ""
        print(f"{active:<6} {mdl['name']:<28} {mdl.get('size_mb', 0):>9}  {mdl['path']}")


def cmd_model(arg):
    path = arg.strip()
    if switch_model(path):
        _apply_settings_to_engine()


def cmd_settings_show():
    print(json.dumps(settings, indent=2))


def cmd_settings_set(arg):
    # arg: key=value key2=value2   (int/float/bool parsed loosely)
    updates = {}
    for pair in arg.split():
        if '=' not in pair:
            continue
        k, v = pair.split('=', 1)
        k = k.strip()
        v = v.strip()
        if v.lower() in ('true', 'false'):
            v = (v.lower() == 'true')
        else:
            try:
                v = int(v)
            except ValueError:
                try:
                    v = float(v)
                except ValueError:
                    pass
        updates[k] = v
    if not updates:
        print("[Settings] usage: /settings key=value ...")
        return
    settings.update(updates)
    save_settings_to_json(updates)
    _apply_settings_to_engine()
    print(f"[Settings] saved: {updates}")


def cmd_persona(arg):
    reasoner.set_persona(arg.strip())
    print(f"[Persona] set to: {reasoner.persona}")


def _apply_settings_to_engine():
    reasoner.update_settings(settings)


def cmd_swarm(arg):
    parts = arg.split()
    task = arg.strip()
    num_agents = 3
    mode = "parallel"
    if task.startswith('--'):
        # allow: /swarm --agents N --mode M task...
        import re as _re
        ma = _re.search(r'--agents\s+(\d+)', arg)
        if ma:
            num_agents = int(ma.group(1))
        mm = _re.search(r'--mode\s+(\w+)', arg)
        if mm:
            mode = mm.group(1)
        task = _re.sub(r'--agents\s+\d+', '', arg)
        task = _re.sub(r'--mode\s+\w+', '', task).strip()
    if not task:
        task = input("Swarm task: ").strip()
    print(f"\n[Swarm] running {num_agents} agents in '{mode}' mode...\n")
    result = _run_swarm(task, num_agents=num_agents, mode=mode)
    print(f"\n=== Swarm Synthesis (elapsed {result['elapsed_s']}s) ===\n")
    print(result['synthesis']['response'])
    print("\n--- Agent drafts ---")
    for a in result['agents']:
        print(f"\n[{a['role']}] {a['response'][:400]}")


def cmd_council(arg):
    parts = arg.split()
    task = arg.strip()
    num_drafts, num_critics = 3, 3
    import re as _re
    md = _re.search(r'--drafts\s+(\d+)', arg)
    if md:
        num_drafts = int(md.group(1))
    mc = _re.search(r'--critics\s+(\d+)', arg)
    if mc:
        num_critics = int(mc.group(1))
    task = _re.sub(r'--drafts\s+\d+', '', task)
    task = _re.sub(r'--critics\s+\d+', '', task).strip()
    if not task:
        task = input("Council task: ").strip()
    print(f"\n[Council] {num_drafts} drafts x {num_critics} critics...\n")
    result = _run_council(task, num_drafts=num_drafts, num_critics=num_critics)
    print(f"\n=== Council Final Answer (winner Draft {result['winner']['id']} / {result['winner']['role']}) ===\n")
    print(result['final']['response'])
    print(f"\n[Critiques] {result['suggestions']}")


def cmd_research(arg):
    topic = arg.strip() or input("Research topic: ").strip()
    print(f"\n[Deep Research] {topic} ...\n")
    report = reasoner.execute_tool('deep_research', {'topic': topic, 'max_searches': 5})
    print(report)


def cmd_websearch(arg):
    q = arg.strip() or input("Search query: ").strip()
    print(reasoner.execute_tool('web_search', {'query': q}))


def cmd_improve():
    log = _load_improvement_log()
    print(f"Total feedback: {log['stats'].get('total_feedback', 0)}  "
          f"up: {log['stats'].get('up', 0)}  down: {log['stats'].get('down', 0)}")
    print(f"Corrections: {log['stats'].get('corrections', 0)}  auto_tunes: {log['stats'].get('auto_tunes', 0)}")
    sugg = log.get('suggestions', [])
    if sugg:
        print("\nLatest suggestions:")
        for s in sugg[-8:]:
            print(f"  - [{s.get('ts', '')[:19]}] {s.get('text', '')[:120]}")
    entries = log.get('entries', [])
    if entries:
        print("\nRecent activity:")
        for e in entries[-10:]:
            print(f"  - {e.get('type')}: {str(e.get('prompt', ''))[:60]}")


def cmd_feedback_up():
    _save_feedback_entry({'rating': 'up', 'ts': datetime.datetime.now().isoformat()})
    _self_improvement_stats['total_feedback'] += 1
    _self_improvement_stats['up'] += 1
    print("[Feedback] 👍 recorded")


def cmd_feedback_down():
    _save_feedback_entry({'rating': 'down', 'ts': datetime.datetime.now().isoformat()})
    _self_improvement_stats['total_feedback'] += 1
    _self_improvement_stats['down'] += 1
    print("[Feedback] 👎 recorded")


def cmd_selfimprove(arg):
    # /selfimprove analyze | auto-tune | regenerate <text>
    sub = arg.strip().split()[0] if arg.strip() else 'analyze'
    log = _load_improvement_log()
    if sub == 'analyze':
        sugg = log.get('suggestions', [])
        if sugg:
            print("Suggestions from learning log:")
            for s in sugg[-10:]:
                print(f"  - {s.get('text', '')[:140]}")
        else:
            print("No suggestions yet. Chat / give feedback to build the learning log.")
    elif sub == 'auto-tune':
        # Suggested temperature nudge based on feedback ratio (heuristic).
        up = _self_improvement_stats['up']
        down = _self_improvement_stats['down']
        if up + down >= 3:
            if down > up:
                new_t = max(0.3, round(settings['temperature'] - 0.05, 2))
            else:
                new_t = min(1.2, round(settings['temperature'] + 0.02, 2))
            settings['temperature'] = new_t
            save_settings_to_json({'temperature': new_t})
            _apply_settings_to_engine()
            _self_improvement_stats['auto_tunes'] += 1
            _append_improvement({'type': 'auto-tune', 'prompt': '', 'stats_delta': {}},
                                suggestion=f"auto-tuned temperature -> {new_t}")
            print(f"[SelfImprove] auto-tuned temperature -> {new_t}")
        else:
            print("[SelfImprove] not enough feedback yet to tune (need >=3 ratings).")
    elif sub == 'regenerate':
        text = arg.strip()[len('regenerate'):].strip()
        if text:
            print("[SelfImprove] regenerating with self-critique...")
            thought, ans = reasoner.solve_with_agent(
                f"Regenerate with self-critique: be more direct and relevant to the prompt.\n\n{text}")
            print(f"\n<think>\n{thought}\n</think>\n\n{ans}")
    else:
        print("[SelfImprove] usage: /selfimprove analyze | auto-tune | regenerate <text>")


# =====================================================================
#  Terminal theme — cybernetic
# =====================================================================
RESET = "\033[0m"
BOLD  = "\033[1m"
DIM   = "\033[2m"
GREY  = "\033[90m"
AMBER = "\033[33m"
CYAN  = "\033[96m"   # cybernetic primary
MAG   = "\033[95m"   # cybernetic accent
GREEN = "\033[92m"   # cybernetic ok / sources
RED   = "\033[91m"   # errors
WHITE = "\033[97m"   # bright white answer text
# Box-drawing glyphs for the cybernetic frame
BOX_TL, BOX_TR, BOX_BL, BOX_BR, BOX_H, BOX_V = "\u2554", "\u2557", "\u255a", "\u255d", "\u2550", "\u2551"


def _supports_ansi():
    if os.environ.get("NO_COLOR"):
        return False
    env = os.environ.get("ASHEN_CYBER")
    if env == "0":
        return False
    if env == "1":
        return True
    return sys.stdout.isatty()


_CYBER = _supports_ansi()


def _apply_theme():
    global GREY, AMBER, BOLD, RESET, CYAN, MAG, GREEN, RED, DIM, WHITE, THEME
    if _CYBER:
        GREY  = "\033[90m"; AMBER = "\033[33m"; BOLD = "\033[1m"
        RESET = "\033[0m";  CYAN  = "\033[96m"; MAG  = "\033[95m"
        GREEN = "\033[92m"; RED   = "\033[91m"; DIM  = "\033[2m"; WHITE = "\033[97m"
    else:
        GREY = AMBER = BOLD = RESET = CYAN = MAG = GREEN = RED = DIM = WHITE = ""
    THEME = {
        "user":  CYAN + BOLD,
        "ai":    MAG + BOLD,
        "cot":   GREY,
        "hdr":   CYAN,
        "ok":    GREEN,
        "warn":  AMBER,
        "err":   RED,
        "dim":   DIM,
        "ans":   WHITE,
        "reset": RESET,
        "box":   (BOX_TL, BOX_TR, BOX_BL, BOX_BR, BOX_H, BOX_V),
    }


_apply_theme()


def set_cyber(on):
    """Toggle the cybernetic theme at runtime."""
    global _CYBER
    _CYBER = bool(on)
    _apply_theme()


def cyber_box(title, lines):
    if not _CYBER:
        out = [title]
        out.extend("  " + l for l in lines)
        return "\n".join(out)
    tl, tr, bl, br, h, v = THEME["box"]
    def _aw(s):
        return len(re.sub(r"\033\[[0-9;]*m", "", s))
    w = max(_aw(title), max((_aw(l) for l in lines), default=0))
    bar = h * (w + 2)
    out = [f"{tl}{bar}{tr}", f"{v} {title}{' ' * (w - _aw(title))} {v}"]
    for l in lines:
        out.append(f"{v} {l}{' ' * (w - _aw(l))} {v}")
    out.append(f"{bl}{bar}{br}")
    return "\n".join(out)


# =====================================================================
#  Streaming render helpers
# =====================================================================


def run_streaming(prompt):
    """Drive solve_with_agent_stream and render the chain-of-thought exactly
    like the web chatbot:

      ● Thought for <model>          (amber dot + model label — web's <summary>)
      <gray streamed chain-of-thought>
      · Xs · ~N tokens               (timing filled on thought_done — web does same)
      <white streamed answer>
      ◈ <model> <time>               (footer — web's liveFooter)
      Sources                        (when present — web's renderSources)
        1› title — url
      [intent is internal to both — not surfaced per turn, matching web]

    The event schema and ordering come from AshenAIAgenticEngine
    .solve_with_agent_stream, which is identical to the web's engine, so the
    thought/response/source content is guaranteed the same as the web UI.
    """
    show_cot = settings.get('show_chain_of_thought', True)
    model_label = os.path.basename(current_model_filename)
    _think_start = time.time()
    _tok = 0
    resp_buf = []
    cot_buf = []
    for ev in reasoner.solve_with_agent_stream(prompt):
        t = ev.get('type')
        if t == 'thought_delta' and show_cot:
            if not cot_buf:
                print(f"{THEME['warn']}● Thought for {model_label}{RESET}")
            _tok += len(ev['chunk'].split())
            cot_buf.append(ev['chunk'])
            print(THEME['cot'] + ev['chunk'] + RESET, end='', flush=True)
        elif t == 'thought_done' and show_cot:
            _secs = time.time() - _think_start
            print(f"\n{THEME['dim']}· {_secs:.1f}s · ~{_tok} tokens{RESET}")
        elif t == 'response_delta':
            if not resp_buf:
                print()
            resp_buf.append(ev['chunk'])
            print(THEME['ans'] + ev['chunk'] + RESET, end='', flush=True)
        elif t == 'tool_start':
            print(f"\n{THEME['warn']}[TOOL: {ev.get('tool', '')}({ev.get('args', '')})]{RESET}")
        elif t == 'tool_result':
            _obs = (ev.get('observation') or '')[:120]
            print(f"{THEME['dim']}{_obs}{RESET}")
        elif t == 'done':
            print(RESET)
            srcs = ev.get('sources') or []
            if srcs:
                print(f"{THEME['ok']}Sources{RESET}")
                for i, s in enumerate(srcs, 1):
                    print(f"  {i}› {s.get('title', s.get('url'))} — {s.get('url')}")
            _now = time.strftime('%H:%M:%S')
            print(f"{THEME['dim']}◈ {model_label} · {_now}{RESET}")
            full_resp = ''.join(resp_buf).strip()
            return ''.join(cot_buf), full_resp
    return ''.join(cot_buf), ''.join(resp_buf).strip()


def run_once(prompt):
    thought, ans = reasoner.solve_with_agent(prompt)
    show_cot = settings.get('show_chain_of_thought', True)
    if show_cot and thought:
        print(f"\n{THEME['hdr']}◇ CORE :: THOUGHT CYCLE ◇{RESET}")
        print(f"{THEME['cot']}{thought}{RESET}")
    print(f"\n{THEME['ai']}◆ ASHEN GPT ◆{RESET}\n{THEME['ans']}{ans}{RESET}")
    if getattr(reasoner, 'last_sources', []):
        print(f"\n{THEME['ok']}◇ SOURCES ◇{RESET}")
        for i, s in enumerate(reasoner.last_sources, 1):
            print(f"  {i}› {s.get('title', s.get('url'))} — {s.get('url')}")
    return thought, ans


# Reserved rows for the pinned input box (top border, prompt line, bottom border)
BOX_H = 3


def _setup_pinned():
    """Reserve the bottom BOX_H rows for the input box via an ANSI scroll
    region, so the conversation scrolls *above* the box.

    No-op (returns False) when the cybernetic theme is off, stdout isn't a
    TTY, or the terminal is too short. Otherwise sets the scroll region to
    rows 1..(rows-BOX_H) and returns True.
    """
    if not _CYBER or not sys.stdout.isatty():
        return False
    cols, rows = shutil.get_terminal_size((80, 24))
    if rows < BOX_H + 5:
        return False
    sys.stdout.write(f"\033[1;{rows - BOX_H}r")  # scroll region = top .. (rows-BOX_H)
    sys.stdout.write("\033[H")                    # cursor to home
    sys.stdout.flush()
    return True


def _teardown_pinned():
    """Restore the full-screen scroll region and clear."""
    try:
        sys.stdout.write("\033[r")          # reset scroll region to full screen
        sys.stdout.write("\033[2J\033[H")    # clear + home
        sys.stdout.flush()
    except Exception:
        pass


def _input_box():
    """Pinned cybernetic input box at the bottom of the terminal.

    The box is anchored in the bottom BOX_H rows (outside the scroll region)
    so it stays visible while the model's output scrolls in the region above
    it. Falls back to a plain prompt when pinning is unavailable.
    """
    if not _CYBER:
        return input(f"{THEME['user']}◇ YOU ▷ {RESET}")
    cols, rows = shutil.get_terminal_size((80, 24))
    if rows < BOX_H + 5:
        return input(f"{THEME['user']}◇ YOU ▷ {RESET}")
    inner = max(10, cols - 2)
    tl, tr, bl, br, h, v = THEME["box"]
    start_row = rows - BOX_H + 1
    prompt_row = start_row + 1
    bottom_row = start_row + 2
    # Draw the complete frame in the reserved bottom region. The scroll region
    # above it remains available for the model's response.
    sys.stdout.write(f"\033[{start_row};1H\033[0J")
    sys.stdout.write(f"{tl}{h * inner}{tr}")
    sys.stdout.write(f"\033[{bottom_row};1H{bl}{h * inner}{br}")
    sys.stdout.write(f"\033[{prompt_row};1H{v} {THEME['user']}◇ YOU ▷ {RESET}")
    sys.stdout.flush()
    val = input()
    # Clear only the prompt line after Enter, then redraw the empty prompt and
    # bottom border. The box remains visible while the model responds.
    sys.stdout.write(f"\033[{prompt_row};1H\033[2K")
    sys.stdout.write(f"{v} {THEME['user']}◇ YOU ▷ {RESET}")
    sys.stdout.write(f"\033[{bottom_row};1H{bl}{h * inner}{br}")
    # Park the cursor at the bottom of the scroll region so model output stays
    # above the still-visible box.
    sys.stdout.write(f"\033[{rows - BOX_H};1H")
    sys.stdout.flush()
    return val


# =====================================================================
#  Main loop
# =====================================================================
if __name__ == "__main__":
    if os.name == "nt":
        try:
            os.system("")  # enable VT100 / ANSI on Windows consoles
        except Exception:
            pass
    _cli_args = parse_cli_args()
    if _cli_args.settings and os.path.abspath(_cli_args.settings) != os.path.abspath(SETTINGS_FILE):
        SETTINGS_FILE = os.path.abspath(_cli_args.settings)
        settings = load_settings_from_json(SETTINGS_FILE)
        reasoner.update_settings(settings)

    print()
    print(THEME['hdr'] + cyber_box("ASHEN GPT // AGENTIC CORE", [
        f"{THEME['ai']}model ›{RESET} {os.path.basename(current_model_filename)} "
        f"({('Qwen' if getattr(model, 'is_qwen', False) else 'custom .pk1')})",
        f"{THEME['ai']}core  ›{RESET} chain-of-thought · intent-classify · agentic tools",
        f"{THEME['ai']}mesh  ›{RESET} Swarm+Council · deep-web research · self-improvement",
    ]) + RESET)
    print(f"{THEME['dim']}› /help  /models  /model <path>  /settings  /cyber on|off{RESET}")

    _setup_pinned()  # reserve the bottom rows for the pinned input box

    current_session_id = create_new_session("Initial Session")
    reasoner.session_id = current_session_id

    # Optional one-shot benchmark (mirrors the web 'run_benchmark' tool)
    def run_benchmark():
        BENCHMARK_TESTS = [
            {"category": "Knowledge", "instruction": "What year was Python programming language first released?",
             "expected_keywords": ["1991"]},
            {"category": "Knowledge", "instruction": "What does RAM stand for in computer science?",
             "expected_keywords": ["random", "access", "memory"]},
            {"category": "Code Generation", "instruction": "Write a Python function fibonacci(n) using recursion.",
             "expected_keywords": ["def", "fibonacci", "return"]},
            {"category": "Math", "instruction": "Calculate: 2 + 2 = ? Show your work.",
             "expected_keywords": ["4"]},
            {"category": "Language", "instruction": "How do you say 'Hello World' in Spanish?",
             "expected_keywords": ["hola", "mundo"]},
        ]
        total = earned = 0
        for i, t in enumerate(BENCHMARK_TESTS, 1):
            print(f"[Benchmark] {i}/{len(BENCHMARK_TESTS)}: {t['category']}", flush=True)
            ids = torch.tensor([encode(t['instruction'])], dtype=torch.long, device=device)
            with torch.no_grad():
                out = model.generate(ids, max_new_tokens=200, current_block_size=8192,
                                      temperature=0.7, top_k=40)
            resp = decode(out[0].tolist()).lower()
            k = sum(1 for kw in t['expected_keywords'] if kw in resp)
            sc = (k / len(t['expected_keywords'])) * 2
            earned += sc
            total += 2
        pct = (earned / total * 100) if total else 0
        print(f"\n=== Benchmark: {earned:.1f}/{total} ({pct:.1f}%) ===")

    while True:
        try:
            prompt = _input_box()
            if not prompt.strip():
                continue
            cmd = prompt.strip().lower()

            if cmd in ('exit', 'quit', '/exit', '/quit'):
                print("Goodbye!")
                break
            if cmd == '/clear':
                if current_session_id and load_session(current_session_id):
                    save_session(current_session_id, {
                        'name': load_session(current_session_id)['name'],
                        'history': list(reasoner.history),
                        'workspace_context': reasoner.workspace_context})
                reasoner.clear_history()
                reasoner.set_workspace_context("")
                print("[Conversation history cleared]")
                continue
            if cmd == '/sessions':
                ss = list_sessions()
                if not ss:
                    print("No sessions found.")
                else:
                    print(f"{'ID':<25} {'Name':<20} {'Msgs':>6}  Updated")
                    print("-" * 80)
                    for s in ss:
                        print(f"{s['id']:<25} {s['name']:<20} {s['message_count']:>6}  {s['updated'][:16]}")
                continue
            if cmd == '/new':
                current_session_id = create_new_session("New Session")
                reasoner.session_id = current_session_id
                reasoner.history = []
                reasoner.workspace_context = ""
                print(f"[Created new session: {current_session_id}]")
                continue
            if cmd.startswith('/load '):
                sid = cmd[6:].strip()
                data = load_session(sid) if sid else None
                if data:
                    current_session_id = sid
                    reasoner.session_id = sid
                    reasoner.history = [(m['user'], m['assistant']) for m in data.get('history', [])]
                    reasoner.workspace_context = data.get('workspace_context', '')
                    print(f"[Loaded session '{data['name']}' with {len(data['history'])} messages]")
                else:
                    print(f"[Session not found: {sid}]")
                continue
            if cmd.startswith('/delete '):
                sid = cmd[8:].strip()
                if delete_session(sid):
                    if current_session_id == sid:
                        current_session_id = None
                        reasoner.session_id = None
                        reasoner.history = []
                    print(f"[Deleted session: {sid}]")
                else:
                    print(f"[Session not found: {sid}]")
                continue
            if cmd.startswith('/rename '):
                name = cmd[8:].strip()
                if rename_session(current_session_id, name):
                    print(f"[Renamed session to '{name}']")
                else:
                    print("[Failed to rename session]")
                continue
            if cmd.startswith('/workspace '):
                dir_path = cmd[11:].strip()
                if os.path.isdir(dir_path):
                    reasoner.set_workspace_context(dir_path)
                    print(f"[Workspace context set: {dir_path}]")
                else:
                    print(f"[Invalid directory: {dir_path}]")
                continue
            if cmd in ('/wctx off', '/workspace off'):
                reasoner.set_workspace_context("")
                print("[Workspace context cleared]")
                continue
            if cmd.startswith('/cd '):
                d = cmd[4:].strip()
                if not d:
                    print(f"[Working directory: {WORKING_DIR}]")
                elif change_working_dir(d):
                    print(f"[Working directory changed to: {WORKING_DIR}]")
                else:
                    print(f"[Invalid directory: {d}]")
                continue
            if cmd == '/pwd':
                print(f"[Working directory: {WORKING_DIR}]")
                continue
            if cmd == '/models':
                cmd_models()
                continue
            if cmd.startswith('/model '):
                cmd_model(cmd[7:])
                continue
            if cmd == '/settings':
                cmd_settings_show()
                continue
            if cmd.startswith('/settings '):
                cmd_settings_set(cmd[9:])
                continue
            if cmd.startswith('/persona '):
                cmd_persona(cmd[9:])
                continue
            if cmd.startswith('/swarm'):
                cmd_swarm(cmd[6:])
                continue
            if cmd.startswith('/council'):
                cmd_council(cmd[8:])
                continue
            if cmd.startswith('/research'):
                cmd_research(cmd[9:])
                continue
            if cmd.startswith('/websearch'):
                cmd_websearch(cmd[10:])
                continue
            if cmd.startswith('/selfimprove'):
                cmd_selfimprove(cmd[11:])
                continue
            if cmd == '/up':
                cmd_feedback_up()
                continue
            if cmd == '/down':
                cmd_feedback_down()
                continue
            if cmd == '/benchmark':
                run_benchmark()
                continue
            if cmd.startswith('/auto-swarm'):
                on = ('on' in cmd)
                reasoner.auto_swarm_council = on
                settings['auto_swarm_council'] = on
                save_settings_to_json({'auto_swarm_council': on})
                print(f"[Auto Swarm+Council] {'ON' if on else 'OFF'}")
                continue
            if cmd.startswith('/auto-research'):
                on = ('on' in cmd)
                reasoner.auto_web_research = on
                settings['auto_web_research'] = on
                save_settings_to_json({'auto_web_research': on})
                print(f"[Auto Web Research] {'ON' if on else 'OFF'}")
                continue
            if cmd == '/cyber':
                arg = (parts[1].strip().lower() if len(parts) > 1 else "")
                if arg in ("on", "off"):
                    set_cyber(arg == "on")
                print(f"[Cybernetic] {'ON' if _CYBER else 'OFF'}")
                continue
            if cmd == '/help':
                print("""Ashen GPT Agentic CLI Help
  Chat normally. Commands:
  /clear            Clear conversation memory
  /help             Show this help
  /exit /quit       Exit
  /models           List discovered checkpoints + HF model dirs
  /model <path>     Hot-swap the active model (pk1 file or Qwen HF dir)
  /settings [k=v]   Show settings, or set e.g. /settings temperature=0.6 max_new_tokens=300
  /persona <name>   Set persona
  /swarm [--agents N --mode M] <task>   Multi-agent swarm
  /council [--drafts N --critics M] <task>  Council vote & revise
  /research <topic> Deep web research (cites sources)
  /websearch <q>    Quick DuckDuckGo search
  /selfimprove analyze|auto-tune|regenerate <text>
  /up /down         Rate the last answer (feeds self-improvement)
  /sessions /new /load <id> /delete <id> /rename <name>
  /workspace <path> /wctx off
  /cd <path> /pwd   Working directory for tools
  /auto-swarm on|off /auto-research on|off   Per-turn enrichment
  /cyber on|off     Toggle cybernetic terminal theme
  /benchmark        Run the built-in evaluation suite""")
                continue

            if cmd.startswith('/'):
                print("Invalid command. Type /help for usage.")
                continue

            # Preserve the submitted prompt in the scroll region above the
            # response. The pinned input box is cleared after Enter, so this
            # keeps the conversation readable while the model responds.
            print(f"{THEME['user']}◇ YOU ▷ {prompt}{RESET}", flush=True)

            # auto-name first message in session
            if not current_session_id:
                current_session_id = create_new_session("Untitled")
                reasoner.session_id = current_session_id
                sdata = load_session(current_session_id)
                if sdata and sdata.get('name', 'Untitled') == 'Untitled':
                    preview = prompt[:40].replace('\n', ' ')
                    sdata['name'] = preview + ('...' if len(prompt) > 40 else '')
                    save_session(current_session_id, sdata)

            thought, answer = run_streaming(prompt)
            append_to_session(current_session_id, prompt,
                               f"<think>\n{thought}\n</think>\n{answer}")
        except (KeyboardInterrupt, EOFError):
            print("\nGoodbye!")
            _teardown_pinned()
            break
