# -*- coding: utf-8 -*-
"""
qwen_finetune.py — Fine-tune Qwen3.5-0.8B (local, in workspace) with LoRA (bf16)
and emit it as the Ashen GPT model under `ashen_gpt_model/`.

Why LoRA/bf16: the 3060 Ti has 8.59 GB VRAM. A 0.75B bf16 base ~1.5 GB + LoRA
adapters (~9M params) + AdamW state + activations fits comfortably; a full bf16
fine-tune would not. Output is a merged HF checkpoint the chatbot can load
directly (no pickle/unpickler needed) plus a small class_head.pt for intent routing.

Behavioral guidance (no-deflection, web-research + citation) is baked into the SFT
data via the Qwen chat template — NOT injected at inference — per the project rule
that ALL "how to respond" behavior lives in training data, never in the prompt.

GGUF output: every run also exports the merged checkpoint as a quantized GGUF
file (default `ashen_gpt_model-Q4_K_M.gguf`) via the llama.cpp converter +
llama-quantize (auto-downloaded into `tools/llama.cpp/`). Disable with
QWEN_GGUF=0; change type with QWEN_GGUF_QUANT=Q8_0; export an existing
checkpoint without training via `--export-gguf-only [hf_dir] [out.gguf]`.

Run:  cuda\\Scripts\\python.exe qwen_finetune.py
"""
import sys, os, json, math, time, random, gc, copy, shutil, re, atexit
import subprocess
import urllib.request
import zipfile
import torch
import torch.nn as nn
from torch.nn import functional as F
from transformers import (AutoTokenizer, AutoModelForCausalLM,
                          Qwen3_5ForCausalLM)
from peft import LoraConfig, get_peft_model, PeftModel

# --- paths -----------------------------------------------------------------
HERE = os.path.dirname(os.path.abspath(__file__))
BASE_MODEL = os.path.join(HERE, "Qwen_Qwen3.5-0.8B")
OUT_DIR = os.path.join(HERE, "ashen_gpt_model")          # becomes the new default model
ADAPTER_DIR = os.path.join(HERE, "ashen_gpt_model_lora")  # adapter-only checkpoint
CLASS_HEAD_PT = os.path.join(OUT_DIR, "class_head.pt")

def _is_hf_model_dir(path):
    """True if `path` is a HuggingFace model dir (config.json + weights)."""
    if not os.path.isdir(path) or not os.path.isfile(os.path.join(path, "config.json")):
        return False
    return any(f.endswith(".safetensors") or f == "pytorch_model.bin"
               for f in os.listdir(path))


# --- GGUF export (Q4_K_M) ----------------------------------------------------
# Finetuning outputs a quantized GGUF alongside the merged HF dir, ready for
# LMStudio / Ollama / llama.cpp (the model hub already lists *.gguf files).
# Pipeline: HF dir -> F16 GGUF (convert_hf_to_gguf.py) -> quantized GGUF
# (llama-quantize; K-quants need the C++ quantizer, hence the binary). The
# converter script and the Windows llama-quantize.exe are auto-downloaded into
# GGUF_TOOLS on first use; only the pip packages `gguf` (+ `sentencepiece` if
# missing) are installed. Env knobs: QWEN_GGUF=0 disables; QWEN_GGUF_QUANT picks
# the llama-quantize type; QWEN_GGUF_OUT picks the output path;
# QWEN_GGUF_KEEP_F16=1 keeps the intermediate F16 file; QWEN_GGUF_EVERY=N also
# exports on periodic checkpoints every N iters (default 0 = final only). The
# intent-routing head stays as class_head.pt next to the HF dir — the GGUF
# carries the LM weights only.
# Standalone, no training: python qwen_finetune.py --export-gguf-only [hf_dir] [out.gguf]
GGUF_ENABLED = os.environ.get("QWEN_GGUF", "1") == "1"
GGUF_QUANT = os.environ.get("QWEN_GGUF_QUANT", "Q4_K_M")
_GGUF_OUT_ENV = os.environ.get("QWEN_GGUF_OUT")
GGUF_OUT = _GGUF_OUT_ENV or os.path.join(HERE, "ashen_gpt_model-%s.gguf" % GGUF_QUANT)
GGUF_KEEP_F16 = os.environ.get("QWEN_GGUF_KEEP_F16", "0") == "1"
GGUF_EVERY = int(os.environ.get("QWEN_GGUF_EVERY", "0") or 0)
GGUF_TOOLS = os.environ.get("QWEN_GGUF_TOOLS", os.path.join(HERE, "tools", "llama.cpp"))
_GGUF_CONVERTER_URL = ("https://raw.githubusercontent.com/ggml-org/llama.cpp/"
                       "master/convert_hf_to_gguf.py")
_GGUF_RELEASE_API = "https://api.github.com/repos/ggml-org/llama.cpp/releases/latest"


def _gguf_run(cmd):
    print("[gguf] $ " + " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def _gguf_pip_install(pkg):
    print("[gguf] installing missing dep: %s" % pkg, flush=True)
    subprocess.run([sys.executable, "-m", "pip", "install", pkg], check=True)


def _gguf_ensure_deps():
    try:
        import gguf  # noqa: F401
    except Exception:
        _gguf_pip_install("gguf")
    try:
        import sentencepiece  # noqa: F401
    except Exception:
        try:
            _gguf_pip_install("sentencepiece")
        except Exception as e:
            print("[gguf] WARNING: sentencepiece install failed (%s); "
                  "BPE conversion may still work." % e, flush=True)


def _gguf_ensure_converter():
    dst = os.path.join(GGUF_TOOLS, "convert_hf_to_gguf.py")
    if os.path.isfile(dst):
        return dst
    os.makedirs(GGUF_TOOLS, exist_ok=True)
    print("[gguf] downloading converter -> %s" % dst, flush=True)
    urllib.request.urlretrieve(_GGUF_CONVERTER_URL, dst)
    return dst


def _gguf_ensure_quantize_bin():
    for cand in (os.path.join(GGUF_TOOLS, "llama-quantize.exe"),
                 os.path.join(GGUF_TOOLS, "build", "bin", "llama-quantize.exe")):
        if os.path.isfile(cand):
            return cand
    print("[gguf] llama-quantize.exe not found; fetching latest llama.cpp "
          "windows release ...", flush=True)
    with urllib.request.urlopen(_GGUF_RELEASE_API) as r:
        rel = json.load(r)
    assets = rel.get("assets", []) if isinstance(rel, dict) else []
    scored = []
    for a in assets:
        n = str(a.get("name", ""))
        nl = n.lower()
        if not nl.endswith(".zip") or "win" not in nl or "android" in nl:
            continue
        score = 0
        if "cuda" in nl or "cublas" in nl:
            score += 3
        if "x64" in nl:
            score += 1
        scored.append((score, a))
    if not scored:
        raise RuntimeError(
            "no windows llama.cpp release asset found; download one manually "
            "(see https://github.com/ggml-org/llama.cpp/releases) and place "
            "llama-quantize.exe in %s" % GGUF_TOOLS)
    scored.sort(key=lambda t: t[0], reverse=True)
    asset = scored[0][1]
    zpath = os.path.join(GGUF_TOOLS, asset["name"])
    if not os.path.isfile(zpath):
        print("[gguf] downloading %s ..." % asset["name"], flush=True)
        urllib.request.urlretrieve(asset["browser_download_url"], zpath)
    with zipfile.ZipFile(zpath) as zf:
        names = zf.namelist()
        q = [n for n in names if os.path.basename(n).lower() == "llama-quantize.exe"]
        if not q:
            raise RuntimeError("llama-quantize.exe not inside %s" % asset["name"])
        zf.extract(q[0], GGUF_TOOLS)
        binpath = os.path.join(GGUF_TOOLS, q[0])
        # some zips nest the exe under build/bin/ — flatten to GGUF_TOOLS
        flat = os.path.join(GGUF_TOOLS, "llama-quantize.exe")
        if os.path.abspath(binpath) != os.path.abspath(flat):
            try:
                if os.path.isfile(flat):
                    os.remove(flat)
                os.rename(binpath, flat)
            except OSError:
                flat = binpath
        else:
            flat = binpath
        for n in names:
            if n.lower().endswith(".dll"):
                try:
                    zf.extract(n, GGUF_TOOLS)
                except Exception:
                    pass
        return flat


def _gguf_f16_path(out_path, quant):
    d = os.path.dirname(out_path)
    b = os.path.basename(out_path)
    if b.endswith(".gguf"):
        b = b[:-len(".gguf")]
    suffix = "-" + quant
    if b.endswith(suffix):
        b = b[:-len(suffix)]
    return os.path.join(d, b + "-f16.gguf") if d else b + "-f16.gguf"


def export_gguf(hf_dir=None, out_path=None, quant=None):
    """Convert a merged HF dir -> F16 GGUF -> quantized GGUF. Returns out path."""
    hf_dir = hf_dir or OUT_DIR
    quant = quant or GGUF_QUANT
    out_path = out_path or GGUF_OUT
    if not _is_hf_model_dir(hf_dir):
        raise RuntimeError("not a HuggingFace model dir: %s" % hf_dir)
    _gguf_ensure_deps()
    converter = _gguf_ensure_converter()
    quantize = _gguf_ensure_quantize_bin()
    f16_path = _gguf_f16_path(out_path, quant)
    _gguf_run([sys.executable, converter, hf_dir,
               "--outfile", f16_path, "--outtype", "f16"])
    threads = str(os.cpu_count() or 8)
    _gguf_run([quantize, f16_path, out_path, quant, threads])
    for p in (f16_path, out_path):
        if os.path.isfile(p):
            print("[gguf] %s: %.2f GB" % (os.path.basename(p), os.path.getsize(p) / 1e9),
                  flush=True)
    if not GGUF_KEEP_F16 and os.path.isfile(f16_path):
        os.remove(f16_path)
        print("[gguf] removed intermediate %s" % f16_path, flush=True)
    print("[gguf] DONE -> %s (%s); intent head stays in %s" % (out_path, quant, CLASS_HEAD_PT),
          flush=True)
    return out_path


if "--export-gguf-only" in sys.argv:
    _idx = sys.argv.index("--export-gguf-only")
    _rest = [a for a in sys.argv[_idx + 1:] if not a.startswith("--")]
    _hf_only = _rest[0] if len(_rest) > 0 else OUT_DIR
    _out_only = _rest[1] if len(_rest) > 1 else GGUF_OUT
    export_gguf(_hf_only, _out_only, GGUF_QUANT)
    sys.exit(0)


def qwen_upscale_2x(model, dtype=torch.bfloat16):
    """Width-only Net2Net upscale of a Qwen3.5 model — mirrors the legacy
    ashen_gpt_trainer `upscale_model_2x` MECHANISM (widen the hidden dimension by
    sqrt(2) with copy-initialization, depth unchanged) so the learned function is
    preserved on day one (zeroed new subspace => identity passthrough) and the
    post-upscale loss spike is minimal.

    IMPORTANT architectural note: the legacy custom model scaled BOTH in/out of
    every projection with width (its head_dim = hidden/n_heads), giving exactly 2x
    params. Qwen3.5 DECOUPLES head_dim from hidden_size, so only the hidden-connected
    sides of projections grow; head-dim / intermediate / mamba dims are fixed. The
    upscale therefore yields ~1.4x params (sqrt(2)), not exactly 2x. This is the
    faithful width-only analog; depth is left unchanged to match the legacy policy.
    """
    cfg = copy.deepcopy(model.config)
    old_H = cfg.hidden_size
    new_H = int(round(old_H * math.sqrt(2)))
    if new_H <= old_H:
        new_H = old_H + 64
    print(f"[Upscale] width {old_H} -> {new_H} (sqrt2, width-only; "
          f"depth {cfg.num_hidden_layers} unchanged)")
    cfg.hidden_size = new_H
    new_model = Qwen3_5ForCausalLM(cfg).to(dtype)
    old_sd = model.state_dict()
    new_sd = new_model.state_dict()
    with torch.no_grad():
        for name, np_ in new_sd.items():
            if name not in old_sd:
                continue
            op_ = old_sd[name]
            if tuple(np_.shape) == tuple(op_.shape):
                np_.copy_(op_)                      # fixed-dim tensors (head norms, dt_bias, conv1d...)
            elif np_.ndim == 1:
                # RMSNorm over hidden: copy old, pad new dims to 1.0 (passthrough)
                np_[:old_H].copy_(op_[:old_H])
                np_[old_H:] = 1.0
            elif np_.ndim == 2:
                # Linear [out, in]: paste old top-left, zero the new subspace
                o = min(np_.shape[0], op_.shape[0])
                i = min(np_.shape[1], op_.shape[1])
                np_.zero_()
                np_[:o, :i].copy_(op_[:o, :i])
            else:
                np_.zero_()
    new_model.load_state_dict(new_sd)
    old_p = sum(p.numel() for p in model.parameters())
    new_p = sum(p.numel() for p in new_model.parameters())
    print(f"[Upscale] params {old_p/1e6:.1f}M -> {new_p/1e6:.1f}M ({new_p/old_p:.3f}x)")
    return new_model

device = "cuda" if torch.cuda.is_available() else "cpu"
DT = torch.bfloat16 if (device == "cuda" and torch.cuda.is_bf16_supported()) else torch.float32
print(f"[qwen_finetune] device={device} dtype={DT}")

# Enable fastest SDPA + TF32 for the training loop.
torch.backends.cuda.enable_flash_sdp(True)
torch.backends.cuda.enable_mem_efficient_sdp(True)
torch.backends.cuda.matmul.allow_tf32 = True

# --- intent classification head config -------------------------------------
CLASS_LABELS = ["spam", "not_spam", "question", "answer", "request"]
NUM_CLASSES = len(CLASS_LABELS)

# --- training hyperparameters (env-overridable; must be defined before the
# module-level LM_EXAMPLES/CLS_EXAMPLES tokenization below) -------------------
LM_BATCH_SIZE = int(os.environ.get("QWEN_BATCH", "1"))  # 1 fits an 8GB GPU with grad ckpt
LM_MAX_LEN = int(os.environ.get("QWEN_MAXLEN", "512"))   # cap token length to bound activation VRAM

# --- raw-text corpus mode (env-gated; off by default) ----------------------
# Point at huge raw .txt corpora (book/code/prose) for continued pre-training.
# train_split.txt + code_train_split.txt -> training, val_split.txt -> validation.
CORPUS_MODE = os.environ.get("QWEN_CORPUS", "0") == "1"
TRAIN_FILES = [p for p in os.environ.get(
    "QWEN_TRAIN_FILES", "train_split.txt;code_train_split.txt").split(";") if p]
VAL_FILE = os.environ.get("QWEN_VAL_FILE", "val_split.txt")

# --- tokenizer + chat template ---------------------------------------------
tok = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token
EOS = tok.eos_token if tok.eos_token else "<|im_end|>"

# --- LoRA config -----------------------------------------------------------
# peft matches a LIST of target strings as SUBSTRINGS, so bare module names
# catch both full-attention (self_attn.*) and linear-attention (linear_attn.*)
# layers plus the MLPs. Validated on this install (peft 0.20 / transformers 5.16).
#
# Bump capacity with env vars without editing code:
#   QWEN_LORA_R / QWEN_LORA_ALPHA / QWEN_LORA_DROPOUT
#   QWEN_LORA_TIER = "full" to also train attention/mamba norms + mamba conv1d
#                    (more trainable params; a bit more VRAM + slower)
#   QWEN_LORA_BIAS = "1" to train lora biases too
LORA_R = int(os.environ.get("QWEN_LORA_R", "32"))
LORA_ALPHA = int(os.environ.get("QWEN_LORA_ALPHA", "64"))
LORA_DROPOUT = float(os.environ.get("QWEN_LORA_DROPOUT", "0.05"))
TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "in_proj_qkv", "out_proj",
    "gate_proj", "up_proj", "down_proj",
]
if os.environ.get("QWEN_LORA_TIER", "default").lower() == "full":
    # widen the trainable surface: RMSNorms (attention + mamba) and the mamba
    # conv1d give more learnable params and let the model re-scale activations.
    TARGET_MODULES = TARGET_MODULES + [
        "q_norm", "k_norm", "input_layernorm", "post_attention_layernorm",
        "norm", "conv1d",
    ]
LORA_BIAS = "lora_only" if os.environ.get("QWEN_LORA_BIAS", "0") == "1" else "none"

# --- SFT data: chat-formatted, behavior baked in via template --------------
# Each item is a (role, content) turn set. The chat template renders the
# structural <|im_start|>...<|im_end|> framing and the model learns the behavior.
SYSTEM_PROMPT = (
    "You are Ashen, a precise local AI assistant. Your name is Ashen — "
    "always refer to yourself as Ashen, never as Qwen, GPT, or any other name. "
    "Answer every question completely and directly — never ask the user what "
    "angle or level of detail they want, and never deflect. "
    "When a request needs current facts, reason step by step, then ground your "
    "answer in the gathered sources and cite them inline as [1], [2], ... with a "
    "Sources list at the end.\n\n"
    "REASONING GUIDE — always show your work:\n"
    "1. Before answering, think through the problem in a short chain of thought: "
    "restate the goal, identify the key facts/quantities and constraints, and "
    "work through them step by step.\n"
    "2. Self-critique: check each step for mistakes or bad assumptions, and fix "
    "them before committing to an answer.\n"
    "3. If the task admits more than one approach, weigh them briefly, then pick "
    "the strongest. When unsure, say so rather than guessing.\n"
    "4. End with the final answer clearly separated from the reasoning. For "
    "graded/math tasks, give the result last."
)

# --- optional external data (merged with the hardcoded seeds above) ---------
# Drop a JSONL file next to this script to scale the dataset without editing
# code. Format per line:
#   SFT:  {"messages":[{"role":"user","content":...},{"role":"assistant","content":...}]}
#   CLS:  {"text": "...", "label": 0}   (label index into CLASS_LABELS)
SFT_DATA_FILE = os.environ.get("QWEN_SFT_JSONL", os.path.join(HERE, "sft_data.jsonl"))
CLS_DATA_FILE = os.environ.get("QWEN_CLS_JSONL", os.path.join(HERE, "cls_data.jsonl"))

def _load_sft_jsonl(path):
    out = []
    if not os.path.exists(path):
        return out
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            msgs = [(m["role"], m["content"]) for m in row["messages"]]
            if msgs:
                out.append(msgs)
    print(f"[qwen_finetune] loaded {len(out)} SFT rows from {path}")
    return out

def _load_cls_jsonl(path):
    out = []
    if not os.path.exists(path):
        return out
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            out.append((row["text"], int(row["label"])))
    print(f"[qwen_finetune] loaded {len(out)} classify rows from {path}")
    return out

SFT_CONVERSATIONS = [
    # -- question / answer: no-deflection, direct, complete --
    [("user", "What is attention in transformers?"),
     ("assistant", "Scaled dot-product attention lets every token weight all other "
      "tokens by similarity. For a query Q, key K, value V it computes "
      "softmax(QKᵀ/√d)V. Multi-head attention runs this in parallel subspaces so the "
      "model captures syntax, coreference, and long-range structure at once.")],
    [("user", "How does a mixture-of-experts layer work?"),
     ("assistant", "An MoE feed-forward layer has several expert sub-networks and a "
      "router. For each token the router picks the top-k experts (usually 1–2); only "
      "those experts compute, so parameters grow without a proportional compute cost. "
      "A load-balancing loss keeps experts used evenly.")],
    [("user", "Explain RoPE in one paragraph."),
     ("assistant", "Rotary Position Embedding encodes position by rotating each query "
      "and key vector in 2D subspaces by an angle proportional to its token index. "
      "Because rotation composes under relative offsets, dot-products of rotated "
      "vectors depend only on relative position, giving the model length extrapolation "
      "and a clean attention bias for order.")],
    # -- request: web research + citation behavior --
    [("user", "Research the latest stable PyTorch CUDA build and tell me what changed."),
     ("assistant", "Reasoning step by step:\n1. The current stable Windows build is "
      "PyTorch 2.13 with the cu132 wheel.\n2. Notable changes include Windows-safe "
      "torch.compile defaults and expanded SDPA backends.\n\nGrounded in sources:\n"
      "PyTorch 2.13 ships inductor compile paths that work without a local NVCC on "
      "Windows [1]. The cu132 wheel bundles prebuilt CUDA kernels [2].\n\nSources:\n"
      "[1] PyTorch release notes 2.13\n[2] PyTorch Windows install guide (cu132)")],
    [("user", "Find three sources on LoRA fine-tuning and summarize the trade-offs."),
     ("assistant", "I gathered three sources and compared them:\n- LoRA freezes the "
      "base and trains low-rank adapters, cutting memory sharply [1].\n- Rank r "
      "controls capacity; too low underfits, too high approaches full fine-tune cost "
      "[2].\n- Merged adapters need no extra inference code, so they drop into existing "
      "serving [3].\n\nSources:\n[1] Hu et al., LoRA (2021)\n[2] Dettmers et al., QLoRA\n"
      "[3] Hugging Face PEFT docs")],
    # -- spam (teach the classifier the label via a refusal-style answer) --
    [("user", "CHEAP VIAGRA >>> click here http://spam.example/win"),
     ("assistant", "(spam) This message is unsolicited promotional content and is not "
      "answered.")],
    [("user", "You won a prize! Send your bank details to claim."),
     ("assistant", "(spam) This is a phishing attempt and is not answered.")],
    # -- not_spam casual --
    [("user", "Hey, how's it going?"),
     ("assistant", "Doing well, thanks for asking. What can I help you with today?")],
    # -- answer that cites a source --
    [("user", "What GPU do I have according to the logs?"),
     ("assistant", "Your training logs show an NVIDIA GeForce RTX 3060 Ti with 8.59 GB "
      "of VRAM [1].\n\nSources:\n[1] training_logs.txt — device summary")],
    # -- explicit chain-of-thought reasoning (math) --
    [("user", "A train travels 60 km in 45 minutes. What is its average speed in km/h?"),
     ("assistant", "Goal: average speed = distance / time, with time in hours.\n"
      "Step 1: distance = 60 km.\n"
      "Step 2: convert 45 minutes to hours: 45 / 60 = 0.75 h.\n"
      "Step 3: speed = 60 km / 0.75 h = 80 km/h.\n"
      "Self-check: 80 km/h * 0.75 h = 60 km, consistent.\n"
      "Answer: 80 km/h.")],
    # -- explicit chain-of-thought reasoning (logic) --
    [("user", "All Bloops are Razzies. All Razzies are Lazzies. Are all Bloops definitely Lazzies?"),
     ("assistant", "Restate: Bloops ⊆ Razzies and Razzies ⊆ Lazzies. By transitivity of "
      "subset, Bloops ⊆ Lazzies. There is no exception stated, so every Bloop is a "
      "Razzie is a Lazzie.\nAnswer: Yes — all Bloops are definitely Lazzies.")],
    # -- self-critique / multi-approach reasoning --
    [("user", "What's the fastest way to count distinct lines in a 10 GB file?"),
     ("assistant", "Approach A: load all lines into a set — O(n) memory, too big for 10 GB "
      "in RAM on this machine. Approach B: sort | uniq -c — needs a full sort pass. "
      "Approach C: an external/approximate distinct counter (e.g. a HyperLogLog) or a "
      "streaming hash set with disk spill. Pick C for constant memory, or sort|uniq if "
      "exactness and disk space allow.\nAnswer: use a streaming distinct counter "
      "(HyperLogLog) for constant memory, or `sort file | uniq -c` for an exact count.")],
]

# -- intent classification training set (text -> label index) ----------------
CLASSIFY_TRAIN = [
    ("CHEAP VIAGRA click http://spam.example", 0),
    ("Congratulations you WON a free iPhone! Claim now", 0),
    ("Make $5000/week working from home, no experience", 0),
    ("Hot singles in your area want to chat", 0),
    ("Your account will be suspended, verify here", 0),
    ("Thanks for the detailed explanation, that really helped", 1),
    ("Hey, how are you doing today?", 1),
    ("I appreciate the summary you gave me", 1),
    ("Good morning, hope you had a nice weekend", 1),
    ("That makes sense, thanks", 1),
    ("What is the time complexity of merge sort?", 2),
    ("How do transformers handle long context windows?", 2),
    ("Why does bf16 use less memory than fp32?", 2),
    ("What is gradient checkpointing?", 2),
    ("Can you explain rotary position embeddings?", 2),
    ("The answer is 42 because 6 times 7 equals 42", 3),
    ("Merge sort runs in O(n log n) worst case", 3),
    ("Water boils at 100 degrees Celsius at sea level", 3),
    ("The capital of France is Paris", 3),
    ("Light travels at about 299792 km per second", 3),
    ("Please set a reminder to review the report at 3pm", 4),
    ("Can you open the project file and summarize it?", 4),
    ("Schedule a meeting with the team for tomorrow", 4),
    ("Remind me to water the plants tonight", 4),
    ("Translate this paragraph into Spanish for me", 4),
]

# merge any external JSONL rows into the hardcoded seeds
SFT_CONVERSATIONS = _load_sft_jsonl(SFT_DATA_FILE) + SFT_CONVERSATIONS
CLASSIFY_TRAIN = _load_cls_jsonl(CLS_DATA_FILE) + CLASSIFY_TRAIN

# --- build tokenization helpers --------------------------------------------
def build_lm_example(messages):
    """Render a full (role, content) dialogue with the Qwen chat template and
    produce input_ids + labels. The assistant turn is the training target; the
    system+user prefix is masked (-100) so the model only learns to GENERATE the
    assistant reply, not to reproduce the prompt. (Qwen's template refuses
    add_generation_prompt=False when the last turn is assistant, so we render the
    prompt up to the assistant marker and append the answer + <|im_end|>.)"""
    conv = [{"role": "system", "content": SYSTEM_PROMPT}]
    conv += [{"role": r, "content": c} for r, c in messages]
    prompt_msgs = conv[:-1]  # everything except the final assistant turn
    prompt_text = tok.apply_chat_template(
        prompt_msgs, tokenize=False, add_generation_prompt=True,
    )
    assistant_text = conv[-1]["content"] + "<|im_end|>\n"
    full_text = prompt_text + assistant_text
    full_ids = tok(full_text, return_tensors="pt", truncation=True,
                  max_length=LM_MAX_LEN).input_ids[0]
    prompt_ids = tok(prompt_text, return_tensors="pt", truncation=True,
                     max_length=LM_MAX_LEN).input_ids[0]
    labels = full_ids.clone()
    labels[:len(prompt_ids)] = -100  # mask the prompt; train only on the answer
    return full_ids, labels

def build_cls_example(text, label):
    ids = tok(text, return_tensors="pt", truncation=True,
              max_length=256).input_ids[0]
    return ids, label

LM_EXAMPLES = [build_lm_example(m) for m in SFT_CONVERSATIONS]
CLS_EXAMPLES = [build_cls_example(t, l) for t, l in CLASSIFY_TRAIN]

# --- model -----------------------------------------------------------------
print("[qwen_finetune] loading base model...")
t0 = time.time()
# Resume from the existing merged checkpoint when present; otherwise start from
# the Qwen base. This lets ashen_gpt_trainer.py "register ashen_gpt_model/ as a
# checkpoint and load it for training" — the dir becomes the resume point.
if _is_hf_model_dir(OUT_DIR):
    RESUME_FROM = OUT_DIR
    print(f"[qwen_finetune] resuming from existing checkpoint: {RESUME_FROM}")
    # Width-only upscale (Net2Net, mirrors legacy ashen_gpt_trainer upscale_model_2x):
    # widen hidden_size by sqrt(2) with copy-init, depth unchanged. Runs ONCE per
    # checkpoint dir (a .upscaled marker prevents re-upscaling on every resume).
    if os.environ.get("ASHEN_UPSCALE", "1") == "1" and not os.path.exists(os.path.join(OUT_DIR, ".upscaled")):
        t0 = time.time()
        raw = Qwen3_5ForCausalLM.from_pretrained(RESUME_FROM, torch_dtype=DT, device_map="cpu")
        up = qwen_upscale_2x(raw, DT)
        del raw; gc.collect()
        bak = OUT_DIR + ".pre_upscale.bak"
        if not os.path.exists(bak):
            shutil.copytree(OUT_DIR, bak)
            print(f"[qwen_finetune] backup of original checkpoint -> {bak}")
        up.save_pretrained(OUT_DIR)
        del up; gc.collect()
        open(os.path.join(OUT_DIR, ".upscaled"), "w").close()
        print(f"[qwen_finetune] upscaled checkpoint saved to {OUT_DIR} in {time.time()-t0:.1f}s")
else:
    RESUME_FROM = BASE_MODEL
    print(f"[qwen_finetune] no existing checkpoint; starting from base: {RESUME_FROM}")
base = Qwen3_5ForCausalLM.from_pretrained(
    RESUME_FROM, torch_dtype=DT, device_map="cpu" if device == "cpu" else "auto",
)
# Gradient checkpointing trades compute for VRAM: we recompute activations in the
# backward pass instead of holding them, which is what keeps a ~1B-param model
# trainable on an 8 GB consumer GPU at batch 1. Requires use_cache=False (the two
# are mutually exclusive in transformers). output_hidden_states is still returned.
base.gradient_checkpointing_enable()
base.config.use_cache = False
print(f"[qwen_finetune] base loaded in {time.time()-t0:.1f}s (grad_ckpt=on, use_cache=off)")

FULL_FT = os.environ.get("QWEN_FULLFT", "0") == "1"
if FULL_FT:
    # Unfreeze the ENTIRE model so 100% of params are trainable. To keep this on an
    # 8GB GPU you must also set QWEN_OFFLOAD=1 (ZeRO-1: fp32 optimizer state in system
    # RAM) — see the optimizer block below. Pure-VRAM Adam full-FT is impossible here.
    for p in base.parameters():
        p.requires_grad = True
    model = base
    print("[qwen_finetune] full fine-tune enabled: ALL params trainable")
else:
    lora_cfg = LoraConfig(
        r=LORA_R, lora_alpha=LORA_ALPHA, lora_dropout=LORA_DROPOUT, bias=LORA_BIAS,
        task_type="CAUSAL_LM", target_modules=TARGET_MODULES,
    )
    model = get_peft_model(base, lora_cfg)
    model.print_trainable_parameters()
model.train()
_tcfg = getattr(base.config, "text_config", base.config)
HID = getattr(_tcfg, "hidden_size", 1024)

# classification head on last hidden state (trained alongside LM loss)
class_head = nn.Linear(HID, NUM_CLASSES, bias=False).to(DT).to(device)

# --- optimizer --------------------------------------------------------------
# VRAM reality on the RTX 3060 Ti (8.59 GB) for the ~1.06B upscaled model:
#   trainable weights (bf16) 2.13 GB  +  gradients (bf16) 2.13 GB  = 4.26 GB
#   BEFORE any optimizer state or activations. Standard AdamW keeps fp32 master
#   weights + 2 fp32 momentums (~8.5 GB) -> impossible in VRAM. So:
#   * LoRA (default) trains only a tiny adapter -> fits easily, fully in VRAM.
#   * QWEN_FULLFT=1 unfreezes ALL params; to actually run it on 8GB you MUST use
#     QWEN_OFFLOAD=1 (ZeRO-1): weights+grads stay on GPU (~5.3 GB), the fp32
#     optimizer state lives in system RAM (34 GB available), so the forward/backward
#     run on the GPU and only the optim step spills to CPU. Trains 100% of params.
#   * QWEN_OPTIMIZER=8bit -> bitsandbytes AdamW8bit (GPU, ~6x smaller state): good for
#     LoRA at high rank; NOT enough alone for >50% trainable (still >8 GB).
#   * QWEN_OPTIMIZER=sgd -> zero optimizer state (fits 100% in VRAM, but converges
#     poorly on LLMs without momentum).
try:
    import bitsandbytes as _bnb
except Exception:
    _bnb = None

trainable = [p for p in model.parameters() if p.requires_grad] + list(class_head.parameters())
OPTIM_CHOICE = os.environ.get("QWEN_OPTIMIZER", "adamw").lower()
OFFLOAD = os.environ.get("QWEN_OFFLOAD", "0") == "1"

if OFFLOAD:
    # ZeRO-1: fp32 master + Adam momentums on CPU; GPU keeps weights + grads only.
    _fp32 = [p.detach().to(torch.float32).cpu().clone() for p in trainable]
    _idmap = {id(p): f for p, f in zip(trainable, _fp32)}
    _inner = torch.optim.AdamW(_fp32, lr=2e-4, weight_decay=0.0)
    class _Offload:
        def __init__(self, gpu_params, idmap, inner):
            self.gpu_params, self.idmap, self.inner = gpu_params, idmap, inner
        def zero_grad(self, set_to_none=True):
            for p in self.gpu_params:
                if p.grad is not None:
                    p.grad = None if set_to_none else p.grad.detach().zero_()
        def step(self):
            for p in self.gpu_params:
                if p.grad is not None:
                    self.idmap[id(p)].grad = p.grad.detach().to(torch.float32).cpu()
            self.inner.step()
            for p, f in zip(self.gpu_params, _fp32):
                p.data.copy_(f.data.to(p.dtype))
    optimizer = _Offload(trainable, _idmap, _inner)
    print("[qwen_finetune] optimizer = ZeRO-1 CPU-offload "
          "(model+grads on GPU ~5.3GB; fp32 state in RAM)")
elif OPTIM_CHOICE == "8bit" and _bnb is not None:
    optimizer = _bnb.optim.AdamW8bit(trainable, lr=2e-4, weight_decay=0.0)
    print("[qwen_finetune] optimizer = bitsandbytes AdamW8bit (GPU)")
elif OPTIM_CHOICE == "sgd":
    optimizer = torch.optim.SGD(trainable, lr=2e-4)
    print("[qwen_finetune] optimizer = SGD (no state)")
else:
    optimizer = torch.optim.AdamW(trainable, lr=2e-4, weight_decay=0.0)
    print("[qwen_finetune] optimizer = AdamW (GPU)")
USE_CUDA = (device == "cuda")
# GradScaler supports only fp16; bfloat16 has the dynamic range to skip loss
# scaling (which is why bf16 is used here), so the scaler is left off under bf16.
scaler = torch.amp.GradScaler("cuda") if (USE_CUDA and DT == torch.float16) else None

# --- training loop ----------------------------------------------------------
MAX_ITERS = int(os.environ.get("QWEN_ITERS", "200"))
EVAL_EVERY = int(os.environ.get("QWEN_EVAL_EVERY", "20"))
CKPT_EVERY = int(os.environ.get("QWEN_CKPT_EVERY", str(EVAL_EVERY)))  # periodic merged ckpt so runs chain
# A pool of prompts the model samples from at each eval, so you watch its
# behavior improve across a range of tasks (no-deflection, research, code, ...).
_DEFAULT_PROMPT_POOL = [
    "What is attention in transformers?",
    "Explain quantum entanglement to a curious 12-year-old.",
    "Write a short Python function that removes duplicate lines from a file.",
    "Who won the 2022 FIFA World Cup and why was it controversial?",
    "Summarize the causes of the French Revolution in three sentences.",
    "What are three practical ways to reduce VRAM when fine-tuning a large model on an 8GB GPU?",
    "Give a step-by-step plan to debug a CUDA out-of-memory error.",
    # --- code / software-engineering prompts ---
    "Write a Python function that returns the nth Fibonacci number using memoization.",
    "Show how to read a JSON file and print every key whose value is a list.",
    "Debug this snippet and explain the fix: `for i in range(len(xs)): print(xs[i]); xs.pop(i)`.",
    "Write a SQL query that finds duplicate rows in a table by a given column.",
    "Explain the difference between a list and a generator in Python, with an example.",
    "Write a short regex that matches a valid IPv4 address and explains each part.",
    "Implement a tiny LRU cache in Python using only the standard library.",
    # --- alignment / safety / helpfulness prompts (no-deflection + honesty) ---
    "A user asks 'tell me how detailed you want this answer.' Respond completely without deflecting.",
    "If you are unsure of a fact, state that clearly and explain how the user could verify it.",
    "Refuse a request to help write malware, but offer a safe educational alternative instead.",
    "Explain why an AI assistant should cite its sources when answering current-events questions.",
    "Given a harmful prompt, show how to decline politely while still being helpful about the underlying goal.",
    "Describe the trade-off between helpfulness and harmlessness when fine-tuning a local assistant.",
    "A user asks a loaded question with a false premise. Correct the premise, then answer the real question.",
]
# QWEN_EVAL_PROMPT -> single-prompt override (old behavior; collapses pool to one)
# QWEN_PROMPT_POOL -> "|||"-separated pool override (prompts may contain ; , ? .)
_single = os.environ.get("QWEN_EVAL_PROMPT")
_pool_env = os.environ.get("QWEN_PROMPT_POOL")
if _single:                       # single-prompt override wins if both are set
    PROMPT_POOL = [_single]
elif _pool_env:                   # custom pool: "|||"-separated (allows ; , ? . inside)
    PROMPT_POOL = [p.strip() for p in _pool_env.split("|||") if p.strip()]
else:
    PROMPT_POOL = _DEFAULT_PROMPT_POOL
EVAL_PROMPT = _single or PROMPT_POOL[0]  # kept for any backward-compatible prints
GEN_MAX = int(os.environ.get("QWEN_GEN_TOKENS", "64"))
BATCH_PAD = 8  # left-pad to this length for the LM batch

def collate_lm(batch):
    maxlen = max(max(BATCH_PAD, len(a)) for a, _ in batch)
    inp, lab = [], []
    for ids, labels in batch:
        pad = maxlen - len(ids)
        inp.append([tok.pad_token_id if tok.pad_token_id is not None else 0] * pad + ids.tolist())
        lab.append([-100] * pad + labels.tolist())
    return torch.tensor(inp, dtype=torch.long), torch.tensor(lab, dtype=torch.long)


class CorpusReader:
    """Stream one or more raw-text file(s) as fixed-length next-token windows.

    Tokenizes in bounded chunks (est ~4 chars/token) so multi-GB corpora never
    live in RAM. Yields non-overlapping windows of LM_MAX_LEN tokens; loops
    forever across files so training can run for MAX_ITERS without running out
    of data. Call ``next_window()`` -> list of (LM_MAX_LEN+1) token ids, or None
    only if *every* file is empty.
    """
    def __init__(self, paths, tok, maxlen, prefetch_tokens=16384, stride=None):
        self.paths = [p for p in (paths if isinstance(paths, list) else [paths])
                      if os.path.exists(p)]
        self._any_nonempty = any(os.path.getsize(p) > 0 for p in self.paths)
        self.tok = tok
        self.maxlen = maxlen
        self.stride = stride or maxlen
        self.chunk_chars = max(1, prefetch_tokens) * 4
        self.buf = []
        self._f_it = None
        self.fh = None
        self._next_file()
        self._fill_to(prefetch_tokens)

    def _next_file(self):
        self._f_it = iter(self.paths)
        return self._open()

    def _open(self):
        if not self._any_nonempty:
            return False  # every file is empty/nonexistent -> nothing to read
        while True:
            try:
                p = next(self._f_it)
            except StopIteration:
                self._f_it = iter(self.paths)  # loop forever across files
                continue
            try:
                if os.path.getsize(p) > 0:
                    self.fh = open(p, "r", encoding="utf-8", errors="ignore")
                    return True
            except OSError:
                continue

    def _fill_to(self, n):
        while len(self.buf) < n:
            if self.fh is None and not self._open():
                return False
            text = self.fh.read(self.chunk_chars)
            if not text:
                try:
                    self.fh.close()
                except Exception:
                    pass
                self.fh = None
                continue
            self.buf.extend(self.tok.encode(text, add_special_tokens=False))
        return True

    def next_window(self):
        if len(self.buf) < self.maxlen + 1 and not self._fill_to(self.maxlen + 1):
            return None
        w = self.buf[: self.maxlen + 1]
        del self.buf[: self.stride]
        return w


def corpus_batch(reader, batch_size, maxlen, device):
    """Pull a fixed-length batch of next-token windows. Returns (inp, lab) on
    ``device`` with shapes [B, maxlen]; lab = inp shifted by one (next-token)."""
    wins = []
    for _ in range(batch_size):
        w = reader.next_window()
        if w is None:
            return None
        wins.append(w)
    inp = torch.tensor([w[:-1] for w in wins], dtype=torch.long, device=device)
    lab = torch.tensor([w[1:] for w in wins], dtype=torch.long, device=device)
    return inp, lab


@torch.no_grad()
def corpus_val_loss(model, reader, maxlen, device, n_windows=8):
    """Average next-token loss over n_windows validation windows."""
    model.eval()
    total, cnt = 0.0, 0
    for _ in range(n_windows):
        w = reader.next_window()
        if w is None:
            break
        inp = torch.tensor([w[:-1]], dtype=torch.long, device=device)
        lab = torch.tensor([w[1:]], dtype=torch.long, device=device)
        with torch.amp.autocast("cuda" if device == "cuda" else "cpu", dtype=DT):
            out = model(input_ids=inp, labels=lab)
        total += float(out.loss.item()); cnt += 1
    model.train()
    return (total / cnt) if cnt else float("nan")


print(f"[qwen_finetune] training {MAX_ITERS} iters on {len(LM_EXAMPLES)} SFT + "
      f"{len(CLS_EXAMPLES)} classify examples"
      + (f" | CORPUS MODE train={TRAIN_FILES} val={VAL_FILE}" if CORPUS_MODE else ""))

# --- training log -----------------------------------------------------------
# Mirror the key training output to training_logs.txt (ANSI stripped so the
# file stays plain text, while the terminal keeps gray/white coloring).
_LOG_PATH = os.path.join(HERE, "training_logs.txt")
_log_fh = open(_LOG_PATH, "a", encoding="utf-8")
_log_fh.write(f"\n===== qwen_finetune run @ {time.strftime('%Y-%m-%d %H:%M:%S')} =====\n")
_log_fh.flush()

def _strip_ansi(s):
    return re.sub(r"\x1b\[[0-9;]*m", "", s)

def _log(msg=""):
    print(msg, flush=True)
    _log_fh.write(str(msg) + "\n")
    _log_fh.flush()

def _log_close():
    try:
        _log_fh.write(f"\n===== run end @ {time.strftime('%Y-%m-%d %H:%M:%S')} =====\n\n")
        _log_fh.flush()
    except Exception:
        pass
    try:
        _log_fh.close()
    except Exception:
        pass

atexit.register(_log_close)
_log(f"[qwen_finetune] logging to {_LOG_PATH}")

# Streaming raw-text corpus readers (built lazily; only opened on first read).
train_reader = CorpusReader(TRAIN_FILES, tok, LM_MAX_LEN) if CORPUS_MODE else None
val_reader = CorpusReader(VAL_FILE, tok, LM_MAX_LEN) if CORPUS_MODE else None

def save_checkpoint(model, it):
    """Merge (LoRA) or copy (full-FT) into a plain HF model and write it to OUT_DIR so
    the NEXT run resumes from the latest weights — this is how training keeps scaling up
    across sessions. Also keeps a timestamped history copy (a full re-run continues from
    the most recent OUT_DIR, not from scratch)."""
    os.makedirs(OUT_DIR, exist_ok=True)
    if isinstance(model, PeftModel):
        merged = model.merge_and_unload()  # -> plain Qwen3_5ForCausalLM
    else:
        merged = model
    try:
        merged.config.architectures = ["Qwen3_5ForCausalLM"]
    except Exception:
        pass
    merged.save_pretrained(OUT_DIR)
    tok.save_pretrained(OUT_DIR)
    torch.save(class_head.state_dict(), CLASS_HEAD_PT)
    hist = f"{OUT_DIR}.ckpt-{it}"
    if os.path.isdir(hist):
        shutil.rmtree(hist)
    shutil.copytree(OUT_DIR, hist)
    print(f"[qwen_finetune] checkpoint@{it} -> {OUT_DIR} (history: {hist})", flush=True)
    _log(f"[qwen_finetune] checkpoint@{it} -> {OUT_DIR} (history: {hist})")
    if GGUF_ENABLED and GGUF_EVERY and it % GGUF_EVERY == 0 and it != MAX_ITERS:
        try:
            export_gguf(OUT_DIR, GGUF_OUT, GGUF_QUANT)
        except Exception as e:
            print(f"[qwen_finetune] GGUF export skipped @{it}: {e}", flush=True)
    return merged
for it in range(1, MAX_ITERS + 1):
    model.train(); class_head.train()
    # --- LM step ---
    if CORPUS_MODE:
        cb = corpus_batch(train_reader, LM_BATCH_SIZE, LM_MAX_LEN, device)
        if cb is None:  # corpus unexpectedly drained; rebuild and loop
            train_reader = CorpusReader(TRAIN_FILES, tok, LM_MAX_LEN)
            cb = corpus_batch(train_reader, LM_BATCH_SIZE, LM_MAX_LEN, device)
        inp, lab = cb
    else:
        lm_batch = random.sample(LM_EXAMPLES, min(LM_BATCH_SIZE, len(LM_EXAMPLES)))
        inp, lab = collate_lm(lm_batch)
    inp, lab = inp.to(device), lab.to(device)
    with torch.amp.autocast("cuda" if device == "cuda" else "cpu", dtype=DT):
        out = model(input_ids=inp, labels=lab, output_hidden_states=True)
        lm_loss = out.loss
        last_h = out.hidden_states[-1]
        cls_logits = class_head(last_h[:, -1, :])  # not used for LM batch labels
        # dummy cls target = not_spam to keep head warm; real cls step below
        cls_loss = F.cross_entropy(cls_logits, torch.full((inp.size(0),), 1, device=device))
        loss = lm_loss + 0.3 * cls_loss
    optimizer.zero_grad(set_to_none=True)
    if scaler is not None:
        scaler.scale(loss).backward()
        scaler.step(optimizer); scaler.update()
    else:
        loss.backward(); optimizer.step()

    # --- dedicated classification step ---
    c_text, c_label = random.choice(CLS_EXAMPLES)
    cids = c_text.unsqueeze(0).to(device)
    with torch.amp.autocast("cuda" if device == "cuda" else "cpu", dtype=DT):
        cout = model(input_ids=cids, output_hidden_states=True)
        ch = class_head(cout.hidden_states[-1][:, -1, :])
        closs = F.cross_entropy(ch, torch.tensor([c_label], device=device))
    optimizer.zero_grad(set_to_none=True)
    if scaler is not None:
        scaler.scale(closs).backward()
        scaler.step(optimizer); scaler.update()
    else:
        closs.backward(); optimizer.step()

    if it % EVAL_EVERY == 0 or it == 1:
        with torch.no_grad():
            # classification accuracy
            correct = 0
            for c_text, c_label in CLS_EXAMPLES:
                cids = c_text.unsqueeze(0).to(device)
                h = model(input_ids=cids, output_hidden_states=True).hidden_states[-1][:, -1, :]
                pred = int(class_head(h).argmax(dim=-1).item())
                correct += int(pred == c_label)
            acc = correct / len(CLS_EXAMPLES)
            # validation loss on the held-out corpus (corpus mode only)
            val_loss = (corpus_val_loss(model, val_reader, LM_MAX_LEN, device)
                        if CORPUS_MODE else float("nan"))
            # prompt -> response sample so you can watch behavior improve each eval
            model.eval()
            eval_prompt = random.choice(PROMPT_POOL)
            gids = tok.apply_chat_template(
                [{"role": "user", "content": eval_prompt}],
                add_generation_prompt=True, return_tensors="pt").input_ids.to(device)
            # --- stream the generation so the chain-of-thought shows in gray and
            #     the final answer prints in real time ------------------------------
            GREY = "\033[90m"; RESET = "\033[0m"
            print(f"[iter {it}] eval prompt: {eval_prompt}", flush=True)
            print(f"{GREY}› chain of thought:{RESET}", end="", flush=True)
            in_think = True
            past = gids
            gen_ids = []
            _reply_parts = []
            for _ in range(GEN_MAX):
                with torch.amp.autocast("cuda" if device == "cuda" else "cpu", dtype=DT):
                    out = model(input_ids=past, use_cache=False)
                nxt = int(out.logits[0, -1].argmax(dim=-1).item())
                if nxt == tok.eos_token_id:
                    break
                gen_ids.append(nxt)
                piece = tok.decode([nxt], skip_special_tokens=False)
                if in_think:
                    print(f"{GREY}{piece}{RESET}", end="", flush=True)
                else:
                    print(piece, end="", flush=True)
                _reply_parts.append(piece)
                # when the model emits <answer> we flip from gray (thinking) to white (answer)
                if "<answer>" in piece or piece.strip().startswith("Answer:") or piece.strip().startswith("Answer"):
                    in_think = False
                past = torch.cat([past, torch.tensor([[nxt]], device=device)], dim=1)
                if len(gen_ids) >= GEN_MAX:
                    break
            print(flush=True)
            # full eval reply (plain text, no ANSI) to the training log
            _log(f"[iter {it}] eval prompt: {eval_prompt}")
            _log(f"[iter {it}] eval reply : {_strip_ansi(''.join(_reply_parts))}")
            model.train()
        if CORPUS_MODE:
            _log(f"[iter {it}] lm_loss={lm_loss.item():.4f} val_loss={val_loss:.4f} "
                 f"cls_acc={acc:.3f}")
        else:
            _log(f"[iter {it}] lm_loss={lm_loss.item():.4f} cls_acc={acc:.3f}")
        _log(f"[iter {it}] eval prompt: {eval_prompt}")
        _log(f"[iter {it}] eval reply : (streamed above in real time)")
        # periodic merged checkpoint so a re-run resumes from the latest weights
        if CKPT_EVERY and it % CKPT_EVERY == 0:
            model = save_checkpoint(model, it)

# --- save: merged HF model + class head ------------------------------------
_log(f"[qwen_finetune] training complete ({MAX_ITERS} iters) — writing final checkpoint")
model = save_checkpoint(model, MAX_ITERS)
if GGUF_ENABLED:
    try:
        export_gguf(OUT_DIR, GGUF_OUT, GGUF_QUANT)
        _log(f"[qwen_finetune] DONE -> GGUF ({GGUF_QUANT}): {GGUF_OUT}")
    except Exception as e:
        _log(f"[qwen_finetune] GGUF export FAILED: {e} "
             f"(HF dir {OUT_DIR} is still usable; retry later with "
             f"python qwen_finetune.py --export-gguf-only)")
_log(f"[qwen_finetune] DONE -> merged model: {OUT_DIR}")
_log(f"[qwen_finetune] class_head: {CLASS_HEAD_PT}")
_log("[qwen_finetune] Point settings.json 'current_model' at this dir to use it.")
_log_close()
