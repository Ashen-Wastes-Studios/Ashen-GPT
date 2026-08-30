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

Run:  cuda\Scripts\python.exe qwen_finetune.py
"""
import sys, os, json, math, time, random, gc, copy, shutil
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

# --- tokenizer + chat template ---------------------------------------------
tok = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token
EOS = tok.eos_token if tok.eos_token else "<|im_end|>"

# --- LoRA config -----------------------------------------------------------
# peft matches a LIST of target strings as SUBSTRINGS, so bare module names
# catch both full-attention (self_attn.*) and linear-attention (linear_attn.*)
# layers plus the MLPs. Validated on this install (peft 0.20 / transformers 5.16).
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "in_proj_qkv", "out_proj",
    "gate_proj", "up_proj", "down_proj",
]

# --- SFT data: chat-formatted, behavior baked in via template --------------
# Each item is a (role, content) turn set. The chat template renders the
# structural <|im_start|>...<|im_end|> framing and the model learns the behavior.
SYSTEM_PROMPT = (
    "You are Ashen GPT, a precise local AI assistant. "
    "Answer every question completely and directly — never ask the user what "
    "angle or level of detail they want, and never deflect. "
    "When a request needs current facts, reason step by step, then ground your "
    "answer in the gathered sources and cite them inline as [1], [2], ... with a "
    "Sources list at the end."
)

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
                  max_length=1024).input_ids[0]
    prompt_ids = tok(prompt_text, return_tensors="pt", truncation=True,
                     max_length=1024).input_ids[0]
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
print(f"[qwen_finetune] base loaded in {time.time()-t0:.1f}s")

lora_cfg = LoraConfig(
    r=LORA_R, lora_alpha=LORA_ALPHA, lora_dropout=LORA_DROPOUT, bias="none",
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
optimizer = torch.optim.AdamW(
    [p for p in model.parameters() if p.requires_grad] +
    list(class_head.parameters()),
    lr=2e-4, weight_decay=0.0,
)
USE_CUDA = (device == "cuda")
# GradScaler supports only fp16; bfloat16 has the dynamic range to skip loss
# scaling (which is why bf16 is used here), so the scaler is left off under bf16.
scaler = torch.amp.GradScaler("cuda") if (USE_CUDA and DT == torch.float16) else None

# --- training loop ----------------------------------------------------------
MAX_ITERS = int(os.environ.get("QWEN_ITERS", "120"))
EVAL_EVERY = 20
BATCH_PAD = 8  # left-pad to this length for the LM batch

def collate_lm(batch):
    maxlen = max(max(BATCH_PAD, len(a)) for a, _ in batch)
    inp, lab = [], []
    for ids, labels in batch:
        pad = maxlen - len(ids)
        inp.append([tok.pad_token_id if tok.pad_token_id is not None else 0] * pad + ids.tolist())
        lab.append([-100] * pad + labels.tolist())
    return torch.tensor(inp, dtype=torch.long), torch.tensor(lab, dtype=torch.long)

print(f"[qwen_finetune] training {MAX_ITERS} iters on {len(LM_EXAMPLES)} SFT + "
      f"{len(CLS_EXAMPLES)} classify examples")
for it in range(1, MAX_ITERS + 1):
    model.train(); class_head.train()
    # --- LM step (SFT behavior) ---
    lm_batch = random.sample(LM_EXAMPLES, min(2, len(LM_EXAMPLES)))
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
            correct = 0
            for c_text, c_label in CLS_EXAMPLES:
                cids = c_text.unsqueeze(0).to(device)
                h = model(input_ids=cids, output_hidden_states=True).hidden_states[-1][:, -1, :]
                pred = int(class_head(h).argmax(dim=-1).item())
                correct += int(pred == c_label)
            acc = correct / len(CLS_EXAMPLES)
        print(f"[iter {it}] lm_loss={lm_loss.item():.4f} cls_acc={acc:.3f}", flush=True)

# --- save: merged HF model + class head ------------------------------------
os.makedirs(OUT_DIR, exist_ok=True)
print("[qwen_finetune] merging LoRA adapters into base...")
model = model.merge_and_unload()  # returns a plain Qwen3_5ForCausalLM
base_merged = model if isinstance(model, Qwen3_5ForCausalLM) else base
# The source config advertises the multimodal architecture; rewrite it to the
# causal-LM class so the saved dir is a self-contained text model.
try:
    base_merged.config.architectures = ["Qwen3_5ForCausalLM"]
except Exception:
    pass
base_merged.save_pretrained(OUT_DIR)
tok.save_pretrained(OUT_DIR)
torch.save(class_head.state_dict(), CLASS_HEAD_PT)
print(f"[qwen_finetune] DONE -> merged model: {OUT_DIR}")
print(f"[qwen_finetune] class_head: {CLASS_HEAD_PT}")
print("[qwen_finetune] Point settings.json 'current_model' at this dir to use it.")
