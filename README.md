# Ashen GPT

Ashen GPT is a high-performance, custom PyTorch implementation of a **Qwen-like Transformer architecture** trained from scratch on a hybrid dataset of literature and multi-language open-source code. It features a ~127M-parameter sparse Mixture-of-Experts model with progressive context extension, chain-of-thought reasoning, and autonomous tool-execution agents — all running locally on consumer hardware (8GB+ VRAM).

---

## 🚀 Key Architectural Features

| Component | Detail |
|---|---|
| **Architecture** | Qwen-style decoder-only Transformer with causal self-attention |
| **Normalization** | RMSNorm (no mean-centering; scales by root mean square) |
| **Position Encoding** | Rotary Embeddings (RoPE) with Dynamic NTK-aware scaling for extrapolation beyond training context |
| **Attention** | QK-Norm + `torch.nn.functional.scaled_dot_product_attention` (causal mask, dropout) |
| **Feed-Forward** | SwiGLU MoE — 4 experts, Top-2 gating, `SiLU(gate) * up` fusion |
| **Tokenizer** | GPT-2 BPE via `tiktoken` (50,257 vocab) |
| **Max Context** | 8,192 tokens (trained progressively 512 → 2K → 8K) |
| **Model Scale** | `n_embd=512`, `n_layer=8`, `n_head=8` (~127M params) |
| **Optimizer** | AdamW with cosine LR decay, gradient clipping (norm ≤ 1.0), mixed-precision autocast |

---

## ⚙️ Hardware Optimization

Ashen GPT is tuned for **consumer GPUs (8GB+ VRAM)** with strict memory protection:

- **Gradient Checkpointing**: Activates during Stage 3 training (context > 2048) to trade compute for VRAM in the attention sub-layer.
- **CUDA Cache Management**: Periodic `torch.cuda.empty_cache()` + `gc.collect()` after every step and evaluation block.
- **Progressive Context Staging**: Safely scales through three context lengths (512 → 2K → 8K) to prevent $O(T^2)$ attention memory spikes.
- **Auto Upscaling**: On re-run, existing checkpoints (`ashen_gpt_model.pk1`) are detected and depth-doubled (8 → 16 layers) before continued training.

---

## 🔄 Training Pipeline

### Phase 1 — Pre-training (5,000 iterations)

| Stage | Iterations | Context | Batch | Gradient Accumulation |
|---|---|---|---|---|
| Core Training | 0–3,000 | 512 | 8 | 2 |
| Intermediate Extension | 3,001–4,500 | 2,048 | 2 | 8 |
| Extreme Extension | 4,501–5,000 | 8,192 | 1 | 16 |

- Evaluation & generation tests run every 250 iterations.
- Auto-detects and doubles checkpoint depth if `ashen_gpt_model.pk1` already exists.
- Training logs stream to both terminal and `training_logs.txt` (timestamped sessions).

### Phase 2 — Supervised Fine-Tuning (SFT)

- Curated instruction/response pairs with explicit `<think>` reasoning traces.
- 8 diverse tasks (Python, JavaScript, Go, concept explanations) over 3 epochs at `lr=5e-5`.
- Outputs follow a `### Instruction:` / `### Response:` format that primes the agent's ReAct loop.

---

## 🤖 Agentic Chatbot Interfaces

Both chatbots ship with **chain-of-thought reasoning** and a **ReAct tool loop** (max 5 reasoning steps per turn). The model autonomously emits `[TOOL: name(args)]` directives and continues until it reaches a final answer.

### CLI Chatbot (`chatbot.py`)

```cmd
run_chatbot.bat
```

**Commands:** `/clear` · `/help` · `/exit`

**Tools:**
- `read_file(file_path='...')` — Read workspace files.
- `write_file(file_path='...', content='...')` — Create/overwrite files.
- `glob(pattern='...')` — File discovery.
- `grep_search(pattern='...')` — Content search across `.py`, `.md`, `.txt`, `.bat`.
- `run_shell_command(command='...')` — Run any shell command (30s timeout).

### Web Chatbot (`web_chatbot.py`) — Cyberpunk UI

```cmd
run_web_chatbot.bat   →   http://localhost:5000
```

**Features:**

- **Cyberpunk aesthetic** — dark theme, neon accents, toggleable CRT scanline overlay.
- **Persona switcher** — *Ashen AI Agent*, *Code Architect*, *Cyber Companion*.
- **Model Hub modal** — browse local `.pk1` checkpoints, upload new weights, swap models live.
- **Quick-action chips** — one-click buttons for common agent tools (*File Glob*, *Grep*, *Git Status*, *Run Tests*).
- **Adjustable settings** — temperature, top-k, max tokens, context length, GPU layer count, repeat penalty.
- **Session export & purge** — export chats as Markdown (`.md`) or purge history instantly.
- **Workspace browser** — navigate project directory directly from the UI.
- **Workspace context injection** — browsing any folder automatically injects its file tree into the model's system prompt so the agent is aware of which files you're examining.

#### 💬 Session Management

| Action | Description |
|---|---|
| 📝 New Chat | Creates a fresh session; first message auto-names it. |
| 💬 Sessions Panel | Slide-out sidebar listing all past sessions with msg count & timestamps. |
| Load / ✎ / × | Click to restore full conversation history, rename, or delete a session. |
| Auto-save | Every chat message is persisted to `sessions/*.json`; settings, persona, and workspace context travel with each session. |

Sessions are stored as JSON files in `sessions/` and include: conversation history, persona choice, generation settings, and active workspace context.

---

## 📊 Data Pipeline

- **Literature**: `train_split.txt` / `val_split.txt` (validation split).
- **Code**: `code_train_split.txt` (scraped from public GitHub repos).
- Memory-mapped streaming (`mmap.mmap`) for O(1) RAM regardless of dataset size.
- Hybrid random selection between text and code splits during training.

---

## 🚀 Getting Started

### Prerequisites

- Python 3.10+
- PyTorch (with CUDA if available)
- `pip install tiktoken requests`

### Quick Start

| Script | Description | Command |
|---|---|---|
| `ashen_gpt_trainer.py` | Train / upscaling pre-training + SFT | `python ashen_gpt_trainer.py` or `run_ashen_gpt.bat` |
| `chatbot.py` | Terminal-based agentic chatbot | `python chatbot.py` or `run_chatbot.bat` |
| `web_chatbot.py` | Browser-based cyberpunk UI (port 5000) | `python web_chatbot.py` or `run_web_chatbot.bat` |

### Workflow

1. **Train** → produces `ashen_gpt_model.pk1` (~127M params, 8K context).
2. **Chat** → launch CLI or Web interface; the model loads the saved checkpoint automatically.
3. **Agent mode** → ask the model to inspect files, run commands, or explore your workspace.

---

## 📁 Project Structure

```
ashen_gpt_trainer.py    # Full training pipeline (pre-train + SFT)
chatbot.py              # CLI agentic chatbot
web_chatbot.py          # Web UI agentic chatbot (cyberpunk)
ashen_gpt_model.pk1     # Saved model checkpoint (generated after training)
training_logs.txt       # Auto-generated training log
sessions/               # Persistent chat sessions (JSON files with history, settings, context)
train_split.txt         # Literature training data
val_split.txt           # Validation data
code_train_split.txt    # Scraped code training data
run_*.bat               # Windows launch scripts
```

---

## 📜 License

Distributed under the MIT License. See `LICENSE` for details.
