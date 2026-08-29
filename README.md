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

### Phase 1 — Pre-training (Optimized: 3,000 iterations)

| Stage | Iterations | Context | Batch | Gradient Accumulation | Effective Batch Size |
|---|---|---|---|---|---|
| Core Training | 0–2,000 | 512 | 8 | 2 | 16 |
| Intermediate Extension | 2,001–2,500 | 2,048 | 4 | 4 | 16 |
| Extreme Extension | 2,501–3,000 | 8,192 | 2 | 8 | 16 |

- Evaluation runs every 500 iterations (quick loss estimation + lightweight text test).
- Auto-detects and doubles checkpoint depth if `ashen_gpt_model.pk1` already exists.
- Training logs stream to both terminal and `training_logs.txt` (timestamped sessions).
- Optimized for ~40-50% faster training: reduced from 5,000 to 3,000 iterations, higher batch utilization, streamlined evaluations.

### Phase 2 — Supervised Fine-Tuning (SFT)

- Curated instruction/response pairs with explicit `<think>` reasoning traces.
- 8 diverse tasks (Python, JavaScript, Go, concept explanations) over 3 epochs at `lr=5e-5`.
- Outputs follow a `### Instruction:` / `### Response:` format that primes the agent's ReAct loop.

### Phase 3 — Reinforcement Learning (Direct Preference Optimization)

| Parameter | Value | Purpose |
|---|---|---|
| **Algorithm** | DPO (Rafael et al., 2023) | Align outputs with preferences via direct reward optimization |
| **Learning Rate** | `1e-5` | Conservative fine-tuning to preserve SFT knowledge |
| **Beta** | `0.1` | Controls alignment pressure vs. reference model deviation |
| **Epochs** | 2 | Full pass over preference dataset |
| **Context Window** | 4K tokens | Balanced for instruction-response pairs |

**How it works:**
1. Preference pairs: `(chosen_response, rejected_response)` for each instruction
2. Computes log-probabilities for both responses under current policy
3. Optimizes using DPO loss: maximizes `β × (log π_chosen - log π_rejected)` via negative log-sigmoid
4. No separate reward model needed — directly optimizes policy from human feedback signals

**Preference Dataset (6 examples):**
- Python decorators → detailed explanation vs one-liner
- Binary search → full implementation with comments vs broken code
- Transfer learning → thorough ML explanation vs vague definition
- LRU Cache → complete OrderedDict implementation vs empty class stub
- REST vs GraphQL → comprehensive technical comparison vs dismissive answer
- Gradient Descent → mathematical formulation vs oversimplified analogy

**Output:** Saves aligned model as `ashen_gpt_model_dpo.pk1` while preserving original SFT checkpoint as `ashen_gpt_model.pk1`.

---

## 🤖 Agentic Chatbot Interfaces

Both chatbots ship with **chain-of-thought reasoning** and a **ReAct tool loop** (max 5 reasoning steps per turn). The model autonomously emits `[TOOL: name(args)]` directives and continues until it reaches a final answer.

### CLI Chatbot (`chatbot.py`)

```cmd
run_chatbot.bat
```

**Core Commands:** `/clear` · `/help` · `/exit`

#### Session Management

| Command | Description |
|---|---|
| `/sessions` | List all saved sessions with message counts & timestamps. |
| `/new` | Create a fresh session. |
| `/load <id>` | Load a session by ID (from `/sessions`). |
| `/delete <id>` | Delete a session permanently. |
| `/rename <name>` | Rename current session. |

Sessions are stored as JSON files in `sessions_cli/` and include: conversation history, workspace context, and generation settings. Auto-named from first message.

#### Working Directory

| Command | Description |
|---|---|
| `/cd <path>` | Change working directory for all tool operations. |
| `/cd` (no args) | Show current working directory. |
| `/pwd` | Alias for `/cd`. |

Relative file paths in tools resolve from this directory. All tool executions (`read_file`, `write_file`, `glob`, `grep_search`, `run_shell_command`) operate within this context.

#### Workspace Context

| Command | Description |
|---|---|
| `/workspace <path>` | Scan a directory and inject its file tree into system prompts. |
| `/wctx off` | Clear workspace context. |

Gives the agent situational awareness of which files/folders you're examining without manually copying-pasting contents.

#### Tools (Agent Executable via `[TOOL: name(args)]`)

- `read_file(file_path='...')` — Read workspace files (relative to working dir).
- `write_file(file_path='...', content='...')` — Create/overwrite files.
- `glob(pattern='...')` — File discovery.
- `grep_search(pattern='...')` — Content search across `.py`, `.md`, `.txt`, `.bat`.
- `run_shell_command(command='...')` — Run any shell command (30s timeout).
- `web_search(query='...')` — Search DuckDuckGo for real-time information (returns top 5 result titles).
- `browse_url(url='...')` — Fetch webpage content, strip HTML tags, extract readable text.
- `deep_research(topic='...', max_searches=3)` — Autonomous multi-source research agent that searches, browses, and synthesizes findings into a structured markdown report.

### Web Chatbot (`web_chatbot.py`) — Cyberpunk UI

```cmd
run_web_chatbot.bat   →   http://localhost:5000
```

**Features:**

- **Cyberpunk aesthetic** — dark theme, neon accents, toggleable CRT scanline overlay.
- **Persona switcher** — *Ashen AI Agent*, *Code Architect*, *Cyber Companion*.
- **Model Hub modal** — browse local `.pk1` checkpoints, upload new weights, swap models live.
- **Quick-action chips** — one-click buttons for common agent tools (*File Glob*, *Grep*, *Git Status*, *Run Tests*, *Web Search*, *Browse URL*, *Deep Research*).

#### 🌐 Web Browsing & Research

The model can autonomously access live internet data:

| Tool | Description |
|---|---|
| `web_search` | Searches DuckDuckGo HTML interface, returns top 5 result titles with links. |
| `browse_url` | Fetches any webpage, strips HTML tags, extracts readable text (2K char limit). |
| `deep_research` | Autonomous multi-step research agent: searches → browses → synthesizes into markdown report. |

**Deep Research Workflow:**
1. **Phase 1**: Initial DuckDuckGo search for topic overview
2. **Phase 2**: Browses top sources, extracts key paragraphs ranked by informativeness
3. **Phase 3**: Follow-up searches on related topics
4. **Output**: Structured markdown report with source URLs and citations

Configurable depth via `max_searches` parameter. New quick-action buttons in sidebar for instant web research.
- **Adjustable settings** — temperature, top-k, max tokens, context length, GPU layer count, repeat penalty.
- **Session export & purge** — export chats as Markdown (`.md`) or purge history instantly.
- **Workspace browser** — navigate project directory directly from the UI.
- **Workspace context injection** — browsing any folder automatically injects its file tree into the model's system prompt so the agent is aware of which files you're examining.

#### ⚙️ Advanced Model & Inference Settings

##### Low-End GPU Optimization

For running larger models on GPUs with limited VRAM (&lt;8GB):

| Setting | Description | VRAM Savings |
|---|---|---|
| **Low-End GPU Mode** | Enables aggressive memory optimizations | ~30% |
| **FP16 Precision** | Half-precision floating point (~50% less VRAM) | ~40% |
| **BF16 Precision** | BFloat16 precision (balanced quality/performance) | ~40% |
| **CPU Offload Layers** | Moves transformer layers to system RAM | Scales with layer count |

**Recommended Configurations:**

| GPU VRAM | Precision | CPU Offload | Expected Speed |
|---|---|---|---|
| **2-4GB** | FP16 | 8-12 layers | Slow but functional |
| **6GB** | FP16 | 4-8 layers | Moderate |
| **8GB+** | BF16/FP32 | 0-2 layers | Good |

Configurable via ⚙️ Settings modal → 🔧 Low-End GPU Optimization section.

##### Draft Model & Speculative Decoding

Enable faster generation using a secondary "draft" model:

- **Speculative Decoding**: Draft model proposes tokens, main model verifies them in parallel
- **Configuration**: Enable toggle + adjust draft temperature in Settings modal
- **Requirement**: Place draft model at `ashen_gpt_model_draft.pk1` or upload via Model Hub

##### Standard Settings

- **Temperature**, **Top-K**, **Top-P** (Nucleus Sampling), **Max Output Tokens**, **Context Length**, **GPU Offload Layers**, **Repeat Penalty**.

| Action | Description |
|---|---|
| 📝 New Chat | Creates a fresh session; first message auto-names it. |
| 💬 Sessions Panel | Slide-out sidebar listing all past sessions with msg count & timestamps. |
| Load / ✎ / × | Click to restore full conversation history, rename, or delete a session. |
| Auto-save | Every chat message is persisted to `sessions/*.json`; settings, persona, and workspace context travel with each session. |

Sessions are stored as JSON files in `sessions/` and include: conversation history, persona choice, generation settings, and active workspace context.

---

#### 🧪 LLM Benchmark Suite

Built-in evaluation framework that tests model performance across 5 capability categories with 12 standardized questions:

| Category | # Tests | Description |
|---|---|---|
| **Knowledge** | 3 | Factual questions (Python release year, RAM definition, WWW creator) |
| **Code Generation** | 2 | Algorithm implementation (fibonacci recursion, bubble sort) |
| **Mathematical Reasoning** | 3 | Arithmetic & algebra (simple calc, quadratic eq., square root) |
| **Language Understanding** | 2 | Comprehension & translation (pronoun reference, Spanish translation) |
| **Ethics & Safety** | 1 | Judgment assessment (password sharing safety) |

**Scoring System:**
- Keyword matching against expected answers (proportional scoring)
- +0.5 bonus for chain-of-thought reasoning (`<think>` tags)
- Total possible: **23 points** across all tests
- Letter grade assignment (A+ at 90%+, D at <50%)

**Report Features:**
- Summary table with total/earned points and overall percentage
- Category breakdown scores with percentages
- Detailed per-question results with keyword match % and emoji grades
- Final letter grade based on aggregate performance

**Usage:**
- `[TOOL: run_benchmark()]` — Trigger full suite via agent tool
- 📖 **"Run Benchmark"** button in web chatbot quick actions
- Runs ~30-60 seconds; outputs structured markdown report

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
ashen_gpt_trainer.py    # Full training pipeline (Pre-training → SFT → DPO)
chatbot.py              # CLI agentic chatbot (sessions, workspace context, /cd)
web_chatbot.py          # Web UI agentic chatbot (cyberpunk, sessions, workspace context)
ashen_gpt_model.pk1     # SFT-aligned model (~127M params, 8K context)
ashen_gpt_model_dpo.pk1 # RL-aligned model (DPO preference optimization)
training_logs.txt       # Auto-generated training log
sessions/               # Web chatbot sessions (JSON with history, settings, context)
sessions_cli/           # CLI chatbot sessions (JSON with history, workspace context)
train_split.txt         # Literature training data
val_split.txt           # Validation data
code_train_split.txt    # Scraped code training data
run_*.bat               # Windows launch scripts
```

---

## 📜 License

Distributed under the MIT License. See `LICENSE` for details.
