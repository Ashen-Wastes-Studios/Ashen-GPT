# Ashen GPT

Ashen GPT is a local, self-hosted AI assistant that runs entirely on consumer hardware
(8 GB+ VRAM). It pairs **two model pipelines** with **two user interfaces** that share the
same backend behaviors:

- **Qwen3.5 fine-tune pipeline** (`qwen_finetune.py`) — the default/active model
  (`ashen_gpt_model/`, a LoRA fine-tune of Qwen3.5-0.8B emitted as a merged HuggingFace
  checkpoint). This is what the chatbots load by default.
- **Legacy custom-model pipeline** (`ashen_gpt_trainer.py`) — the original from-scratch
  Qwen-style MoE trained on the book-code corpus, saved as `ashen_gpt_model.pk1`. Still
  supported and swappable via Model Hub / `/model`.

Two front-ends, identical capabilities:

- **`chatbot.py`** — a single-file terminal (CLI) chatbot. Self-contained: it folds the
  full Qwen/engine/swarm/council/research/settings backend into one file with **zero
  dependency on `web_chatbot.py`**.
- **`web_chatbot.py`** — a browser UI (stdlib `http.server`, `http://localhost:5000`)
  with a cyberpunk theme, Model Hub, live-streaming chain-of-thought, swarm/council, and a
  self-improvement dashboard.

Both front-ends route generation through the same `QwenModelAdapter` (and, for the legacy
model, the same `AshenAIAgenticEngine`), so behavior — reasoning, tool calls, citations,
CoT display — is consistent across CLI and web.

---

## Key Architectural Features

| Component | Qwen fine-tune (`ashen_gpt_model/`) | Legacy custom (`ashen_gpt_model.pk1`) |
|---|---|---|
| Base | Qwen3.5-0.8B (HuggingFace) | From-scratch Qwen-style decoder-only |
| Norm | Qwen `RMSNorm` + QK-Norm | `RMSNorm` (no mean-centering) |
| Position | Qwen `RotaryEmbedding` (RoPE) | `RotaryEmbedding` with Dynamic NTK scaling |
| Attention | `scaled_dot_product_attention` (causal) | same |
| FFN | SwiGLU | SwiGLU **MoE** — 4 experts, Top-2 gating |
| Tokenizer | Qwen BPE (`AutoTokenizer`) | GPT-2 BPE via `tiktoken` (50,257 vocab) |
| Fine-tune | LoRA (bf16, `r=32`) → merged HF ckpt + `class_head.pt` | Full / width-upscaled weights (`.pk1`) |
| Max context | 8,192 tokens (`context_length`) | 8,192 tokens (`block_size`) |

**Hardware optimization (RTX 3060 Ti, 8.59 GB VRAM):**
- Qwen inference defaults to **4-bit (bitsandbytes)** to fit the weights in ~0.6 GB; full
  precision is opt-in via `QWEN_INFER_4BIT=0`.
- `kv_cap` / `gen_cap` clamp the KV-cache and generation length so the model cannot OOM the
  GPU even at `context_length=8192` (tunable: `QWEN_KV_CAP`, `QWEN_GEN_CAP`).
- `ashen_gpt_trainer.py` uses gradient checkpointing (Stage 3 / long context) and periodic
  `empty_cache()` + `gc.collect()` to stay within VRAM.
- **Width-upscaling:** on re-run, existing checkpoints are detected and **width-upscaled by
  √2** (hidden size / `n_embd` widened, depth unchanged) with copy-init, so parameter count
  ~doubles with minimal loss spike. `ashen_gpt_trainer.py` does this for the legacy model
  (512 → 720); `qwen_finetune.py` does it for the Qwen checkpoint on resume.

> **Note on bitsandbytes warning:** the fine-tuned `hidden_size` (1448) is not a multiple of
> 64, so bitsandbytes' 4-bit matmul falls back to a slower kernel and prints a harmless
> `UserWarning`. The dimension is fixed by the saved weights and cannot be changed without
> re-training; it is silenced in both loaders. **Do not** edit `config.json`'s `hidden_size`
> to "fix" this — it would crash loading with a shape mismatch.

---

## Agentic Chatbot Interfaces

Both chatbots feature **chain-of-thought reasoning** streamed live (gray thought / white
answer in the CLI; collapsible `🧠 Thought for <model>` panel on the web) and a **ReAct
tool loop** (`[TOOL: name(args)]`) that runs up to a few reasoning steps per turn, then
emits a final answer with citations when a tool was used.

### CLI Chatbot (`chatbot.py`)

```cmd
run_chatbot.bat
:: or
cuda\Scripts\python.exe chatbot.py
```

**Slash commands** (type `/help` in-app for the full, current list):

`/clear` · `/help` · `/exit` · `/models` · `/model <path>` · `/settings [k=v]` ·
`/persona <name>` · `/swarm` · `/council` · `/research <topic>` · `/websearch <query>` ·
`/selfimprove` · `/up` `/down` · `/sessions` · `/workspace` · `/cd` `/pwd` ·
`/auto-swarm` `/auto-research` · `/cyber on|off` · `/benchmark`

#### Cybernetic theme & pinned input box
- **Cybernetic theme** — neon / box-drawing terminal UI (toggle with `/cyber on|off`, or set
  `ASHEN_CYBER=1`). Defaults on when stdout is a TTY.
- **Pinned input box** — the prompt box is anchored to the bottom of the terminal via an ANSI
  scroll region (`DECSTBM`); the model's chain-of-thought and answer stream *above* it. After
  you press Enter the box **clears in place** so you get a fresh prompt line each turn. Falls
  back to a plain prompt when not a TTY, when the cyber theme is off, or on a short terminal.
- On `/exit` (or EOF) the scroll region is restored so your shell isn't left in a broken state.

#### Chain-of-thought display — identical to the web chatbot
The CLI renders the streamed chain-of-thought with the **exact same UX as the web UI**,
because both consume the same `AshenAIAgenticEngine.solve_with_agent_stream` event stream
(the CLI is a self-contained port — no import of `web_chatbot.py`):

```
● Thought for <model>          (amber dot + model label)
<gray streamed chain-of-thought>
· Xs · ~N tokens               (thought timing + token estimate)
<white streamed answer>
◈ <model> · <time>             (footer)
Sources                        (when the model cites sources)
  1› title — url
[TOOL: name(args)]             (agent tool calls, when used)
```

CoT is gray and the final answer is white — same semantics as the web chatbot's collapsed
`🧠 Thought for <model>` panel.

#### Sessions, workspace, and tools
- **Sessions** — `/sessions` lists saved sessions; `/new`, `/load <id>`, `/delete <id>`,
  `/rename <name>` manage them. Stored as JSON in `sessions_cli/` (history + workspace
  context + generation settings), auto-named from the first message.
- **Working directory** — `/cd <path>` / `/pwd` set the root for all file tools.
- **Workspace context** — `/workspace <path>` scans a directory and injects its file tree
  into the system prompt so the agent knows which files you're looking at.
- **Agent tools (via `[TOOL: name(args)]`)** — `read_file`, `write_file`, `glob`,
  `grep_search`, `run_shell_command`, `web_search`, `browse_url`,
  `deep_research(topic, max_searches)`.

### Web Chatbot (`web_chatbot.py`) — Cyberpunk UI

```cmd
run_web_chatbot.bat            →   http://localhost:5000
:: or
cuda\Scripts\python.exe web_chatbot.py --port 5000 --host localhost
```

> **Bind:** `http://localhost:5000` (also `127.0.0.1:5000`). Always use `http://` —
> `https://` will not connect. If the port is busy the server logs
> `[ERROR] Could not bind to localhost:5000`.

**Features:**
- **Cyberpunk aesthetic** — dark theme, neon accents, toggleable CRT scanline overlay.
- **Persona switcher** — *Ashen AI Agent*, *Code Architect*, *Cyber Companion*.
- **Model Hub modal** — browse local `.pk1`/`.gguf` checkpoints **and Qwen HuggingFace
  dirs** (e.g. `ashen_gpt_model/`, auto-tagged `QWEN`), upload weights, swap models live.
  GGUF alternatives (e.g. `Jackrong_Qwopus3.5-9B-coder-Exp-Q3_K_M.gguf`) are switchable here.
- **Quick-action chips** — one-click buttons for common tools (*File Glob*, *Grep*,
  *Git Status*, *Run Tests*, *Web Search*, *Browse URL*, *Deep Research*).
- **Live CoT streaming** — `POST /api/chat/stream` returns NDJSON
  (`thought_delta → thought_done → response_delta → response_done → done`, plus
  `tool_start`/`tool_result` mid-loop); the UI updates the thought `pre` and response `div`
  in place with a live cursor. Batched `POST /api/chat` is the fallback.
- **Web browsing & research** — `web_search` (DuckDuckGo, top 5 titles+links), `browse_url`
  (fetches + strips a page to readable text), and `deep_research` (autonomous
  search → browse → synthesize into a cited markdown report).
- **Adjustable settings** — temperature, top-k, top-p, max tokens, context length, GPU
  layers, repeat penalty, precision, CPU offload. Low-end GPU presets (FP16/BF16, CPU
  offload layers) for <8 GB cards. Optional **speculative decoding** with a draft model.
- **Session export & purge**, **workspace browser**, and **workspace context injection**.

#### Settings persistence (`settings.json`)
`web_chatbot.py` resolves `settings.json` via `__file__` so it works regardless of `cwd`:

```python
SETTINGS_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'settings.json')
```

Precedence: `CLI --settings <path>` > `SETTINGS_PATH` / `ASHEN_SETTINGS` env > `settings.json`
next to the script. On first run a default file is auto-created and merged, so partial saves
never wipe `current_model`.

| Key | Default | Notes |
|---|---|---|
| `temperature` | `0.70` | |
| `top_k` / `top_p` | `40` / `0.9` | |
| `max_new_tokens` / `context_length` | `250` / `8192` | |
| `gpu_layers` / `precision` / `cpu_offload_layers` | `16` / `fp16` / `0` | |
| `current_model` | `ashen_gpt_model.pk1` — or an HF dir like `ashen_gpt_model` (Qwen3.5) | |
| `show_chain_of_thought` | `true` | |

#### Self-improvement loop, Swarm, and Council
- **Self-improvement** — file-based (no DB): `feedback.json` (last 500 ratings) +
  `self_improvement.json` (`{stats, log, suggestions}`). Every reply has
  `👍 / 👎 / 🔄 Retry` → `POST /api/feedback`; `🔄 Retry` / correction re-runs the engine
  with `User correction:` injected. The `🧬 Self-Improve` dashboard shows stats and an
  Auto-Tune (drops temperature when down-vote ratio is high).
- **Swarm** — spawns 2–6 isolated engine copies (roles: Researcher/Coder/Critic/Planner/
  Executor/Analyst) in `parallel` / `divide` / `debate` modes, then a Synthesizer merges the
  drafts. Live variant streams each agent's CoT.
- **Council** — drafts → critics vote & suggest → tallied winner → finalizer refines.
  Useful for hard prompts that benefit from multi-perspective critique.

#### LLM benchmark suite
Built-in evaluation across 5 categories (Knowledge, Code Generation, Math Reasoning,
Language Understanding, Ethics & Safety) with 12 standardized questions and a letter grade.
Trigger via `[TOOL: run_benchmark()]` or the web **Run Benchmark** button (~30–60 s).

---

## Training & Fine-Tuning

There are **two** training entry points, one per model pipeline. Both tee their output to
**`training_logs.txt`** (real-time, ANSI-stripped) in addition to the terminal.

### `qwen_finetune.py` — LoRA fine-tune of the Qwen3.5 checkpoint (default)

Fine-tunes `Qwen_Qwen3.5-0.8B` (or an existing `ashen_gpt_model/` resume) with **LoRA in
bf16** and emits a **merged HuggingFace checkpoint** at `ashen_gpt_model/` (plus a small
`class_head.pt` for intent routing). This is the model the chatbots load by default.

Why LoRA/bf16: a 0.75B bf16 base (~1.5 GB) + LoRA adapters (~9M params) + AdamW state +
activations fits comfortably in 8.59 GB; a full bf16 fine-tune would not.

- **Behavior is baked into training data, not the prompt.** The `SYSTEM_PROMPT` carries a
  `REASONING GUIDE` (show your work → self-critique → weigh approaches → final answer), and
  the SFT pool includes explicit chain-of-thought examples. Per the project rule, *all*
  "how to respond" behavior lives in SFT/DPO data — the inference prompt is never used to
  steer style.
- **Reasoning in the training loop.** The eval step streams generation token-by-token: the
  chain-of-thought prints **in gray**, flips to **white** when the model reaches its answer,
  and flushes every token so you watch it in real time. The plain-text reply is also written
  to `training_logs.txt`.
- **Key knobs (env vars):**
  - `QWEN_ITERS` — training iterations (default `200`)
  - `QWEN_EVAL_EVERY` — eval/stream cadence (default `20`)
  - `QWEN_CKPT_EVERY` — periodic merged checkpoint (default = eval cadence)
  - `QWEN_GEN_TOKENS` — eval generation length (default `64`)
  - `QWEN_CORPUS` — set `1` to train on the raw book/code corpora instead of the curated SFT
    examples
  - `QWEN_LORA_R` / `QWEN_LORA_ALPHA` / `QWEN_LORA_DROPOUT` — LoRA rank / alpha / dropout
    (default rank `32`)
  - `QWEN_KV_CAP` / `QWEN_GEN_CAP` — inference-time VRAM caps (default `2048` / `512`)
  - `QWEN_EVAL_PROMPT` / `QWEN_PROMPT_POOL` — override the eval prompt (single / `|||`-joined)
  - `QWEN_SFT_JSONL` / `QWEN_CLS_JSONL` — paths to SFT / classification data
- **Width-upscale on resume** — hidden size is widened by √2 (copy-init, depth unchanged) so
  a resumed run grows capacity without retraining from scratch.
- **Checkpoints** — `save_checkpoint()` merges LoRA via `merge_and_unload()` into a plain
  `Qwen3_5ForCausalLM`, writes `ashen_gpt_model/`, and keeps a timestamped history copy
  `ashen_gpt_model.ckpt-{it}`.

Run it:

```cmd
cuda\Scripts\python.exe qwen_finetune.py
:: or
run_qwen_finetuner.bat
```

### `ashen_gpt_trainer.py` — legacy custom-model pre-training

Trains the from-scratch Qwen-style **MoE** (SwiGLU, 4 experts, Top-2 gating) on the
book/code corpus and saves `ashen_gpt_model.pk1`. Architecture: `n_embd=512`,
`n_layer=8`, `n_head=8`, `block_size=8192`, progressive context (512 → 2K → 8K),
gradient checkpointing at long context, and **width-upscaling by √2** (512 → 720) on re-run.
Data is streamed from `train_split.txt` / `code_train_split.txt` / `val_split.txt` via
memory-mapped windows (O(1) RAM regardless of dataset size). Output is teed to
`training_logs.txt` through a `Tee` stdout wrapper.

Run it:

```cmd
python ashen_gpt_trainer.py
:: or
run_ashen_gpt.bat
```

---

## Data Pipeline

- **Literature** — `train_split.txt` / `val_split.txt` (raw-text book corpus).
- **Code** — `code_train_split.txt` (scraped public GitHub source).
- **SFT / classification** — `qwen_finetune.py` consumes `sft_data.jsonl` /
  `cls_data.jsonl` for reasoning + intent-routing examples.
- Memory-mapped streaming (`mmap.mmap`) for O(1) RAM regardless of dataset size; the Qwen
  trainer selects between text and code splits during training.

---

## Getting Started

### Prerequisites
- Python 3.10+ (the bundled `cuda\` venv has torch + transformers + peft + bitsandbytes).
- A CUDA GPU with 8 GB+ VRAM recommended (4-bit inference works on smaller cards).

### Quick start

| Script | Description | Command |
|---|---|---|
| `ashen_gpt_trainer.py` | Legacy custom-model pre-training + width-upscale | `python ashen_gpt_trainer.py` / `run_ashen_gpt.bat` |
| `qwen_finetune.py` | LoRA bf16 fine-tune + width-upscale the Qwen3.5 checkpoint (`ashen_gpt_model/`) | `cuda\Scripts\python.exe qwen_finetune.py` / `run_qwen_finentuner.bat` |
| `chatbot.py` | Terminal-based agentic chatbot | `python chatbot.py` / `run_chatbot.bat` |
| `web_chatbot.py` | Browser-based cyberpunk UI (port 5000) | `python web_chatbot.py` / `run_web_chatbot.bat` |

**Web chatbot examples:**

```cmd
call cuda\Scripts\activate.bat
python web_chatbot.py                  :: → http://localhost:5000  (settings.json auto-created if missing)
python web_chatbot.py --port 5000 --host localhost
python web_chatbot.py --settings "C:\path\to\custom_settings.json"
set SETTINGS_PATH=C:\path\to\settings.json && python web_chatbot.py
```

### Typical workflow
1. **Train/fine-tune** — `qwen_finetune.py` produces the default `ashen_gpt_model/`;
   `ashen_gpt_trainer.py` produces the legacy `ashen_gpt_model.pk1`.
2. **Chat** — launch the CLI or web interface; the model loads the saved checkpoint
   automatically (`current_model` in `settings.json` selects which).
3. **Agent mode** — ask the model to inspect files, run commands, or explore your workspace.
4. **Evaluate & improve** — `[TOOL: run_benchmark()]` or the web **Run Benchmark** button;
   use Swarm / Council for hard prompts and the Self-Improve dashboard to tune from feedback.

---

## Project Structure

```
qwen_finetune.py       # LoRA (bf16) fine-tune of Qwen3.5 -> ashen_gpt_model/; width-upscale on resume; tees training_logs.txt
ashen_gpt_trainer.py   # Legacy custom MoE pre-training -> ashen_gpt_model.pk1; Tee stdout -> training_logs.txt
run_qwen_finetuner.bat # Windows launcher for qwen_finetune.py (cuda venv)
run_ashen_gpt.bat      # Windows launcher for ashen_gpt_trainer.py
chatbot.py             # Self-contained CLI chatbot (no import of web_chatbot.py)
web_chatbot.py         # Browser cyberpunk UI (stdlib http.server, port 5000)
run_chatbot.bat        # Windows launcher for chatbot.py
run_web_chatbot.bat    # Windows launcher for web_chatbot.py
ashen_gpt_model/       # Default model (Qwen3.5 fine-tune, HF format): config.json + safetensors + class_head.pt
ashen_gpt_model.pk1    # Legacy custom MoE model (~127M params, 8K context)
ashen_gpt_model_lora/  # Adapter-only LoRA checkpoint from qwen_finetune.py
settings.json          # Generation + display config (resolved via __file__)
feedback.json          # Last 500 user ratings (👍/👎 + corrections)
self_improvement.json  # {stats, log, suggestions} — swarm/council/gibberish_fix/regenerate/auto_tune entries
training_logs.txt      # Auto-generated training log (terminal + this file, ANSI-stripped)
sessions/              # Web chatbot sessions (JSON: history, settings, workspace context)
sessions_cli/          # CLI chatbot sessions (JSON: history, workspace context)
train_split.txt        # Literature training data
val_split.txt          # Validation data
code_train_split.txt   # Scraped code training data
sft_data.jsonl         # Qwen SFT reasoning examples
cls_data.jsonl         # Intent-classification examples
run_*.bat              # Windows launch scripts
```

**Key web endpoints** (served by the stdlib `http.server` `ChatHandler`; unknown paths → 404):

*Core chat*
| Method | Path | Purpose |
|---|---|---|
| `GET` | `/` | Cyberpunk UI — injects `current_model_filename` |
| `POST` | `/api/chat` | `{message}` → `{thought, response, model, model_path, show_chain_of_thought}` (batched) |
| `POST` | `/api/chat/stream` | NDJSON live: `thought_delta* → thought_done → response_delta* → response_done → done` (+ `tool_start`/`tool_result`) |
| `POST` | `/api/clear` | Save current session, then clear history + workspace context |
| `POST` | `/api/persona` | `{persona}` → switch active persona |
| `POST` | `/api/settings` | `{...}` → live-apply (no file write) |

*Sessions*
| Method | Path | Purpose |
|---|---|---|
| `GET` | `/api/sessions` | List sessions + `current` id |
| `POST` | `/api/sessions/new` | Create → `201 {session_id}` |
| `POST` | `/api/sessions/load` | `{session_id}` → restore history, persona, workspace context |
| `POST` | `/api/sessions/delete` | `{session_id}` → delete (clears current if active) |
| `POST` | `/api/sessions/rename` | `{session_id, name}` |

*Settings persistence*
| Method | Path | Purpose |
|---|---|---|
| `GET,POST` | `/api/settings/load` | → `{"status":"success","settings":...}` |
| `POST` | `/api/settings/save` | Merge + write `settings.json` + live-apply → `{"status":"saved","settings":...}` |

*Model Hub*
| Method | Path | Purpose |
|---|---|---|
| `GET` | `/api/models` | Scan local PC for `.pk1`/`.gguf` (deduped by path) |
| `GET` | `/api/hf-search` | `?q=` → HuggingFace model search (top 20 by downloads) |
| `POST` | `/api/models/list` | `scan_available_models()` → `{models, current}` |
| `POST` | `/api/models/set-default` | `{path}` → set + persist `current_model` |
| `POST` | `/api/models/scan-pc` | Re-scan local PC |
| `POST` | `/api/models/switch` | `{filename}` → swap live model (`.gguf` by path, `.pk1` via `pickle.load`) |
| `POST` | `/api/models/upload` | `{filename, content_base64}` → save weights to script dir |
| `POST` | `/api/models/download-hf-repo` | `{repo_id}` → `snapshot_download` from HuggingFace |

*Workspace*
| Method | Path | Purpose |
|---|---|---|
| `GET` | `/api/workspace/list` | `?dir=` → directory listing |
| `POST` | `/api/workspace/read` | `{path}` → file content (utf-8, latin-1 fallback) |
| `POST` | `/api/workspace/context` | `{dir}` → inject dir tree into model context |
| `POST` | `/api/workspace/write` | `{path, content}` → write file |

*Feedback & Self-Improvement*
| Method | Path | Purpose |
|---|---|---|
| `GET` | `/api/feedback` | Last 50 feedback entries |
| `GET` | `/api/self-improve` | `{stats, entries[-20], suggestions[-10], feedback_recent, gibberish_rate}` |
| `POST` | `/api/feedback` | `{rating, prompt, response, thought, correction, model}` |
| `POST` | `/api/self-improve` | `{action: regenerate\|auto-tune\|analyze, run_benchmark}` |

*Swarm*
| Method | Path | Purpose |
|---|---|---|
| `GET` | `/api/swarm` | `{roles[6], recent_runs, model, model_path}` |
| `POST` | `/api/swarm` | `{task\|prompt, num_agents:2-6, mode}` → `{agents, synthesis, elapsed_s}` |
| `POST` | `/api/swarm/stream` | Live `swarm_start → agent_* → synthesis_* → done` |

*Council*
| Method | Path | Purpose |
|---|---|---|
| `GET` | `/api/council` | `{roles (5 critics), proposers[6], recent_runs, model, model_path}` |
| `POST` | `/api/council` | `{task, num_drafts:2-5, num_critics:2-5}` → `{drafts, critics, tally, winner, final}` |
| `POST` | `/api/council/stream` | Live `council_start → draft_* → critic_* → tally → final_* → done` |

---

## License

Distributed under the MIT License. See `LICENSE` for details.
