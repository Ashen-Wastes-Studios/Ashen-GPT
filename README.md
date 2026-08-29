# Ashen GPT

Ashen GPT is a high-performance, custom PyTorch implementation of a **Qwen-like Transformer architecture** trained from scratch on a hybrid dataset of literature and multi-language open-source code. It is specifically optimized for consumer hardware (tuned for 8GB+ VRAM GPUs with strict memory protection against shared memory bleeding).

---

## 🚀 Key Architectural Features

- **RMSNorm**: Replaces standard LayerNorm for enhanced training stability and efficiency.
- **Rotary Position Embeddings (RoPE)**: Features Dynamic NTK-aware scaling for robust length extrapolation across extended contexts.
- **QK-Norm**: Applies RMS normalization directly to Query and Key projections prior to attention score calculations.
- **SwiGLU Mixture of Experts (MoE)**: Features sparse MoE feed-forward networks with Top-2 expert routing and SwiGLU activation (`SiLU(gate) * up`).
- **Memory-Efficient Attention**: Utilizes PyTorch's optimized `scaled_dot_product_attention` combined with self-attention gradient checkpointing.

---

## ⚙️ Hardware Optimization (~127M Scale & Shared Memory Protection)

Training large models from scratch on consumer GPUs is heavily constrained by activation memory and optimizer states. Ashen GPT is tuned to **~127 Million parameters** (`n_embd = 512`, `n_layer = 8`, `n_head = 8`) for lightning-fast, OOM-free training on consumer GPUs:
- **Zero Shared Memory Bleeding**: Implements strict self-attention gradient checkpointing (`torch.utils.checkpoint`), periodic CUDA cache clearing (`torch.cuda.empty_cache()`), and garbage collection (`gc.collect()`) to ensure training stays 100% inside dedicated VRAM.
- **Progressive Context Staging (512 -> 2K -> 8K)**: Safely scales context length through progressive stages to prevent quadratic ($O(T^2)$) attention memory spikes.

---

## 🔄 Automatic Checkpoint Detection & Training Logs

- **2x Depth Upscaling**: Automatically detects existing checkpoints (`ashen_gpt_model.pk1`) and doubles transformer layer depth (e.g., 8 -> 16 layers) while preserving weights.
- **Automated Training Logging (`training_logs.txt`)**: All training output, step losses, evaluation results, generation tests, and fine-tuning epochs are automatically captured and streamed to `training_logs.txt` with timestamped run sessions while remaining visible in your terminal.

---

## 🤖 Ashen AI Cybernetic Hub & Agentic CLI Chatbots

Ashen GPT includes both a multi-turn CLI chatbot and a feature-rich, cybernetic **Ashen AI Web Interface** equipped with **full agentic capabilities (ReAct tool execution loop)**.

### ⚡ Ashen AI Web Hub Features
- **Cyberpunk / Retro-Gaming Aesthetic**: Deep dark theme with neon accents and toggleable CRT scanline visual effects (`CRT FX: ON/OFF`).
- **Model Hub & Checkpoint Manager**: A dedicated modal window allowing users to **view local model checkpoints**, **upload new model weights (`.pk1` / `.pt`)**, and **instantly switch active models** in memory.
- **Live System Telemetry**: Real-time display of active model name, PyTorch CUDA backend, and 8K token context window.
- **Persona Switcher**: Switch between *Ashen AI Agent*, *Code Architect*, and *Cyber Companion* personas on the fly.
- **Quick Action Chips**: One-click execution for common tasks (*Run Tests*, *Git Status*, *File Glob*, *Grep Search*).
- **Session Export & Purge**: Export chat sessions as Markdown (`.md`) or purge memory instantly.

### 🛠️ Agentic Tool Execution
The model can autonomously reason over tasks and invoke the following tools:
- `read_file(file_path="...")` — Read file contents from the workspace.
- `write_file(file_path="...", content="...")` — Create or modify files.
- `glob(pattern="...")` — Discover files matching glob patterns.
- `grep_search(pattern="...")` — Search for code patterns across project files.
- `run_shell_command(command="...")` — Execute terminal operations (e.g. `pytest`, `git status`).

---

## 📊 Hybrid Training & Evaluation

### 1. Progressive Staged Training Pipeline (5,000 Max Iters)
- **Stage 1 (Core Training)**: 512 context length (`iters 0-3000`)
- **Stage 2 (Intermediate Extension)**: 2,048 context length (`iters 3001-4500`)
- **Stage 3 (Extreme Extension)**: 8,192 (8K) context length (`iters 4501-5000`)

### 2. Hybrid Data Pipeline
Combines literature (`train_split.txt`, `val_split.txt`) and scraped multi-language open-source code (`code_train_split.txt`) using robust BPE tokenization (`tiktoken` GPT-2 encoding) and memory-mapped streaming.

### 3. Supervised Fine-Tuning (SFT)
- **Curated Instruction Dataset**: Diverse instruction-following and coding tasks structured with explicit `<think>` reasoning traces.

---

## 🚀 Getting Started & Usage

### Prerequisites
- Python 3.10+
- PyTorch with CUDA support
- `tiktoken`, `requests`

### Running the Scripts

- **Train / Upscale Custom ~127M Model (Logs to `training_logs.txt`)**:
  ```cmd
  run_ashen_gpt.bat
  ```
- **Interact via Agentic CLI Chatbot (`chatbot.py`)**:
  ```cmd
  run_chatbot.bat
  ```
- **Launch Ashen AI Cybernetic Web Hub (`http://localhost:5000`)**:
  ```cmd
  run_web_chatbot.bat
  ```

---

## 📜 License

Distributed under the MIT License. See `LICENSE` for details.
