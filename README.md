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

## 🔄 Automatic Checkpoint Detection & 2x Depth Upscaling

The training script features an intelligent model upscaling mechanism:
- **Checkpoint Detection**: Automatically detects if an existing checkpoint (`ashen_gpt_model.pk1`) is saved in the working directory.
- **2x Depth Multiplication**: If detected, it loads the model and automatically **doubles its transformer layer depth** (e.g., stacking layers from 8 to 16) using deep copies.
- **In-Place Overwrite**: Saves the upscaled model back under the exact same filename (`ashen_gpt_model.pk1`) before continuing training/fine-tuning.

---

## 📊 Hybrid Training & Evaluation

### 1. Progressive Staged Training Pipeline (500 Max Iters)
- **Stage 1 (Core Training)**: 512 context length (`eval_iters = 100`)
- **Stage 2 (Intermediate Extension)**: 2,048 context length (`eval_iters = 100`)
- **Stage 3 (Extreme Extension)**: 8,192 (8K) context length (`eval_iters = 100`)

### 2. Hybrid Data Pipeline
Combines literature (`train_split.txt`, `val_split.txt`) and scraped multi-language open-source code (`code_train_split.txt`) using robust BPE tokenization (`tiktoken` GPT-2 encoding).

### 3. Multi-Language Code Evaluation Suite
During pre-training evaluations, the model is tested across 6 programming languages (Python, JavaScript/TypeScript, Go, Rust, C++, Ruby) alongside natural language concept tests.

---

## 🤖 Smart Code-Output Chatbot (`chatbot.py` - 8K Context & Greyed-Out CoT)

The inference chatbot features an 8K context window, intent-based code filtering, and reasoning visualization:
- **Chain-of-Thought (CoT) Reasoning**: Automatically parses and displays the model's `<think>` reasoning block in **greyed-out terminal text** (`\033[90m`) before outputting the final response.
- **Conceptual Queries**: Automatically suppresses code blocks and replies in pure natural language.
- **Code Requests**: Permits code generation only when the user explicitly asks for code, scripts, functions, or syntax.

---

## 🛠️ Getting Started & Usage

### Prerequisites
- Python 3.10+
- PyTorch with CUDA support
- `tiktoken`, `transformers`, `accelerate`

### Running the Scripts

- **Train / Upscale Custom ~127M Model (512 -> 2K -> 8K Pre-training + SFT + Auto-Upscaling)**:
  ```cmd
  run_ashen_gpt.bat
  ```
- **Interact with Custom Chatbot (8K Context & Grey CoT)**:
  ```cmd
  run_chatbot.bat
  ```

---

## 📜 License

Distributed under the MIT License. See `LICENSE` for details.
