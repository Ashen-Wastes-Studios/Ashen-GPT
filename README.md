# Ashen GPT

Ashen GPT is a high-performance, custom PyTorch implementation of a **Qwen-like Transformer architecture** trained from scratch on a hybrid dataset of literature and multi-language open-source code. It is specifically optimized for consumer hardware (tested and tuned for 8GB VRAM GPUs).

---

## 🚀 Key Architectural Features

- **RMSNorm**: Replaces standard LayerNorm for enhanced training stability and efficiency.
- **Rotary Position Embeddings (RoPE)**: Replaces absolute position embeddings for superior context handling and length extrapolation.
- **QK-Norm**: Applies RMS normalization directly to Query and Key projections prior to attention score calculations.
- **SwiGLU Mixture of Experts (MoE)**: Features sparse MoE feed-forward networks with Top-2 expert routing and SwiGLU activation (`SiLU(gate) * up`).
- **FlashAttention**: Utilizes PyTorch's optimized `scaled_dot_product_attention` for fast, memory-efficient attention execution.

---

## ⚙️ Hardware Optimization (8GB VRAM Peak 450M Scale)

Training large models from scratch on consumer GPUs is heavily constrained by optimizer states and gradient memory. Ashen GPT is tuned to the **peak safe limit (~450 Million parameters)** for an 8GB GPU:
- **Embedding Dimension (`n_embd`)**: `896`
- **Transformer Layers (`n_layer`)**: `16`
- **Attention Heads (`n_head`)**: `14`
- **MoE Experts (`num_experts`)**: `4` (Top-2 routing)
- **Memory Savers**: Built-in **Gradient Checkpointing** (`torch.utils.checkpoint`), Automatic Mixed Precision (`torch.amp`), and optional 8-bit AdamW optimizer (`bitsandbytes`).

---

## 📊 Hybrid Training & Evaluation

### 1. Hybrid Data Pipeline
Combines literature (`train_split.txt`, `val_split.txt`) and scraped multi-language open-source code (`code_train_split.txt`) using robust BPE tokenization (`tiktoken` GPT-2 encoding).

### 2. Multi-Language Code Evaluation Suite
During pre-training evaluations, the model is tested on:
- **Natural Language Text Tests**: Evaluated on conceptual tasks with automated smart code-output filtering.
- **Clock App Code Generation Tests**: Prompted to build a clock app across **6 supported programming languages**:
  1. Python
  2. JavaScript / TypeScript
  3. Go
  4. Rust
  5. C++
  6. Ruby

---

## 🤖 Smart Code-Output Chatbot (`chatbot.py`)

The inference chatbot features intent-based code filtering:
- **Conceptual Queries**: Automatically suppresses code blocks and replies in pure natural language.
- **Code Requests**: Permits code generation only when the user explicitly asks for code, scripts, functions, or syntax.

---

## 🛠️ Getting Started & Usage

### Prerequisites
- Python 3.10+
- PyTorch with CUDA support
- `tiktoken`, `bitsandbytes`, `transformers`, `accelerate`

### Running the Scripts

- **Train Custom 450M Model (Pre-training + SFT)**:
  ```cmd
  run_ashen_gpt.bat
  ```
- **Interact with Custom Chatbot**:
  ```cmd
  run_chatbot.bat
  ```

---

## 📜 License

Distributed under the MIT License. See `LICENSE` for details.
