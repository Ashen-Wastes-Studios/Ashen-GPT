---
name: Hybrid Code and Literature Training Pipeline with MoE
description: Custom GPT transformer training pipeline in ashen-gpt.ipynb combining literature and multi-language open source code with a Mixture of Experts (MoE) architecture.
type: project
---

The project trains a custom PyTorch GPT model (`ashen-gpt.ipynb`) on a hybrid dataset combining literature (`train_split.txt`) and multi-language open-source code (`code_train_split.txt` scraped via GitHub raw tree crawling in a round-robin fashion across Python, JavaScript, Go, Rust, C++, and Ruby). The tokenizer is robust against unseen code characters via `.get(c, 0)`, and the Transformer block uses a Top-2 Mixture of Experts (MoE) FFN architecture with 4 experts.

**Why:** To train the language model on both natural language text and diverse programming languages while experimenting with advanced scaling architectures like MoE.

**How to apply:** When modifying training data loading or model architecture in this repository, preserve the hybrid data pooling (`train_split.txt` + `code_train_split.txt`), robust tokenization fallback (`.get(c, 0)`), and the MoE FeedForward block.
