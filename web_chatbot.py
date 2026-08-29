import sys
import io
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

import os
import json
import http.server
import socketserver
import threading
import torch
import torch.nn as nn
from torch.nn import functional as F
import tiktoken
import pickle
import re
import subprocess
import glob as glob_module
import datetime
import base64
import requests
from urllib.parse import urlparse, parse_qs
from pathlib import Path

# --- Session Storage ---
SESSIONS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'sessions')
os.makedirs(SESSIONS_DIR, exist_ok=True)
current_session_id = None

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Device: {device}")

block_size = 8192
n_embd = 512
n_layer = 8
n_head = 8
dropout = 0.1
num_experts = 4
top_k = 2

enc = tiktoken.get_encoding("gpt2")
vocab_size = enc.n_vocab

encode = lambda s: enc.encode(s, allowed_special={"<|endoftext|>"})
decode = lambda l: enc.decode(l)

def user_wants_code(prompt):
    code_keywords = [
        'code', 'write a', 'function', 'script', 'program', 'python', 'javascript',
        'html', 'css', 'sql', 'syntax', 'class ', 'def ', 'implementation', 'algorithm'
    ]
    prompt_lower = prompt.lower()
    return any(keyword in prompt_lower for keyword in code_keywords)

def filter_code_output(text):
    text_no_blocks = re.sub(r'```[\s\S]*?```', '[Code logic analyzed internally. Ask me to write code if you want to see the implementation snippet.]', text)
    text_clean = re.sub(r'`[^`]*`', '', text_no_blocks)
    return text_clean

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

    def forward(self, index, targets=None, current_block_size=8192):
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
        loss = None if targets is None else F.cross_entropy(logits.view(B*T, -1), targets.view(B*T))
        return logits, loss

    def generate(self, index, max_new_tokens, current_block_size=8192, temperature=0.8, top_k=50):
        for _ in range(max_new_tokens):
            index_cond = index[:, -current_block_size:]
            logits, loss = self.forward(index_cond, current_block_size=current_block_size)
            logits = logits[:, -1, :] / temperature
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            probs = F.softmax(logits, dim=-1)
            index_next = torch.multinomial(probs, num_samples=1)
            index = torch.cat((index, index_next), dim=-1)
        return index

current_model_filename = 'ashen_gpt_model.pk1'
if os.path.exists(current_model_filename):
    print(f"Loading Ashen GPT model parameters from {current_model_filename}...")
    with open(current_model_filename, 'rb') as f:
        model = pickle.load(f)
    print("Model loaded successfully!")
else:
    print(f"No checkpoint found at {current_model_filename}. Initializing new Ashen GPT model...")
    model = AshenGPTLanguageModel(vocab_size)

m = model.to(device)

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
        self.workspace_context = ""  # injected context from browsed workspace

    def clear_history(self):
        self.history = []

    def set_persona(self, persona):
        self.persona = persona

    def set_workspace_context(self, workspace_info):
        """Set the current workspace directory listing as model context."""
        self.workspace_context = workspace_info

    def update_settings(self, settings):
        self.temperature = float(settings.get('temperature', self.temperature))
        self.top_k = int(settings.get('top_k', self.top_k))
        self.top_p = float(settings.get('top_p', self.top_p))
        self.max_new_tokens = int(settings.get('max_new_tokens', self.max_new_tokens))
        self.context_length = int(settings.get('context_length', self.context_length))
        self.gpu_layers = int(settings.get('gpu_layers', self.gpu_layers))
        self.repeat_penalty = float(settings.get('repeat_penalty', self.repeat_penalty))
        
        global m, device
        target_dev = 'cuda' if (self.gpu_layers > 0 and torch.cuda.is_available()) else 'cpu'
        if target_dev != device:
            device = target_dev
            m = self.model.to(device)

    def execute_tool(self, tool_name, kwargs):
        try:
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
                            except:
                                pass
                return "\n".join(results) if results else "No matches found."

            elif tool_name == 'run_shell_command':
                cmd = kwargs.get('command', '')
                res = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=30, cwd=os.getcwd())
                output = res.stdout if res.returncode == 0 else res.stderr
                return output[:2000] if output else "Command executed with no output."

            else:
                return f"Unknown tool: {tool_name}"
        except Exception as e:
            return f"Error executing tool {tool_name}: {str(e)}"

    @torch.no_grad()
    def solve_with_agent(self, prompt):
        self.model.eval()
        
        persona_instructions = {
            "ashen_ai_agent": "You are Ashen AI, an advanced cybernetic local AI assistant with tool execution capabilities.\n",
            "code_architect": "You are Ashen AI Code Architect, specializing in high-performance PyTorch, Python, and web engineering.\n",
            "cyber_companion": "You are Ashen AI Companion, a razor-sharp cyberpunk assistant with a witty, direct edge.\n"
        }
        
        base_inst = persona_instructions.get(self.persona, persona_instructions["ashen_ai_agent"])
        workspace_context_block = ""
        if self.workspace_context:
            workspace_context_block = f"\n### Current Workspace Context\n{self.workspace_context}\n\n"

        system_instructions = (
            base_inst +
            workspace_context_block +
            "Available tools:\n"
            "- read_file(file_path='...')\n"
            "- write_file(file_path='...', content='...')\n"
            "- glob(pattern='...')\n"
            "- grep_search(pattern='...')\n"
            "- run_shell_command(command='...')\n"
            "To use a tool, output: [TOOL: tool_name(arg1=val1, arg2=val2)]\n"
            "After observing tool output, continue reasoning until you give your final answer.\n\n"
        )

        conversation_context = system_instructions
        for h_user, h_resp in self.history[-2:]:
            conversation_context += f"### Instruction:\n{h_user}\n\n### Response:\n{h_resp}\n\n"

        current_prompt = f"{conversation_context}### Instruction:\n{prompt}\n\n### Response:\n<think>\n"
        
        all_thoughts = []
        tool_observations = []
        final_answer = ""

        for step in range(self.max_steps):
            encoded = self.encode(current_prompt)
            if len(encoded) > self.context_length:
                encoded = encoded[-self.context_length:]
            input_ids = torch.tensor([encoded], dtype=torch.long, device=device)

            output_ids = self.model.generate(
                input_ids, 
                max_new_tokens=self.max_new_tokens, 
                current_block_size=self.context_length, 
                temperature=self.temperature, 
                top_k=self.top_k
            )
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
                for arg_match in re.finditer(r'([a-zA-Z_][a-zA-Z0-9_]*)\s*=\s*(["\'])(.*?)\2', args_str):
                    kwargs[arg_match.group(1)] = arg_match.group(3)

                tool_obs = self.execute_tool(tool_name, kwargs)
                tool_observations.append(f"Tool: {tool_name}({args_str})\nObservation:\n{tool_obs}")

                current_prompt += f"{remainder}\n[OBSERVATION]:\n{tool_obs}\n<think>\n"
            else:
                final_answer = remainder
                break

        if not final_answer:
            final_answer = remainder

        if user_wants_code(prompt):
            clean_final = final_answer
        else:
            clean_final = filter_code_output(final_answer)

        combined_thought = "\n--- Ashen AI Reasoning Step ---\n".join(all_thoughts)
        if tool_observations:
            combined_thought += "\n\n--- Ashen AI Tool Telemetry ---\n" + "\n".join(tool_observations)

        self.history.append((prompt, f"<think>\n{combined_thought}\n</think>\n{clean_final}"))

        return combined_thought, clean_final

reasoner = AshenAIAgenticEngine(m, decode, encode, device)

# Track which session the reasoner belongs to
reasoner.session_id = None

HTML_PAGE = r"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Ashen AI - Cybernetic Local AI Hub</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <script src="https://cdn.jsdelivr.net/npm/marked/marked.js"></script>
    <!-- Prism.js for Syntax Highlighting -->
    <link href="https://cdnjs.cloudflare.com/ajax/libs/prism/1.29.0/themes/prism-okaidia.min.css" rel="stylesheet" />
    <script src="https://cdnjs.cloudflare.com/ajax/libs/prism/1.29.0/prism.min.js"></script>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/prism/1.29.0/components/prism-python.min.js"></script>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/prism/1.29.0/components/prism-json.min.js"></script>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/prism/1.29.0/components/prism-bash.min.js"></script>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/prism/1.29.0/components/prism-markdown.min.js"></script>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/prism/1.29.0/components/prism-javascript.min.js"></script>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/prism/1.29.0/components/prism-typescript.min.js"></script>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/prism/1.29.0/components/prism-css.min.js"></script>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/prism/1.29.0/components/prism-yaml.min.js"></script>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/prism/1.29.0/components/prism-markup.min.js"></script>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/prism/1.29.0/components/prism-go.min.js"></script>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/prism/1.29.0/components/prism-rust.min.js"></script>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/prism/1.29.0/components/prism-c.min.js"></script>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/prism/1.29.0/components/prism-cpp.min.js"></script>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/prism/1.29.0/components/prism-java.min.js"></script>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/prism/1.29.0/components/prism-csharp.min.js"></script>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/prism/1.29.0/components/prism-ruby.min.js"></script>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/prism/1.29.0/components/prism-php.min.js"></script>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/prism/1.29.0/components/prism-sql.min.js"></script>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/prism/1.29.0/components/prism-xml.min.js"></script>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/prism/1.29.0/components/prism-toml.min.js"></script>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/prism/1.29.0/components/prism-lua.min.js"></script>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/prism/1.29.0/components/prism-swift.min.js"></script>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/prism/1.29.0/components/prism-kotlin.min.js"></script>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/prism/1.29.0/components/prism-powershell.min.js"></script>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/prism/1.29.0/components/prism-docker.min.js"></script>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/prism/1.29.0/components/prism-git.min.js"></script>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/prism/1.29.0/components/prism-regex.min.js"></script>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/prism/1.29.0/components/prism-hcl.min.js"></script>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/prism/1.29.0/components/prism-protobuf.min.js"></script>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/prism/1.29.0/components/prism-graphql.min.js"></script>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/prism/1.29.0/components/prism-ini.min.js"></script>
    <link href="https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;700&amp;display=swap" rel="stylesheet">
    <style>
        body { background-color: #050509; color: #e2e8f0; font-family: 'JetBrains Mono', monospace; }
        .scanline {
            background: linear-gradient(to bottom, rgba(255,255,255,0), rgba(255,255,255,0) 50%, rgba(0, 0, 0, 0.3) 50%, rgba(0, 0, 0, 0.3));
            background-size: 100% 4px;
            pointer-events: none;
        }
        .chat-container::-webkit-scrollbar { width: 6px; }
        .chat-container::-webkit-scrollbar-thumb { background: #1e293b; border-radius: 3px; }
        .neon-border { box-shadow: 0 0 15px rgba(99, 102, 241, 0.15); }
    </style>
</head>
<body class="h-screen flex flex-col relative overflow-hidden">
    <!-- Scanlines Overlay -->
    <div id="scanline-overlay" class="absolute inset-0 scanline z-50 opacity-40"></div>

    <!-- Header -->
    <header class="bg-slate-950 border-b border-indigo-900/40 px-6 py-3 flex items-center justify-between shadow-2xl z-10">
        <div class="flex items-center space-x-3">
            <div class="w-3 h-3 bg-cyan-400 rounded-full animate-ping"></div>
            <h1 class="text-lg font-bold tracking-wider text-cyan-400">ASHEN AI <span class="text-xs px-2 py-0.5 bg-indigo-950 text-indigo-300 rounded border border-indigo-700/50">LOCAL AI HUB</span></h1>
        </div>
        <div class="flex items-center space-x-3 text-xs">
            <select id="persona-select" onchange="changePersona(this.value)" class="bg-slate-900 text-indigo-300 border border-indigo-800/60 rounded px-2 py-1 focus:outline-none">
                <option value="ashen_ai_agent">Persona: Ashen AI Agent</option>
                <option value="code_architect">Persona: Code Architect</option>
                <option value="cyber_companion">Persona: Cyber Companion</option>
            </select>
            <button onclick="createNewSession()" class="px-2.5 py-1 bg-violet-950/60 hover:bg-violet-900/60 text-violet-300 rounded border border-violet-800/60 transition" title="Start a new chat session">📝 New Chat</button>
            <button onclick="toggleSessionsPanel(true)" class="px-2.5 py-1 bg-blue-950/60 hover:bg-blue-900/60 text-blue-300 rounded border border-blue-800/60 transition" title="View past chat sessions">💬 Sessions</button>
            <button onclick="toggleModelModal(true)" class="px-2.5 py-1 bg-cyan-950/60 hover:bg-cyan-900/60 text-cyan-300 rounded border border-cyan-800/60 transition">Model Hub</button>
            <button onclick="loadWorkspaceDir('')" class="px-2.5 py-1 bg-emerald-950/60 hover:bg-emerald-900/60 text-emerald-300 rounded border border-emerald-800/60 transition">📁 Workspace</button>
            <button onclick="toggleSettingsModal(true)" class="px-2.5 py-1 bg-indigo-950/60 hover:bg-indigo-900/60 text-indigo-300 rounded border border-indigo-800/60 transition">⚙️ Settings</button>
            <button onclick="toggleScanlines()" id="scanline-btn" class="px-2.5 py-1 bg-slate-900 hover:bg-slate-800 text-slate-300 rounded border border-slate-700 transition">CRT FX: ON</button>
            <button onclick="clearHistory()" class="px-2.5 py-1 bg-red-950/60 hover:bg-red-900/60 text-red-300 rounded border border-red-800/60 transition">Purge</button>
            <button onclick="exportChat()" class="px-2.5 py-1 bg-slate-900 hover:bg-slate-800 text-indigo-300 rounded border border-indigo-800/60 transition">Export</button>
        </div>
    </header>

    <!-- Model Hub Modal -->
    <div id="model-modal" class="fixed inset-0 bg-slate-950/80 backdrop-blur-sm z-50 hidden flex items-center justify-center p-4">
        <div class="bg-slate-900 border border-cyan-500/50 rounded-xl w-full max-w-4xl p-6 shadow-2xl space-y-6 max-h-[90vh] overflow-y-auto">
            <div class="flex justify-between items-center border-b border-slate-800 pb-3">
                <h2 class="text-base font-bold text-cyan-400">Ashen AI Model Hub &amp; Checkpoint Manager</h2>
                <button onclick="toggleModelModal(false)" class="text-slate-400 hover:text-white font-bold">✕</button>
            </div>
            
            <div class="space-y-6">
                <!-- Tab Selector -->
                <div class="flex space-x-2 border-b border-slate-800 pb-2 justify-between items-center">
                    <div class="flex space-x-2">
                        <button onclick="switchTab('local')" id="tab-local" class="px-3 py-1.5 rounded-lg text-xs font-semibold bg-cyan-950 text-cyan-300 border border-cyan-800/60 transition">Local Checkpoints</button>
                        <button onclick="switchTab('hf')" id="tab-hf" class="px-3 py-1.5 rounded-lg text-xs font-semibold bg-slate-800 text-slate-400 hover:text-white transition">Hugging Face Hub (All Models)</button>
                        <button onclick="switchTab('upload')" id="tab-upload" class="px-3 py-1.5 rounded-lg text-xs font-semibold bg-slate-800 text-slate-400 hover:text-white transition">Upload Model</button>
                    </div>
                    <button onclick="scanPcModels()" class="px-3 py-1.5 bg-indigo-950 hover:bg-indigo-900 text-indigo-200 rounded-lg text-xs font-semibold border border-indigo-700/60 transition">🔍 Scan PC for Models</button>
                </div>

                <!-- Local Tab -->
                <div id="section-local" class="space-y-3">
                    <h3 class="text-xs font-semibold text-indigo-300 uppercase tracking-wider">Installed Checkpoints &amp; GGUF Files</h3>
                    <div id="model-list" class="bg-slate-950 border border-slate-800 rounded-lg p-3 max-h-48 overflow-y-auto space-y-2 text-xs font-mono">
                        <div class="text-slate-500">Scanning local checkpoints...</div>
                    </div>
                </div>

                <!-- Hugging Face Tab -->
                <div id="section-hf" class="space-y-3 hidden">
                    <div class="flex space-x-2">
                        <input type="text" id="hf-search-input" placeholder="Search any Hugging Face model (e.g. Qwen, Llama, DeepSeek, GGUF)..." class="flex-1 bg-slate-950 border border-slate-700 rounded-lg p-2.5 text-xs text-cyan-100 placeholder-slate-500 focus:outline-none focus:border-cyan-500 font-mono">
                        <button onclick="searchHfModels()" class="px-4 py-2 bg-cyan-600 hover:bg-cyan-500 text-slate-950 font-bold text-xs rounded-lg transition shrink-0">SEARCH HF</button>
                    </div>
                    <div id="hf-model-list" class="bg-slate-950 border border-slate-800 rounded-lg p-3 max-h-72 overflow-y-auto space-y-3 text-xs">
                        <div class="text-slate-500">Type a query or search to list Hugging Face models...</div>
                    </div>
                </div>

                <!-- Upload Tab -->
                <div id="section-upload" class="space-y-3 hidden">
                    <h3 class="text-xs font-semibold text-indigo-300 uppercase tracking-wider">Upload Custom Model / GGUF File</h3>
                    <div class="flex items-center space-x-3">
                        <input type="file" id="model-file-input" accept=".pk1,.pt,.pth,.gguf" class="block w-full text-xs text-slate-400 file:mr-4 file:py-2 file:px-4 file:rounded-lg file:border-0 file:text-xs file:font-semibold file:bg-cyan-950 file:text-cyan-300 hover:file:bg-cyan-900 cursor-pointer border border-slate-700 rounded-lg bg-slate-950">
                        <button onclick="uploadModel()" class="bg-cyan-600 hover:bg-cyan-500 text-slate-950 px-4 py-2 rounded-lg font-bold text-xs shrink-0 transition">UPLOAD</button>
                    </div>
                </div>

                <div id="upload-status" class="text-xs text-cyan-400 font-mono"></div>
            </div>
        </div>
    </div>

    <!-- Model Settings Modal (LM Studio Style) -->
    <div id="settings-modal" class="fixed inset-0 bg-slate-950/80 backdrop-blur-sm z-50 hidden flex items-center justify-center p-4">
        <div class="bg-slate-900 border border-indigo-500/50 rounded-xl w-full max-w-2xl p-6 shadow-2xl space-y-6 max-h-[90vh] overflow-y-auto">
            <div class="flex justify-between items-center border-b border-slate-800 pb-3">
                <h2 class="text-base font-bold text-indigo-300">⚙️ LM Studio-Style Model &amp; Inference Settings</h2>
                <button onclick="toggleSettingsModal(false)" class="text-slate-400 hover:text-white font-bold">✕</button>
            </div>

            <div class="space-y-4 text-xs">
                <!-- Temperature -->
                <div class="space-y-1">
                    <div class="flex justify-between">
                        <span class="text-slate-300 font-medium">Temperature</span>
                        <span id="temp-val" class="text-cyan-400 font-mono">0.7</span>
                    </div>
                    <input type="range" id="setting-temp" min="0.0" max="2.0" step="0.05" value="0.7" oninput="document.getElementById('temp-val').textContent=this.value" class="w-full accent-cyan-500 bg-slate-950">
                </div>

                <!-- Top-K -->
                <div class="space-y-1">
                    <div class="flex justify-between">
                        <span class="text-slate-300 font-medium">Top-K</span>
                        <span id="topk-val" class="text-cyan-400 font-mono">40</span>
                    </div>
                    <input type="range" id="setting-topk" min="1" max="100" step="1" value="40" oninput="document.getElementById('topk-val').textContent=this.value" class="w-full accent-cyan-500 bg-slate-950">
                </div>

                <!-- Top-P -->
                <div class="space-y-1">
                    <div class="flex justify-between">
                        <span class="text-slate-300 font-medium">Top-P (Nucleus Sampling)</span>
                        <span id="topp-val" class="text-cyan-400 font-mono">0.9</span>
                    </div>
                    <input type="range" id="setting-topp" min="0.0" max="1.0" step="0.05" value="0.9" oninput="document.getElementById('topp-val').textContent=this.value" class="w-full accent-cyan-500 bg-slate-950">
                </div>

                <!-- Max Output Tokens -->
                <div class="space-y-1">
                    <div class="flex justify-between">
                        <span class="text-slate-300 font-medium">Max Output Tokens</span>
                        <span id="tokens-val" class="text-cyan-400 font-mono">250</span>
                    </div>
                    <input type="range" id="setting-tokens" min="50" max="1024" step="25" value="250" oninput="document.getElementById('tokens-val').textContent=this.value" class="w-full accent-cyan-500 bg-slate-950">
                </div>

                <!-- Context Length -->
                <div class="space-y-1">
                    <span class="text-slate-300 font-medium">Context Length (Tokens)</span>
                    <select id="setting-context" class="w-full bg-slate-950 border border-slate-700 rounded-lg p-2 text-xs text-cyan-200">
                        <option value="2048">2,048 Tokens</option>
                        <option value="4096">4,096 Tokens</option>
                        <option value="8192" selected>8,192 Tokens</option>
                        <option value="16384">16,384 Tokens</option>
                        <option value="32768">32,768 Tokens</option>
                    </select>
                </div>

                <!-- GPU Offload Layers Slider -->
                <div class="space-y-1 pt-2 border-t border-slate-800">
                    <div class="flex justify-between">
                        <div>
                            <span class="text-slate-300 font-medium">GPU Offload Layers (CUDA VRAM)</span>
                            <div class="text-slate-500 text-[10px]">Number of transformer layers offloaded to GPU</div>
                        </div>
                        <span id="gpu-layers-val" class="text-cyan-400 font-mono">16</span>
                    </div>
                    <input type="range" id="setting-gpu-layers" min="0" max="32" step="1" value="16" oninput="document.getElementById('gpu-layers-val').textContent=this.value" class="w-full accent-cyan-500 bg-slate-950">
                </div>

                <div class="pt-4 flex justify-end space-x-3 border-t border-slate-800">
                    <button onclick="toggleSettingsModal(false)" class="px-4 py-2 bg-slate-800 hover:bg-slate-700 text-slate-300 rounded-lg font-bold transition">CANCEL</button>
                    <button onclick="saveSettings()" class="px-5 py-2 bg-indigo-600 hover:bg-indigo-500 text-white rounded-lg font-bold transition">SAVE SETTINGS</button>
                </div>
                <div id="settings-status" class="text-xs text-emerald-400 font-mono text-center"></div>
            </div>
        </div>
    </div>

    <!-- Sessions Panel -->
    <div id="sessions-panel" class="fixed inset-y-0 left-0 w-80 bg-slate-950/95 backdrop-blur-sm z-40 border-r border-blue-900/40 transform -translate-x-full transition-transform duration-300 flex flex-col">
        <div class="p-4 border-b border-blue-900/30 flex items-center justify-between">
            <h2 class="text-sm font-bold text-blue-400 uppercase tracking-wider">💬 Chat Sessions</h2>
            <button onclick="toggleSessionsPanel(false)" class="text-slate-400 hover:text-white font-bold text-lg">✕</button>
        </div>

        <!-- New Session Button -->
        <div class="p-3 border-b border-slate-800">
            <button onclick="createNewSession()" class="w-full px-3 py-2 bg-violet-950/60 hover:bg-violet-900/60 text-violet-300 rounded-lg text-xs font-semibold border border-violet-800/60 transition flex items-center justify-center gap-2">
                <span>📝</span> New Chat Session
            </button>
        </div>

        <!-- Session List -->
        <div id="session-list" class="flex-1 overflow-y-auto p-2 space-y-1">
            <div class="text-slate-500 text-xs text-center py-4">Loading sessions...</div>
        </div>

        <!-- Current Session Info -->
        <div class="p-3 border-t border-slate-800 text-[10px] text-slate-500 text-center">
            <span id="current-session-label">No active session</span>
        </div>
    </div>

    <!-- Sessions Panel Overlay -->
    <div id="sessions-overlay" class="fixed inset-0 bg-black/30 z-30 hidden" onclick="toggleSessionsPanel(false)"></div>

    <!-- Main Workspace -->
    <div class="flex-1 flex overflow-hidden z-10">
        <!-- Sidebar Telemetry & Quick Actions -->
        <aside class="w-80 bg-slate-950/80 border-r border-indigo-900/30 p-4 hidden md:flex flex-col justify-between">
            <div class="space-y-4">
                <h2 class="text-xs font-semibold text-cyan-500 uppercase tracking-widest">System Telemetry</h2>
                <div class="bg-slate-900/70 p-3 rounded-lg border border-slate-800 space-y-2 text-xs">
                    <div class="flex justify-between"><span>Model:</span><span class="text-indigo-400 font-bold" id="active-model-name">ashen_gpt_model.pk1</span></div>
                    <div class="flex justify-between"><span>Backend:</span><span class="text-emerald-400 font-bold">PyTorch CUDA</span></div>
                    <div class="flex justify-between"><span>Context Window:</span><span class="text-cyan-400 font-bold" id="sidebar-ctx">8,192 Tokens</span></div>
                    <div class="flex justify-between"><span>GPU Layers:</span><span class="text-amber-400 font-bold" id="sidebar-gpu">16 / 32</span></div>
                    <div class="flex flex-col gap-1 pt-1 border-t border-slate-800 mt-1">
                        <span class="text-slate-500">Workspace Context:</span>
                        <span class="text-emerald-300 font-bold truncate" id="workspace-context-label" title="Currently browsed directory injected into model prompt">📁 None</span>
                    </div>
                </div>

                <h2 class="text-xs font-semibold text-cyan-500 uppercase tracking-widest pt-2">Ashen AI Quick Actions</h2>
                <div class="grid grid-cols-2 gap-2">
                    <button onclick="sendQuickPrompt('Run pytest suite')" class="p-2 text-left text-xs bg-slate-900 hover:bg-indigo-950/50 text-indigo-300 rounded border border-indigo-900/50 transition">⚡ Run Tests</button>
                    <button onclick="sendQuickPrompt('Check git status and diff')" class="p-2 text-left text-xs bg-slate-900 hover:bg-indigo-950/50 text-indigo-300 rounded border border-indigo-900/50 transition">📦 Git Status</button>
                    <button onclick="sendQuickPrompt('List workspace files using glob')" class="p-2 text-left text-xs bg-slate-900 hover:bg-indigo-950/50 text-indigo-300 rounded border border-indigo-900/50 transition">🔍 File Glob</button>
                    <button onclick="sendQuickPrompt('Search codebase for reasoning engine')" class="p-2 text-left text-xs bg-slate-900 hover:bg-indigo-950/50 text-indigo-300 rounded border border-indigo-900/50 transition">🔎 Grep Search</button>
                </div>
            </div>

            <div class="text-[10px] text-slate-500 text-center pt-4 border-t border-slate-900">
                Ashen AI Hub &bull; Autonomous Agent Core
            </div>
        </aside>

        <!-- Chat Area -->
        <main class="flex-1 flex flex-col bg-[#08080f]">
            <div id="chat-container" class="flex-1 overflow-y-auto p-6 space-y-6 chat-container">
                <!-- Welcome Message -->
                <div class="flex items-start space-x-4 max-w-4xl">
                    <div class="w-8 h-8 rounded bg-cyan-600/20 border border-cyan-500 flex items-center justify-center font-bold text-xs text-cyan-400 shrink-0">Ω</div>
                    <div class="bg-slate-900/90 border border-indigo-900/60 rounded-xl p-4 text-slate-200 text-sm shadow-xl space-y-2 neon-border">
                        <p class="font-bold text-cyan-400">Ashen AI Cybernetic Core Online.</p>
                        <p class="text-xs text-slate-300">Initialized with Ashen GPT ~127M MoE model and full agentic tool suite. Access Model Hub, Workspace, or Settings above or enter your prompt below.</p>
                    </div>
                </div>
            </div>

            <!-- Input Bar -->
            <div class="p-4 bg-slate-950 border-t border-indigo-900/40">
                <div class="max-w-4xl mx-auto flex items-end space-x-3">
                    <div class="flex-1 bg-slate-900 border border-indigo-900/60 rounded-lg focus-within:border-cyan-500 transition">
                        <textarea id="user-input" rows="1" placeholder="Enter command or query for Ashen AI... (Shift+Enter for newline)" class="w-full bg-transparent p-3 text-cyan-100 placeholder-slate-500 text-xs focus:outline-none resize-none max-h-32"></textarea>
                    </div>
                    <button id="send-btn" onclick="sendMessage()" class="bg-cyan-600 hover:bg-cyan-500 text-slate-950 px-5 py-3 rounded-lg font-bold text-xs transition shadow-lg shrink-0 flex items-center space-x-2">
                        <span>EXECUTE</span>
                        <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M14 5l7 7m0 0l-7 7m7-7H3"></path></svg>
                    </button>
                </div>
            </div>
        </main>

        <!-- Right Workspace Explorer & Code Editor Panel (Always Open) -->
        <aside class="w-[480px] xl:w-[520px] bg-slate-950/90 border-l border-emerald-900/40 p-4 hidden lg:flex flex-col space-y-3 overflow-hidden shrink-0">
            <div class="flex justify-between items-center border-b border-slate-800 pb-2">
                <div class="flex items-center space-x-2">
                    <h2 class="text-xs font-bold text-emerald-400 uppercase tracking-wider">📁 Workspace Explorer</h2>
                </div>
                <div class="flex items-center space-x-2 text-xs">
                    <button onclick="setEditorMode('view')" id="mode-view-btn" class="px-2.5 py-1 rounded bg-emerald-950 text-emerald-300 border border-emerald-800">View</button>
                    <button onclick="setEditorMode('edit')" id="mode-edit-btn" class="px-2.5 py-1 rounded bg-slate-800 text-slate-400">Edit</button>
                    <button onclick="saveCurrentFile()" class="px-2.5 py-1 bg-emerald-600 hover:bg-emerald-500 text-slate-950 rounded font-bold transition">SAVE</button>
                </div>
            </div>

            <div class="flex items-center justify-between text-[11px] font-mono text-slate-400">
                <span id="current-dir-label" class="truncate">~ (Home)</span>
                <span id="current-file-path" class="text-cyan-300 truncate max-w-[200px]" title="No file selected">No file selected</span>
            </div>

            <!-- File Tree Sidebar -->
            <div class="h-44 bg-slate-900 border border-slate-800 rounded-lg p-2 overflow-y-auto space-y-1 font-mono text-xs flex flex-col shrink-0" id="file-tree">
                <div class="text-slate-500">Loading home directory...</div>
            </div>

            <!-- Code Editor Area -->
            <div class="flex-1 flex flex-col bg-slate-900 border border-slate-800 rounded-lg p-3 space-y-2 overflow-hidden">
                <!-- Syntax Highlighted Code Editor Container (View & Edit Mode) -->
                <div class="flex-1 relative bg-[#1a1a24] rounded-lg overflow-auto font-mono text-xs border border-slate-800" id="editor-container">
                    <pre class="p-3 m-0 bg-transparent pointer-events-none w-max min-w-full"><code id="highlighted-code" class="language-python"># Select a file to view syntax-highlighted code</code></pre>
                    <textarea id="code-editor" class="absolute inset-0 w-full h-full bg-transparent text-transparent caret-emerald-400 p-3 font-mono text-xs focus:outline-none resize-none leading-relaxed overflow-hidden" placeholder="Edit file code here..." readonly oninput="onEditorInput()"></textarea>
                </div>

                <div id="editor-status" class="text-xs text-emerald-400 font-mono"></div>
            </div>
        </aside>
    </div>

    <script>
        const textarea = document.getElementById('user-input');
        textarea.addEventListener('input', function() {
            this.style.height = 'auto';
            this.style.height = (this.scrollHeight) + 'px';
        });

        textarea.addEventListener('keydown', function(e) {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                sendMessage();
            }
        });

        function toggleModelModal(show) {
            const modal = document.getElementById('model-modal');
            modal.style.display = show ? 'flex' : 'none';
            if (show) {
                loadModelsList();
                searchHfModels();
            }
        }

        function toggleSettingsModal(show) {
            const modal = document.getElementById('settings-modal');
            modal.style.display = show ? 'flex' : 'none';
        }

        function toggleSessionsPanel(show) {
            const panel = document.getElementById('sessions-panel');
            const overlay = document.getElementById('sessions-overlay');
            panel.style.transform = show ? 'translateX(0)' : '-translate-x-full';
            overlay.style.display = show ? 'block' : 'none';
            if (show) loadSessionList();
        }

        // Render helpers for loading session history
        function renderUserMessage(msg) {
            return `
                <div class="flex items-start space-x-4 max-w-4xl">
                    <div class="w-8 h-8 rounded bg-slate-700/50 border border-slate-600 flex items-center justify-center font-bold text-xs text-slate-300 shrink-0 mt-1">U</div>
                    <div class="bg-slate-900/90 border border-slate-700/60 rounded-xl p-4 text-slate-200 text-sm shadow-xl"><p>${msg.replace(/</g,'&lt;').replace(/>/g,'&gt;')}</p></div>
                </div>`;
        }

        function renderAssistantResponse(content) {
            // Strip think tags for display, extract thought vs final answer
            let cleanContent = content;
            let thoughtText = '';
            const thinkMatch = content.match(/(<think>)([\s\S]*?)(<\/think>)/);
            if (thinkMatch) {
                thoughtText = thinkMatch[1].trim();
                cleanContent = content.substring(thinkMatch.index + thinkMatch[0].length).trim();
            }
            return `
                <div class="flex items-start space-x-4 max-w-4xl">
                    <div class="w-8 h-8 rounded bg-cyan-600/20 border border-cyan-500 flex items-center justify-center font-bold text-xs text-cyan-400 shrink-0 mt-1">Ω</div>
                    <div class="bg-slate-900/90 border border-indigo-900/60 rounded-xl p-4 text-slate-200 text-sm shadow-xl space-y-2">
                        ${thoughtText ? `<details class="text-[10px] text-slate-500"><summary>Ashen AI Reasoning</summary><pre class="whitespace-pre-wrap mt-1">${thoughtText.replace(/</g,'&lt;').replace(/>/g,'&gt;')}</pre></details>` : ''}
                        <div class="prose prose-invert prose-sm max-w-none">${cleanContent.replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/\n/g,'<br>')}</div>
                    </div>
                </div>`;
        }

        let currentDirPath = '';
        let currentEditingFile = '';
        let editorMode = 'view'; // 'view' or 'edit'

        function onEditorInput() {
            const editor = document.getElementById('code-editor');
            const codeEl = document.getElementById('highlighted-code');
            codeEl.textContent = editor.value;
            Prism.highlightElement(codeEl);
        }

        function setEditorMode(mode) {
            editorMode = mode;
            const editor = document.getElementById('code-editor');
            const viewBtn = document.getElementById('mode-view-btn');
            const editBtn = document.getElementById('mode-edit-btn');

            if (mode === 'view') {
                editor.readOnly = true;
                viewBtn.className = 'px-3 py-1 rounded bg-emerald-950 text-emerald-300 border border-emerald-800';
                editBtn.className = 'px-3 py-1 rounded bg-slate-800 text-slate-400';
            } else {
                editor.readOnly = false;
                viewBtn.className = 'px-3 py-1 rounded bg-slate-800 text-slate-400';
                editBtn.className = 'px-3 py-1 rounded bg-emerald-950 text-emerald-300 border border-emerald-800';
                editor.focus();
            }
        }

        async function loadWorkspaceDir(dirPath) {
            const treeEl = document.getElementById('file-tree');
            treeEl.innerHTML = '<div class="text-slate-500">Loading directory...</div>';
            try {
                const res = await fetch(`/api/workspace/list?dir=${encodeURIComponent(dirPath)}`);
                const data = await res.json();
                currentDirPath = data.current_path;
                document.getElementById('current-dir-label').textContent = currentDirPath;
                treeEl.innerHTML = '';

                if (data.parent_path) {
                    const parentItem = document.createElement('div');
                    parentItem.className = 'p-2 rounded hover:bg-slate-900 cursor-pointer text-amber-300 font-bold transition';
                    parentItem.textContent = '📁 .. (Parent Directory)';
                    parentItem.onclick = () => loadWorkspaceDir(data.parent_path);
                    treeEl.appendChild(parentItem);
                }

                data.items.forEach(item => {
                    const el = document.createElement('div');
                    el.className = `p-2 rounded hover:bg-slate-900 cursor-pointer transition flex items-center justify-between ${item.is_dir ? 'text-cyan-400 font-bold' : 'text-slate-300'}`;
                    el.innerHTML = `<span>${item.is_dir ? '📁 ' : '📄 '}${item.name}</span>`;
                    if (item.is_dir) {
                        el.onclick = () => loadWorkspaceDir(item.path);
                    } else {
                        el.onclick = () => openFile(item.path);
                    }
                    treeEl.appendChild(el);
                });

                // Push this workspace directory into model context and update UI
                const ctxLabel = document.getElementById('workspace-context-label');
                const dirName = dirPath || '(Root)';
                if (ctxLabel) {
                    ctxLabel.textContent = `📁 ${dirName}`;
                    ctxLabel.title = dirName;
                }
                try {
                    await fetch('/api/workspace/context', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ dir: dirPath })
                    });
                } catch (e) { /* non-blocking — don't break UI if fails */ }
            } catch (err) {
                treeEl.innerHTML = '<div class="text-red-400">Failed to load directory.</div>';
            }
        }

        async function openFile(filepath) {
            currentEditingFile = filepath;
            document.getElementById('current-file-path').textContent = filepath;
            const editor = document.getElementById('code-editor');
            const codeEl = document.getElementById('highlighted-code');
            editor.value = 'Loading...';
            codeEl.textContent = 'Loading...';

            const ext = filepath.split('.').pop().toLowerCase();
            const langMap = {
                'py': 'language-python',
                'json': 'language-json',
                'js': 'language-javascript',
                'ts': 'language-typescript',
                'html': 'language-markup',
                'htm': 'language-markup',
                'css': 'language-css',
                'md': 'language-markdown',
                'sh': 'language-bash',
                'bat': 'language-powershell',
                'ps1': 'language-powershell',
                'yml': 'language-yaml',
                'yaml': 'language-yaml',
                'go': 'language-go',
                'rs': 'language-rust',
                'c': 'language-c',
                'h': 'language-c',
                'cpp': 'language-cpp',
                'cc': 'language-cpp',
                'cxx': 'language-cpp',
                'hpp': 'language-cpp',
                'java': 'language-java',
                'cs': 'language-csharp',
                'rb': 'language-ruby',
                'php': 'language-php',
                'sql': 'language-sql',
                'xml': 'language-xml',
                'toml': 'language-toml',
                'lua': 'language-lua',
                'swift': 'language-swift',
                'kt': 'language-kotlin',
                'kts': 'language-kotlin',
                'dockerfile': 'language-docker',
                'dockerignore': 'language-docker',
                'gitignore': 'language-git',
                'gitconfig': 'language-git',
                'regex': 'language-regex',
                're': 'language-regex',
                'ini': 'language-ini',
                'cfg': 'language-ini',
                'env': 'language-bash',
                'tf': 'language-hcl',
                'hcl': 'language-hcl',
                'proto': 'language-protobuf',
                'graphql': 'language-graphql',
                'vue': 'language-markup',
                'svelte': 'language-markup',
                'scss': 'language-css',
                'sass': 'language-css',
                'jsx': 'language-javascript',
                'tsx': 'language-typescript',
            };
            codeEl.className = langMap[ext] || 'language-python';

            try {
                console.log('[DEBUG] Opening file:', filepath);
                const res = await fetch('/api/workspace/read', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ path: filepath })
                });
                console.log('[DEBUG] Response status:', res.status);
                const data = await res.json();
                console.log('[DEBUG] Response:', data);
                if (data.status === 'success') {
                    const editor = document.getElementById('code-editor');
                    const codeEl = document.getElementById('highlighted-code');
                    console.log('[DEBUG] editor:', !!editor, 'codeEl:', !!codeEl, 'content length:', data.content?.length || 0);
                    
                    if (editor) {
                        editor.value = data.content;
                    }
                    
                    if (codeEl) {
                        // Delay highlighting to ensure Prism and CSS are ready
                        setTimeout(() => {
                            codeEl.textContent = data.content;
                            if (typeof Prism !== 'undefined' && Prism.highlightElement) {
                                Prism.highlightElement(codeEl);
                                console.log('[DEBUG] Highlighting applied to', codeEl.className);
                            } else {
                                codeEl.textContent = data.content;
                                console.warn('[DEBUG] Prism not available');
                            }
                        }, 50);
                    }
                    if (!editor && !codeEl) {
                        console.error('[DEBUG] ERROR: code-editor or highlighted-code element not found!');
                    }
                } else {
                    console.error('[DEBUG] Server returned error:', data.message);
                    const editor = document.getElementById('code-editor');
                    if (editor) {
                        editor.value = 'Error: ' + data.message;
                    }
                }
            } catch (err) {
                console.error('[DEBUG] Exception:', err);
                const editor = document.getElementById('code-editor');
                if (editor) {
                    editor.value = 'Failed to load file content: ' + err.message;
                }
            }
        }

        async function saveCurrentFile() {
            if (!currentEditingFile) return;
            const content = document.getElementById('code-editor').value;
            const statusEl = document.getElementById('editor-status');
            statusEl.textContent = `Saving ${currentEditingFile}...`;

            try {
                const res = await fetch('/api/workspace/write', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ path: currentEditingFile, content })
                });
                const data = await res.json();
                if (data.status === 'success') {
                    statusEl.textContent = `Successfully saved ${currentEditingFile}!`;
                    setTimeout(() => statusEl.textContent = '', 3000);
                } else {
                    statusEl.textContent = `Error: ${data.message}`;
                }
            } catch (err) {
                statusEl.textContent = 'Save failed.';
            }
        }

        async function saveSettings() {
            const settings = {
                temperature: parseFloat(document.getElementById('setting-temp').value),
                top_k: parseInt(document.getElementById('setting-topk').value),
                top_p: parseFloat(document.getElementById('setting-topp').value),
                max_new_tokens: parseInt(document.getElementById('setting-tokens').value),
                context_length: parseInt(document.getElementById('setting-context').value),
                gpu_layers: parseInt(document.getElementById('setting-gpu-layers').value)
            };
            const statusEl = document.getElementById('settings-status');
            statusEl.textContent = 'Updating settings...';

            try {
                const res = await fetch('/api/settings', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(settings)
                });
                const data = await res.json();
                if (data.status === 'success') {
                    statusEl.textContent = 'Settings updated successfully!';
                    document.getElementById('sidebar-gpu').textContent = settings.gpu_layers + ' Layers';
                    document.getElementById('sidebar-ctx').textContent = settings.context_length.toLocaleString() + ' Tokens';
                    setTimeout(() => toggleSettingsModal(false), 1000);
                } else {
                    statusEl.textContent = 'Error saving settings.';
                }
            } catch (err) {
                statusEl.textContent = 'Failed to save settings.';
            }
        }

        function switchTab(tab) {
            ['local', 'hf', 'upload'].forEach(t => {
                document.getElementById(`section-${t}`).classList.add('hidden');
                document.getElementById(`tab-${t}`).className = 'px-3 py-1.5 rounded-lg text-xs font-semibold bg-slate-800 text-slate-400 hover:text-white transition';
            });
            document.getElementById(`section-${tab}`).classList.remove('hidden');
            document.getElementById(`tab-${tab}`).className = 'px-3 py-1.5 rounded-lg text-xs font-semibold bg-cyan-950 text-cyan-300 border border-cyan-800/60 transition';
        }

        async function loadModelsList() {
            const listEl = document.getElementById('model-list');
            listEl.innerHTML = '<div class="text-slate-500">Scanning local checkpoints &amp; GGUF files...</div>';
            try {
                const res = await fetch('/api/models');
                const data = await res.json();
                listEl.innerHTML = '';
                if (data.models.length === 0) {
                    listEl.innerHTML = '<div class="text-slate-500">No checkpoints found. Upload or download one!</div>';
                    return;
                }
                data.models.forEach(m => {
                    const item = document.createElement('div');
                    item.className = 'flex justify-between items-center bg-slate-900 p-2.5 rounded border border-slate-800';
                    item.innerHTML = `
                        <div>
                            <span class="text-cyan-300 font-bold">${m.name}</span>
                            <span class="text-slate-500 ml-2">(${m.size_mb} MB)</span>
                            <span class="text-slate-600 text-[10px] ml-2 block">${m.path}</span>
                            ${m.active ? '<span class="mt-1 inline-block px-1.5 py-0.5 bg-emerald-950 text-emerald-300 text-[10px] rounded border border-emerald-800">ACTIVE</span>' : ''}
                        </div>
                        ${!m.active ? `<button onclick="switchModel('${m.name}')" class="px-2.5 py-1 bg-indigo-900 hover:bg-indigo-800 text-indigo-200 rounded text-[11px] transition">ACTIVATE</button>` : ''}
                    `;
                    listEl.appendChild(item);
                });
            } catch (err) {
                listEl.innerHTML = '<div class="text-red-400">Failed to load local checkpoints.</div>';
            }
        }

        async function scanPcModels() {
            const statusEl = document.getElementById('upload-status');
            statusEl.textContent = 'Scanning PC & Hugging Face cache for .pk1 / .gguf files...';
            try {
                const res = await fetch('/api/models/scan-pc', { method: 'POST' });
                const data = await res.json();
                statusEl.textContent = `Scan complete! Found ${data.found} model file(s).`;
                loadModelsList();
            } catch (err) {
                statusEl.textContent = 'PC scan failed.';
            }
        }

        async function searchHfModels() {
            const query = document.getElementById('hf-search-input').value.trim();
            const hfListEl = document.getElementById('hf-model-list');
            hfListEl.innerHTML = '<div class="text-slate-500">Querying Hugging Face Hub API...</div>';
            try {
                const res = await fetch(`/api/hf-search?q=${encodeURIComponent(query)}`);
                const data = await res.json();
                hfListEl.innerHTML = '';
                if (data.models.length === 0) {
                    hfListEl.innerHTML = '<div class="text-slate-500">No Hugging Face models found matching query.</div>';
                    return;
                }
                data.models.forEach(m => {
                    const item = document.createElement('div');
                    item.className = 'bg-slate-900 p-3 rounded-lg border border-slate-800 flex justify-between items-center';
                    item.innerHTML = `
                        <div>
                            <div class="text-cyan-300 font-bold">${m.repo_id}</div>
                            <div class="text-slate-400 text-[11px]">Downloads: ${m.downloads} &bull; Likes: ${m.likes}</div>
                        </div>
                        <button onclick="downloadHfRepo('${m.repo_id}')" class="px-3 py-1.5 bg-cyan-600 hover:bg-cyan-500 text-slate-950 rounded font-bold text-xs transition shrink-0">PULL MODEL</button>
                    `;
                    hfListEl.appendChild(item);
                });
            } catch (err) {
                hfListEl.innerHTML = '<div class="text-red-400">Failed to search Hugging Face Hub.</div>';
            }
        }

        async function downloadHfRepo(repo_id) {
            const statusEl = document.getElementById('upload-status');
            statusEl.textContent = `Pulling repository ${repo_id} from Hugging Face...`;
            try {
                const res = await fetch('/api/models/download-hf-repo', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ repo_id })
                });
                const data = await res.json();
                if (data.status === 'success') {
                    statusEl.textContent = `Successfully pulled ${repo_id}!`;
                    switchTab('local');
                    loadModelsList();
                } else {
                    statusEl.textContent = `Pull error: ${data.message}`;
                }
            } catch (err) {
                statusEl.textContent = 'Pull failed.';
            }
        }

        async function switchModel(filename) {
            const statusEl = document.getElementById('upload-status');
            statusEl.textContent = `Switching active model to ${filename}...`;
            try {
                const res = await fetch('/api/models/switch', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ filename })
                });
                const data = await res.json();
                if (data.status === 'success') {
                    document.getElementById('active-model-name').textContent = filename;
                    statusEl.textContent = `Successfully activated ${filename}!`;
                    loadModelsList();
                } else {
                    statusEl.textContent = `Error: ${data.message}`;
                }
            } catch (err) {
                statusEl.textContent = 'Failed to switch model.';
            }
        }

        async function uploadModel() {
            const fileInput = document.getElementById('model-file-input');
            const statusEl = document.getElementById('upload-status');
            if (!fileInput.files.length) {
                statusEl.textContent = 'Please select a checkpoint or GGUF file.';
                return;
            }
            const file = fileInput.files[0];
            statusEl.textContent = `Uploading ${file.name} (${(file.size/1024/1024).toFixed(1)} MB)...`;

            const reader = new FileReader();
            reader.onload = async function(e) {
                const base64Data = e.target.result.split(',')[1];
                try {
                    const res = await fetch('/api/models/upload', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ filename: file.name, content_base64: base64Data })
                    });
                    const data = await res.json();
                    if (data.status === 'success') {
                        statusEl.textContent = `Successfully uploaded and saved ${file.name}!`;
                        fileInput.value = '';
                        switchTab('local');
                        loadModelsList();
                    } else {
                        statusEl.textContent = `Upload error: ${data.message}`;
                    }
                } catch (err) {
                    statusEl.textContent = 'Upload failed.';
                }
            };
            reader.readAsDataURL(file);
        }

        function toggleScanlines() {
            const overlay = document.getElementById('scanline-overlay');
            const btn = document.getElementById('scanline-btn');
            if (overlay.style.display === 'none') {
                overlay.style.display = 'block';
                btn.textContent = 'CRT FX: ON';
            } else {
                overlay.style.display = 'none';
                btn.textContent = 'CRT FX: OFF';
            }
        }

        async function changePersona(persona) {
            await fetch('/api/persona', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ persona })
            });
        }

        function sendQuickPrompt(promptText) {
            document.getElementById('user-input').value = promptText;
            sendMessage();
        }

        function appendMessage(sender, thought, text) {
            const container = document.getElementById('chat-container');
            const isUser = sender === 'user';
            
            const messageDiv = document.createElement('div');
            messageDiv.className = `flex items-start space-x-4 max-w-4xl ${isUser ? 'ml-auto flex-row-reverse space-x-reverse' : ''}`;
            
            const avatar = document.createElement('div');
            avatar.className = `w-8 h-8 rounded flex items-center justify-center font-bold text-xs shrink-0 ${isUser ? 'bg-emerald-600/20 border border-emerald-500 text-emerald-400' : 'bg-cyan-600/20 border border-cyan-500 text-cyan-400'}`;
            avatar.textContent = isUser ? 'U' : 'Ω';
            
            const contentDiv = document.createElement('div');
            contentDiv.className = `rounded-xl p-4 text-xs shadow-lg space-y-3 ${isUser ? 'bg-emerald-950/40 border border-emerald-800/60 text-emerald-100' : 'bg-slate-900/90 border border-indigo-900/60 text-slate-200 w-full'}`;
            
            if (!isUser && thought) {
                const thinkDetails = document.createElement('details');
                thinkDetails.className = 'group bg-slate-950/80 rounded-lg border border-indigo-900/60 overflow-hidden';
                
                const summary = document.createElement('summary');
                summary.className = 'px-3 py-2 text-[11px] font-medium text-cyan-400 cursor-pointer select-none hover:bg-slate-900 flex items-center justify-between';
                summary.innerHTML = '<span>⚡ Ashen AI Telemetry &amp; Tool Trace</span><span class="text-cyan-500 group-open:rotate-180 transition-transform">▼</span>';
                
                const thinkBody = document.createElement('div');
                thinkBody.className = 'p-3 text-[11px] text-slate-400 font-mono whitespace-pre-wrap border-t border-indigo-900/50 bg-[#030307]';
                thinkBody.textContent = thought;
                
                thinkDetails.appendChild(summary);
                thinkDetails.appendChild(thinkBody);
                contentDiv.appendChild(thinkDetails);
            }
            
            const textBody = document.createElement('div');
            textBody.className = 'prose prose-invert max-w-none text-slate-200 text-xs';
            if (isUser) {
                textBody.textContent = text;
            } else {
                textBody.innerHTML = marked.parse(text);
            }
            contentDiv.appendChild(textBody);
            
            messageDiv.appendChild(avatar);
            messageDiv.appendChild(contentDiv);
            container.appendChild(messageDiv);
            container.scrollTop = container.scrollHeight;
        }

        async function sendMessage() {
            const input = document.getElementById('user-input');
            const msg = input.value.trim();
            if (!msg) return;

            input.value = '';
            input.style.height = 'auto';

            appendMessage('user', '', msg);

            const sendBtn = document.getElementById('send-btn');
            sendBtn.disabled = true;
            sendBtn.textContent = 'PROCESSING...';

            try {
                const res = await fetch('/api/chat', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ message: msg })
                });
                const data = await res.json();
                appendMessage('assistant', data.thought, data.response);
            } catch (err) {
                appendMessage('assistant', 'Error', 'Failed to communicate with Ashen AI core.');
            } finally {
                sendBtn.disabled = false;
                sendBtn.textContent = 'EXECUTE';
            }
        }

        async function clearHistory() {
            await fetch('/api/clear', { method: 'POST' });
            // Reset workspace context indicator
            const ctxLabel = document.getElementById('workspace-context-label');
            if (ctxLabel) {
                ctxLabel.textContent = '📁 None';
                ctxLabel.title = 'No workspace context active';
            }
            const container = document.getElementById('chat-container');
            container.innerHTML = `
                <div class="flex items-start space-x-4 max-w-4xl">
                    <div class="w-8 h-8 rounded bg-cyan-600/20 border border-cyan-500 flex items-center justify-center font-bold text-xs text-cyan-400 shrink-0">Ω</div>
                    <div class="bg-slate-900/90 border border-indigo-900/60 rounded-xl p-4 text-slate-200 text-sm shadow-xl space-y-2">
                        <p class="font-bold text-cyan-400">Session Purged.</p>
                        <p class="text-xs text-slate-300">Memory reset successfully. Ashen AI core ready for new directives.</p>
                    </div>
                </div>
            `;
        }

        function exportChat() {
            const container = document.getElementById('chat-container');
            let textContent = "# Ashen AI Hub - Session Export\\n\\n";
            container.querySelectorAll('.flex.items-start').forEach(el => {
                const isUser = el.classList.contains('ml-auto');
                const role = isUser ? "User" : "Ashen AI";
                const text = el.querySelector('.prose')?.innerText || "";
                textContent += `### ${role}\\n${text}\\n\\n`;
            });
            const blob = new Blob([textContent], { type: 'text/markdown' });
            const url = URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = `ashen_ai_session_${Date.now()}.md`;
            a.click();
        }

        // --- Session Management ---
        async function loadSessionList() {
            try {
                const res = await fetch('/api/sessions');
                const data = await res.json();
                const listEl = document.getElementById('session-list');
                const labelEl = document.getElementById('current-session-label');

                if (!listEl) return;

                if (data.sessions && data.sessions.length > 0) {
                    listEl.innerHTML = '';
                    data.sessions.forEach(s => {
                        const item = document.createElement('div');
                        const isCurrent = s.id === data.current;
                        const dateStr = s.updated ? new Date(s.updated).toLocaleString() : '';
                        item.className = `p-2 rounded cursor-pointer transition flex items-center justify-between ${isCurrent ? 'bg-blue-950/40 border border-blue-800/60' : 'bg-slate-900/50 hover:bg-slate-800 border border-transparent'}`;
                        item.innerHTML = `
                            <div class="flex-1 min-w-0">
                                <div class="text-xs font-semibold ${isCurrent ? 'text-blue-300' : 'text-slate-300'} truncate">${s.name || 'Untitled'}</div>
                                <div class="text-[10px] text-slate-500">${s.message_count} msgs · ${dateStr}</div>
                            </div>
                            <div class="flex gap-1 ml-2 shrink-0">
                                ${!isCurrent ? `<button onclick="event.stopPropagation(); loadSession('${s.id}')" class="text-[10px] px-1.5 py-0.5 bg-blue-950 hover:bg-blue-900 text-blue-300 rounded border border-blue-800/40">Load</button>` : '<span class="text-[10px] text-blue-400">● Active</span>'}
                                <button onclick="event.stopPropagation(); renameSessionPrompt('${s.id}')" class="text-[10px] px-1.5 py-0.5 bg-slate-800 hover:bg-slate-700 text-slate-300 rounded">✎</button>
                                <button onclick="event.stopPropagation(); deleteSessionConfirm('${s.id}')" class="text-[10px] px-1.5 py-0.5 bg-red-950 hover:bg-red-900 text-red-300 rounded">×</button>
                            </div>`;
                        item.onclick = () => loadSession(s.id);
                        listEl.appendChild(item);
                    });
                } else {
                    listEl.innerHTML = '<div class="text-slate-500 text-xs text-center py-4">No past sessions</div>';
                }

                if (labelEl && data.current) {
                    const currentData = data.sessions.find(s => s.id === data.current);
                    labelEl.textContent = currentData ? `Active: ${currentData.name}` : 'No active session';
                }
            } catch (e) { /* non-blocking */ }
        }

        async function createNewSession() {
            try {
                const res = await fetch('/api/sessions/new', { method: 'POST' });
                const data = await res.json();
                if (data.status === 'created') {
                    // Clear chat UI
                    const container = document.getElementById('chat-container');
                    container.innerHTML = `
                        <div class="flex items-start space-x-4 max-w-4xl">
                            <div class="w-8 h-8 rounded bg-cyan-600/20 border border-cyan-500 flex items-center justify-center font-bold text-xs text-cyan-400 shrink-0">Ω</div>
                            <div class="bg-slate-900/90 border border-indigo-900/60 rounded-xl p-4 text-slate-200 text-sm shadow-xl space-y-2">
                                <p class="font-bold text-cyan-400">New Chat Started</p>
                                <p class="text-xs text-slate-300">Fresh session ready for your next prompt.</p>
                            </div>
                        </div>`;
                    // Reset workspace context indicator
                    const ctxLabel = document.getElementById('workspace-context-label');
                    if (ctxLabel) { ctxLabel.textContent = '📁 None'; ctxLabel.title = 'No workspace context active'; }
                    toggleSessionsPanel(false);
                    loadSessionList();
                }
            } catch (e) { alert('Failed to create session.'); }
        }

        async function loadSession(sid) {
            try {
                const res = await fetch('/api/sessions/load', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ session_id: sid })
                });
                const data = await res.json();
                if (data.status === 'loaded') {
                    // Restore chat history in UI
                    const container = document.getElementById('chat-container');
                    const messages = data.session.history || [];
                    let html = '';
                    if (messages.length === 0) {
                        html = `<div class="flex items-start space-x-4 max-w-4xl">
                            <div class="w-8 h-8 rounded bg-cyan-600/20 border border-cyan-500 flex items-center justify-center font-bold text-xs text-cyan-400 shrink-0">Ω</div>
                            <div class="bg-slate-900/90 border border-indigo-900/60 rounded-xl p-4 text-slate-200 text-sm shadow-xl">
                                <p class="font-bold text-cyan-400">Loaded "${data.session.name}"</p>
                                <p class="text-xs text-slate-300 mt-1">This session has no messages yet.</p>
                            </div></div>`;
                    } else {
                        messages.forEach(m => {
                            html += renderUserMessage(m.user);
                            const respContent = m.assistant || '';
                            html += renderAssistantResponse(respContent);
                        });
                    }
                    container.innerHTML = html;
                    container.scrollTop = container.scrollHeight;
                    toggleSessionsPanel(false);
                    loadSessionList();
                }
            } catch (e) { alert('Failed to load session.'); }
        }

        async function deleteSessionConfirm(sid) {
            const session = list_sessions_cache?.find(s => s.id === sid);
            const name = session ? session.name : 'this session';
            if (!confirm(`Delete session "${name}"? This cannot be undone.`)) return;
            try {
                const res = await fetch('/api/sessions/delete', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ session_id: sid })
                });
                const data = await res.json();
                if (data.status === 'deleted') {
                    const container = document.getElementById('chat-container');
                    container.innerHTML = `<div class="flex items-center justify-center text-red-400 text-xs">Session deleted.</div>`;
                    const ctxLabel = document.getElementById('workspace-context-label');
                    if (ctxLabel) { ctxLabel.textContent = '📁 None'; ctxLabel.title = 'No workspace context active'; }
                    loadSessionList();
                }
            } catch (e) { alert('Failed to delete session.'); }
        }

        async function renameSessionPrompt(sid) {
            const session = list_sessions_cache?.find(s => s.id === sid);
            const name = session ? session.name : 'Untitled';
            const newName = prompt('Rename session:', name);
            if (!newName || newName.trim() === '') return;
            try {
                await fetch('/api/sessions/rename', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ session_id: sid, name: newName.trim() })
                });
                loadSessionList();
            } catch (e) { alert('Failed to rename session.'); }
        }

        let list_sessions_cache = null;
        // Cache & periodically refresh session list
        loadSessionList();
        setInterval(loadSessionList, 30000);

        // Initialize workspace file tree on page load
        loadWorkspaceDir('');
    </script>
</body>
</html>
"""

def scan_local_pc_models():
    found_models = []
    search_dirs = [
        '.',
        os.path.expanduser('~/.cache/huggingface/hub'),
        os.path.expanduser('~/.lmstudio/models'),
        os.path.expanduser('~/Downloads')
    ]
    for d in search_dirs:
        if os.path.exists(d):
            try:
                for root, dirs, files in os.walk(d):
                    for file in files:
                        if file.endswith(('.pk1', '.pt', '.pth', '.gguf')):
                            full_path = os.path.abspath(os.path.join(root, file))
                            size_mb = round(os.path.getsize(full_path) / (1024 * 1024), 1)
                            found_models.append({
                                'name': file,
                                'path': full_path,
                                'size_mb': size_mb,
                                'active': (file == current_model_filename)
                            })
            except:
                pass
    return found_models

def get_directory_listing(dir_path):
    if not dir_path:
        target_dir = os.getcwd()
    else:
        target_dir = dir_path

    target_dir = os.path.abspath(target_dir)
    items = []
    parent_path = None

    try:
        parent = os.path.dirname(target_dir)
        if parent != target_dir:
            parent_path = parent

        for entry in os.scandir(target_dir):
            if entry.name.startswith('.'):
                continue
            is_dir = entry.is_dir()
            items.append({
                'name': entry.name,
                'path': entry.path.replace('\\', '/'),  # Normalize to forward slashes for URLs
                'is_dir': is_dir
            })
        items.sort(key=lambda x: (not x['is_dir'], x['name'].lower()))
    except Exception as e:
        pass

    return {
        'current_path': target_dir,
        'parent_path': parent_path,
        'items': items
    }

# --- Session Management Helpers ---

def _session_file(session_id):
    return os.path.join(SESSIONS_DIR, f"{session_id}.json")

def save_session(session_id, data):
    """Save a full session (history + workspace context + settings + persona) to disk."""
    filepath = _session_file(session_id)
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

def load_session(session_id):
    """Load a session from disk. Returns dict or None."""
    filepath = _session_file(session_id)
    if not os.path.exists(filepath):
        return None
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)

def list_sessions():
    """Return a list of all sessions sorted by date (newest first)."""
    sessions = []
    for fname in os.listdir(SESSIONS_DIR):
        if not fname.endswith('.json'):
            continue
        sid = fname[:-5]  # strip .json
        filepath = os.path.join(SESSIONS_DIR, fname)
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
            sessions.append({
                'id': sid,
                'name': data.get('name', 'Untitled'),
                'created': data.get('created_at', ''),
                'updated': data.get('updated_at', ''),
                'message_count': len(data.get('history', [])),
                'persona': data.get('persona', 'ashen_ai_agent'),
                'size_bytes': os.path.getsize(filepath)
            })
        except Exception:
            pass
    sessions.sort(key=lambda s: s.get('updated', ''), reverse=True)
    return sessions

def delete_session(session_id):
    """Delete a session file. Returns True if deleted."""
    filepath = _session_file(session_id)
    if os.path.exists(filepath):
        os.remove(filepath)
        return True
    return False

def create_new_session():
    """Create a new session ID and return it."""
    global current_session_id
    ts = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    current_session_id = f"session_{ts}"
    # Save initial blank session
    save_session(current_session_id, {
        'name': 'Untitled',
        'created_at': datetime.datetime.now().isoformat(),
        'updated_at': datetime.datetime.now().isoformat(),
        'persona': 'ashen_ai_agent',
        'settings': {
            'temperature': 0.7,
            'top_k': 40,
            'top_p': 0.9,
            'max_new_tokens': 250,
            'context_length': 8192,
            'gpu_layers': 16,
            'repeat_penalty': 1.1
        },
        'history': [],
        'workspace_context': ''
    })
    return current_session_id

def append_to_session(session_id, user_msg, assistant_msg):
    """Append a message pair to the session's history."""
    global current_session_id
    data = load_session(session_id)
    if not data:
        return
    data['history'].append({'user': user_msg, 'assistant': assistant_msg})
    data['updated_at'] = datetime.datetime.now().isoformat()
    save_session(session_id, data)
    current_session_id = session_id

class ChatHandler(http.server.SimpleHTTPRequestHandler):
    def do_GET(self):
        parsed_url = urlparse(self.path)
        path = parsed_url.path
        query = parse_qs(parsed_url.query)

        if path == '/' or path == '':
            self.send_response(200)
            self.send_header('Content-Type', 'text/html; charset=utf-8')
            self.end_headers()
            self.wfile.write(HTML_PAGE.encode('utf-8'))
        elif path == '/api/models':
            models = scan_local_pc_models()
            seen = set()
            unique_models = []
            for m in models:
                if m['path'] not in seen:
                    seen.add(m['path'])
                    unique_models.append(m)
            self.send_response(200)
            self.send_header('Content-Type', 'application/json; charset=utf-8')
            self.end_headers()
            self.wfile.write(json.dumps({'models': unique_models}).encode('utf-8'))
        elif path == '/api/hf-search':
            q = query.get('q', [''])[0]
            hf_models = []
            try:
                url = f"https://huggingface.co/api/models?search={q}&limit=20&sort=downloads&direction=-1"
                resp = requests.get(url, timeout=10)
                data = resp.json()
                for item in data:
                    hf_models.append({
                        'repo_id': item.get('id', ''),
                        'downloads': item.get('downloads', 0),
                        'likes': item.get('likes', 0)
                    })
            except Exception as e:
                pass
            self.send_response(200)
            self.send_header('Content-Type', 'application/json; charset=utf-8')
            self.end_headers()
            self.wfile.write(json.dumps({'models': hf_models}).encode('utf-8'))
        elif path == '/api/sessions':
            sessions = list_sessions()
            self.send_response(200)
            self.send_header('Content-Type', 'application/json; charset=utf-8')
            self.end_headers()
            self.wfile.write(json.dumps({'sessions': sessions, 'current': current_session_id}).encode('utf-8'))
        elif path == '/api/workspace/list':
            dir_path = query.get('dir', [''])[0]
            listing = get_directory_listing(dir_path)
            self.send_response(200)
            self.send_header('Content-Type', 'application/json; charset=utf-8')
            self.end_headers()
            self.wfile.write(json.dumps(listing).encode('utf-8'))
        else:
            self.send_response(404)
            self.end_headers()

    def do_POST(self):
        global model, reasoner, current_model_filename, current_session_id
        
        if self.path == '/api/workspace/read':
            content_length = int(self.headers.get('Content-Length', 0))
            body = self.rfile.read(content_length)
            data = json.loads(body.decode('utf-8'))
            file_path = data.get('path', '')
            
            # Debug logging
            import sys
            print(f"[DEBUG] Read request for: {repr(file_path)}", file=sys.stderr, flush=True)
            print(f"[DEBUG] cwd: {repr(os.getcwd())}", file=sys.stderr, flush=True)
            
            if not file_path:
                resp_data = {'status': 'error', 'message': 'No file path provided'}
            else:
                target_path = None
                
                # Strategy 1: Use path as-is (should be absolute)
                exists1 = os.path.isfile(file_path)
                print(f"[DEBUG] Strategy 1 (as-is): exists={exists1}", file=sys.stderr, flush=True)
                
                if exists1:
                    target_path = file_path
                
                # Strategy 2: Try backslash normalization (C:/path -> C:\path)
                elif '/' in file_path:
                    win_path = file_path.replace('/', '\\')
                    exists2 = os.path.isfile(win_path)
                    print(f"[DEBUG] Strategy 2 (backslash): {repr(win_path)} exists={exists2}", file=sys.stderr, flush=True)
                    if exists2:
                        target_path = win_path
                
                # Strategy 3: Join with cwd for relative paths
                if not target_path:
                    joined = os.path.join(os.getcwd(), file_path)
                    exists3 = os.path.isfile(joined)
                    print(f"[DEBUG] Strategy 3 (joined): {repr(joined)} exists={exists3}", file=sys.stderr, flush=True)
                    if exists3:
                        target_path = joined
                
                if target_path and os.path.exists(target_path) and os.path.isfile(target_path):
                    try:
                        with open(target_path, 'rb') as f:
                            raw_bytes = f.read()
                        try:
                            content = raw_bytes.decode('utf-8')
                        except UnicodeDecodeError:
                            content = raw_bytes.decode('latin-1')
                        print(f"[DEBUG] SUCCESS: read {len(raw_bytes)} bytes from {target_path}", file=sys.stderr, flush=True)
                        resp_data = {'status': 'success', 'content': content}
                    except Exception as e:
                        print(f"[DEBUG] ERROR reading file: {e}", file=sys.stderr, flush=True)
                        resp_data = {'status': 'error', 'message': str(e)}
                else:
                    resp_data = {'status': 'error', 'message': f'File not found: {repr(file_path)}. Tried all strategies.'}
            
            self.send_response(200)
            self.send_header('Content-Type', 'application/json; charset=utf-8')
            self.end_headers()
            self.wfile.write(json.dumps(resp_data).encode('utf-8'))
            return
        
        if self.path == '/api/chat':
            content_length = int(self.headers.get('Content-Length', 0))
            body = self.rfile.read(content_length)
            data = json.loads(body.decode('utf-8'))
            message = data.get('message', '')

            # Auto-create session if none active
            if not current_session_id:
                current_session_id = create_new_session()
                reasoner.session_id = current_session_id
                # Auto-name session from first message
                sdata = load_session(current_session_id)
                if sdata and sdata.get('name', 'Untitled') == 'Untitled':
                    preview = message[:50].replace('\n', ' ')
                    sdata['name'] = preview + ('...' if len(message) > 50 else '')
                    save_session(current_session_id, sdata)

            try:
                thought, response = reasoner.solve_with_agent(message)
                # Save to session
                append_to_session(current_session_id, message, f"<think>\n{thought}\n</think>\n{response}")
                resp_data = {'thought': thought, 'response': response}
            except Exception as e:
                resp_data = {'thought': 'Error during Ashen AI execution', 'response': str(e)}

            self.send_response(200)
            self.send_header('Content-Type', 'application/json; charset=utf-8')
            self.end_headers()
            self.wfile.write(json.dumps(resp_data).encode('utf-8'))

        elif self.path == '/api/clear':
            # Save current session before clearing
            if current_session_id and load_session(current_session_id):
                save_session(current_session_id, {
                    'name': load_session(current_session_id)['name'],
                    'persona': reasoner.persona,
                    'settings': {
                        'temperature': reasoner.temperature,
                        'top_k': reasoner.top_k,
                        'top_p': reasoner.top_p,
                        'max_new_tokens': reasoner.max_new_tokens,
                        'context_length': reasoner.context_length,
                        'gpu_layers': reasoner.gpu_layers,
                        'repeat_penalty': reasoner.repeat_penalty
                    },
                    'history': list(reasoner.history),
                    'workspace_context': reasoner.workspace_context
                })
            reasoner.clear_history()
            reasoner.set_workspace_context("")
            self.send_response(200)
            self.send_header('Content-Type', 'application/json; charset=utf-8')
            self.end_headers()
            self.wfile.write(json.dumps({'status': 'cleared'}).encode('utf-8'))

        elif self.path == '/api/persona':
            content_length = int(self.headers.get('Content-Length', 0))
            body = self.rfile.read(content_length)
            data = json.loads(body.decode('utf-8'))
            persona = data.get('persona', 'ashen_ai_agent')
            reasoner.set_persona(persona)
            self.send_response(200)
            self.send_header('Content-Type', 'application/json; charset=utf-8')
            self.end_headers()
            self.wfile.write(json.dumps({'status': 'updated', 'persona': persona}).encode('utf-8'))

        elif self.path == '/api/settings':
            content_length = int(self.headers.get('Content-Length', 0))
            body = self.rfile.read(content_length)
            data = json.loads(body.decode('utf-8'))
            reasoner.update_settings(data)
            self.send_response(200)
            self.send_header('Content-Type', 'application/json; charset=utf-8')
            self.end_headers()
            self.wfile.write(json.dumps({'status': 'success'}).encode('utf-8'))

        # --- Session Management Endpoints ---
        elif self.path == '/api/sessions/new':
            sid = create_new_session()
            self.send_response(201)
            self.send_header('Content-Type', 'application/json; charset=utf-8')
            self.end_headers()
            self.wfile.write(json.dumps({'status': 'created', 'session_id': sid}).encode('utf-8'))

        elif self.path == '/api/sessions':
            sessions = list_sessions()
            self.send_response(200)
            self.send_header('Content-Type', 'application/json; charset=utf-8')
            self.end_headers()
            self.wfile.write(json.dumps({'sessions': sessions, 'current': current_session_id}).encode('utf-8'))

        elif self.path == '/api/sessions/load':
            content_length = int(self.headers.get('Content-Length', 0))
            body = self.rfile.read(content_length)
            data = json.loads(body.decode('utf-8'))
            sid = data.get('session_id', '')
            session_data = load_session(sid)
            if session_data:
                # Restore reasoner state from session
                current_session_id = sid
                reasoner.session_id = sid
                reasoner.persona = session_data.get('persona', 'ashen_ai_agent')
                reasoner.history = list(session_data.get('history', []))
                ws_ctx = session_data.get('workspace_context', '')
                if ws_ctx:
                    reasoner.set_workspace_context(ws_ctx)
                resp = {'status': 'loaded', 'session': session_data}
            else:
                resp = {'status': 'error', 'message': 'Session not found'}
            self.send_response(200)
            self.send_header('Content-Type', 'application/json; charset=utf-8')
            self.end_headers()
            self.wfile.write(json.dumps(resp).encode('utf-8'))

        elif self.path == '/api/sessions/delete':
            content_length = int(self.headers.get('Content-Length', 0))
            body = self.rfile.read(content_length)
            data = json.loads(body.decode('utf-8'))
            sid = data.get('session_id', '')
            if delete_session(sid):
                if current_session_id == sid:
                    current_session_id = None
                    reasoner.session_id = None
                    reasoner.clear_history()
                    reasoner.set_workspace_context("")
                self.send_response(200)
                self.send_header('Content-Type', 'application/json; charset=utf-8')
                self.end_headers()
                self.wfile.write(json.dumps({'status': 'deleted'}).encode('utf-8'))
            else:
                self.send_response(404)
                self.send_header('Content-Type', 'application/json; charset=utf-8')
                self.end_headers()
                self.wfile.write(json.dumps({'status': 'error', 'message': 'Session not found'}).encode('utf-8'))

        elif self.path == '/api/sessions/rename':
            content_length = int(self.headers.get('Content-Length', 0))
            body = self.rfile.read(content_length)
            data = json.loads(body.decode('utf-8'))
            sid = data.get('session_id', '')
            name = data.get('name', 'Untitled')
            sdata = load_session(sid)
            if sdata:
                sdata['name'] = name
                save_session(sid, sdata)
                self.send_response(200)
                self.send_header('Content-Type', 'application/json; charset=utf-8')
                self.end_headers()
                self.wfile.write(json.dumps({'status': 'renamed'}).encode('utf-8'))
            else:
                self.send_response(404)
                self.send_header('Content-Type', 'application/json; charset=utf-8')
                self.end_headers()
                self.wfile.write(json.dumps({'status': 'error', 'message': 'Session not found'}).encode('utf-8'))

        elif self.path == '/api/workspace/context':
            # Push current workspace directory listing into model context
            content_length = int(self.headers.get('Content-Length', 0))
            body = self.rfile.read(content_length)
            data = json.loads(body.decode('utf-8'))
            workspace_dir = data.get('dir', '')
            listing = get_directory_listing(workspace_dir)
            # Build a compact text representation of the workspace
            context_lines = [f"Directory: {listing['current_path']}"]
            for item in listing['items']:
                prefix = "📁 " if item['is_dir'] else "📄 "
                context_lines.append(f"  {prefix}{item['name']}")
            reasoner.set_workspace_context("\n".join(context_lines))
            self.send_response(200)
            self.send_header('Content-Type', 'application/json; charset=utf-8')
            self.end_headers()
            self.wfile.write(json.dumps({'status': 'success', 'path': listing['current_path']}).encode('utf-8'))

        elif self.path == '/api/models/scan-pc':
            models = scan_local_pc_models()
            self.send_response(200)
            self.send_header('Content-Type', 'application/json; charset=utf-8')
            self.end_headers()
            self.wfile.write(json.dumps({'status': 'success', 'found': len(models)}).encode('utf-8'))

        elif self.path == '/api/models/download-hf-repo':
            content_length = int(self.headers.get('Content-Length', 0))
            body = self.rfile.read(content_length)
            data = json.loads(body.decode('utf-8'))
            repo_id = data.get('repo_id', '')

            try:
                from huggingface_hub import snapshot_download
                snapshot_download(repo_id=repo_id, local_dir=repo_id.replace('/', '_'))
                resp_data = {'status': 'success', 'repo_id': repo_id}
            except Exception as e:
                resp_data = {'status': 'error', 'message': str(e)}

            self.send_response(200)
            self.send_header('Content-Type', 'application/json; charset=utf-8')
            self.end_headers()
            self.wfile.write(json.dumps(resp_data).encode('utf-8'))

        elif self.path == '/api/models/switch':
            content_length = int(self.headers.get('Content-Length', 0))
            body = self.rfile.read(content_length)
            data = json.loads(body.decode('utf-8'))
            filename = data.get('filename', '')

            if filename.endswith('.gguf') or '/' in filename or '\\' in filename:
                current_model_filename = filename
                resp_data = {'status': 'success', 'filename': filename}
            elif os.path.exists(filename):
                try:
                    with open(filename, 'rb') as f:
                        model = pickle.load(f)
                    model = model.to(device)
                    reasoner.model = model
                    current_model_filename = filename
                    resp_data = {'status': 'success', 'filename': filename}
                except Exception as e:
                    resp_data = {'status': 'error', 'message': str(e)}
            else:
                resp_data = {'status': 'error', 'message': 'File not found'}

            self.send_response(200)
            self.send_header('Content-Type', 'application/json; charset=utf-8')
            self.end_headers()
            self.wfile.write(json.dumps(resp_data).encode('utf-8'))

        elif self.path == '/api/models/upload':
            content_length = int(self.headers.get('Content-Length', 0))
            body = self.rfile.read(content_length)
            data = json.loads(body.decode('utf-8'))
            filename = data.get('filename', '')
            content_base64 = data.get('content_base64', '')

            if filename and content_base64:
                try:
                    file_bytes = base64.b64decode(content_base64)
                    safe_name = os.path.basename(filename)
                    with open(safe_name, 'wb') as f:
                        f.write(file_bytes)
                    resp_data = {'status': 'success', 'filename': safe_name}
                except Exception as e:
                    resp_data = {'status': 'error', 'message': str(e)}
            else:
                resp_data = {'status': 'error', 'message': 'Invalid payload'}

            self.send_response(200)
            self.send_header('Content-Type', 'application/json; charset=utf-8')
            self.end_headers()
            self.wfile.write(json.dumps(resp_data).encode('utf-8'))

        elif self.path == '/api/workspace/write':
            content_length = int(self.headers.get('Content-Length', 0))
            body = self.rfile.read(content_length)
            data = json.loads(body.decode('utf-8'))
            file_path = data.get('path', '')
            content = data.get('content', '')

            if file_path:
                try:
                    with open(file_path, 'w', encoding='utf-8') as f:
                        f.write(content)
                    resp_data = {'status': 'success'}
                except Exception as e:
                    resp_data = {'status': 'error', 'message': str(e)}
            else:
                resp_data = {'status': 'error', 'message': 'Invalid file path'}

            self.send_response(200)
            self.send_header('Content-Type', 'application/json; charset=utf-8')
            self.end_headers()
            self.wfile.write(json.dumps(resp_data).encode('utf-8'))

        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, format, *args):
        return

def run_server(port=5000):
    server = socketserver.TCPServer(('127.0.0.1', port), ChatHandler)
    print(f"\n========================================================")
    print(f" Ashen AI Cybernetic Hub running at: http://localhost:{port}")
    print(f" Open your browser to experience the Ashen AI interface!")
    print(f"========================================================\n")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down Ashen AI server...")
        server.server_close()

if __name__ == '__main__':
    run_server(5000)
