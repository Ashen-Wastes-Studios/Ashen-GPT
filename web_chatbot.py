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

model_path = 'ashen_gpt_model.pk1'
if os.path.exists(model_path):
    print(f"Loading Ashen GPT model parameters from {model_path}...")
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    print("Model loaded successfully!")
else:
    print(f"No checkpoint found at {model_path}. Initializing new Ashen GPT model...")
    model = AshenGPTLanguageModel(vocab_size)

m = model.to(device)

class AgenticReasoningEngine:
    def __init__(self, model, decode_fn, encode_fn, device, max_steps=5):
        self.model = model
        self.decode = decode_fn
        self.encode = encode_fn
        self.device = device
        self.max_steps = max_steps
        self.history = []

    def clear_history(self):
        self.history = []

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
    def solve_with_agent(self, prompt, max_new_tokens=250):
        self.model.eval()
        
        system_instructions = (
            "You are Ashen GPT, an autonomous AI agent with tool execution capabilities.\n"
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
            if len(encoded) > block_size:
                encoded = encoded[-block_size:]
            input_ids = torch.tensor([encoded], dtype=torch.long, device=self.device)

            output_ids = self.model.generate(input_ids, max_new_tokens=max_new_tokens, current_block_size=block_size, temperature=0.6, top_k=40)
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
                thought_process = "Executing agent step..."
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

        combined_thought = "\n--- Reasoning Step ---\n".join(all_thoughts)
        if tool_observations:
            combined_thought += "\n\n--- Tool Observations ---\n" + "\n".join(tool_observations)

        self.history.append((prompt, f"<think>\n{combined_thought}\n</think>\n{clean_final}"))

        return combined_thought, clean_final

reasoner = AgenticReasoningEngine(m, decode, encode, device)

HTML_PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Ashen GPT - Agentic Web Chat Interface</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <script src="https://cdn.jsdelivr.net/npm/marked/marked.js"></script>
    <style>
        body { background-color: #0f172a; color: #f8fafc; font-family: ui-sans-serif, system-ui, sans-serif; }
        .chat-container::-webkit-scrollbar { width: 6px; }
        .chat-container::-webkit-scrollbar-thumb { background: #334155; border-radius: 3px; }
    </style>
</head>
<body class="h-screen flex flex-col">
    <!-- Header -->
    <header class="bg-slate-900 border-b border-slate-800 px-6 py-4 flex items-center justify-between shadow-md">
        <div class="flex items-center space-x-3">
            <div class="w-3 h-3 bg-emerald-500 rounded-full animate-pulse"></div>
            <h1 class="text-xl font-bold tracking-tight text-white">Ashen GPT <span class="text-xs px-2 py-0.5 bg-indigo-900 text-indigo-200 rounded-full">Agentic MoE &amp; 8K Context</span></h1>
        </div>
        <div class="flex items-center space-x-4">
            <button onclick="clearHistory()" class="px-3 py-1.5 text-xs font-medium bg-slate-800 hover:bg-slate-700 text-slate-300 rounded-lg transition border border-slate-700">Clear Chat</button>
            <span class="text-xs text-slate-400">Agent Tools: <span class="text-emerald-400 font-mono">Active</span></span>
        </div>
    </header>

    <!-- Main Content -->
    <div class="flex-1 flex overflow-hidden">
        <!-- Sidebar -->
        <aside class="w-72 bg-slate-900/55 border-r border-slate-800 p-4 hidden md:flex flex-col justify-between">
            <div class="space-y-4">
                <h2 class="text-xs font-semibold text-slate-400 uppercase tracking-wider">Agentic Capabilities</h2>
                <div class="bg-slate-800/60 p-3 rounded-xl border border-slate-700/50 space-y-2 text-xs text-slate-300">
                    <div class="flex justify-between"><span>read_file</span><span class="font-mono text-indigo-400">Enabled</span></div>
                    <div class="flex justify-between"><span>write_file</span><span class="font-mono text-indigo-400">Enabled</span></div>
                    <div class="flex justify-between"><span>glob</span><span class="font-mono text-indigo-400">Enabled</span></div>
                    <div class="flex justify-between"><span>grep_search</span><span class="font-mono text-indigo-400">Enabled</span></div>
                    <div class="flex justify-between"><span>run_shell_command</span><span class="font-mono text-emerald-400">Enabled</span></div>
                </div>
                <div class="space-y-2">
                    <h2 class="text-xs font-semibold text-slate-400 uppercase tracking-wider">Model Specs</h2>
                    <ul class="text-xs text-slate-400 space-y-1 pl-2">
                        <li>• ~127M Parameters</li>
                        <li>• Mixture of Experts (4 Experts)</li>
                        <li>• Dynamic RoPE 8K Context</li>
                        <li>• ReAct Agent Loop</li>
                    </ul>
                </div>
            </div>
            <div class="text-xs text-slate-500 text-center pt-4 border-t border-slate-800">
                Ashen GPT Agentic Web &bull; Powered by PyTorch
            </div>
        </aside>

        <!-- Chat Area -->
        <main class="flex-1 flex flex-col bg-slate-950">
            <div id="chat-container" class="flex-1 overflow-y-auto p-6 space-y-6 chat-container">
                <!-- Welcome Message -->
                <div class="flex items-start space-x-4 max-w-3xl">
                    <div class="w-8 h-8 rounded-full bg-indigo-600 flex items-center justify-center font-bold text-sm text-white shrink-0">AG</div>
                    <div class="bg-slate-900 border border-slate-800 rounded-2xl p-4 text-slate-200 text-sm shadow-sm space-y-2">
                        <p class="font-semibold text-white">Hello! I am Ashen GPT Agent.</p>
                        <p>I have full agentic capabilities including tool execution (`read_file`, `write_file`, `glob`, `grep_search`, `run_shell_command`). Ask me to inspect files, search code, or execute tasks!</p>
                    </div>
                </div>
            </div>

            <!-- Input Bar -->
            <div class="p-4 bg-slate-900 border-t border-slate-800">
                <div class="max-w-4xl mx-auto flex items-end space-x-3">
                    <div class="flex-1 bg-slate-950 border border-slate-700 rounded-xl focus-within:border-indigo-500 transition">
                        <textarea id="user-input" rows="1" placeholder="Type your agent request... (e.g. 'Run pytest' or 'Search for model')" class="w-full bg-transparent p-3 text-slate-100 placeholder-slate-500 text-sm focus:outline-none resize-none max-h-32"></textarea>
                    </div>
                    <button id="send-btn" onclick="sendMessage()" class="bg-indigo-600 hover:bg-indigo-500 text-white px-5 py-3 rounded-xl font-medium text-sm transition shadow-lg shrink-0 flex items-center space-x-2">
                        <span>Send</span>
                        <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M14 5l7 7m0 0l-7 7m7-7H3"></path></svg>
                    </button>
                </div>
            </div>
        </main>
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

        function appendMessage(sender, thought, text) {
            const container = document.getElementById('chat-container');
            const isUser = sender === 'user';
            
            const messageDiv = document.createElement('div');
            messageDiv.className = `flex items-start space-x-4 max-w-3xl ${isUser ? 'ml-auto flex-row-reverse space-x-reverse' : ''}`;
            
            const avatar = document.createElement('div');
            avatar.className = `w-8 h-8 rounded-full flex items-center justify-center font-bold text-sm text-white shrink-0 ${isUser ? 'bg-emerald-600' : 'bg-indigo-600'}`;
            avatar.textContent = isUser ? 'U' : 'AG';
            
            const contentDiv = document.createElement('div');
            contentDiv.className = `rounded-2xl p-4 text-sm shadow-sm space-y-3 ${isUser ? 'bg-emerald-900/40 border border-emerald-800 text-emerald-100' : 'bg-slate-900 border border-slate-800 text-slate-200 w-full'}`;
            
            if (!isUser && thought) {
                const thinkDetails = document.createElement('details');
                thinkDetails.className = 'group bg-slate-950/60 rounded-xl border border-indigo-900/50 overflow-hidden';
                
                const summary = document.createElement('summary');
                summary.className = 'px-3 py-2 text-xs font-medium text-indigo-300 cursor-pointer select-none hover:bg-indigo-950/40 flex items-center justify-between';
                summary.innerHTML = '<span>🧠 Agentic Reasoning &amp; Tool Observations</span><span class="text-indigo-400 group-open:rotate-180 transition-transform">▼</span>';
                
                const thinkBody = document.createElement('div');
                thinkBody.className = 'p-3 text-xs text-slate-400 font-mono whitespace-pre-wrap border-t border-indigo-900/40 bg-slate-950/90';
                thinkBody.textContent = thought;
                
                thinkDetails.appendChild(summary);
                thinkDetails.appendChild(thinkBody);
                contentDiv.appendChild(thinkDetails);
            }
            
            const textBody = document.createElement('div');
            textBody.className = 'prose prose-invert max-w-none text-slate-200';
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
            sendBtn.innerHTML = '<svg class="w-4 h-4 animate-spin" fill="none" viewBox="0 0 24 24"><circle class="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4"></circle><path class="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8v8H4z"></path></svg>';

            try {
                const res = await fetch('/api/chat', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ message: msg })
                });
                const data = await res.json();
                appendMessage('assistant', data.thought, data.response);
            } catch (err) {
                appendMessage('assistant', 'Error', 'Failed to communicate with Ashen GPT agent backend.');
            } finally {
                sendBtn.disabled = false;
                sendBtn.innerHTML = '<span>Send</span><svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M14 5l7 7m0 0l-7 7m7-7H3"></path></svg>';
            }
        }

        async function clearHistory() {
            await fetch('/api/clear', { method: 'POST' });
            const container = document.getElementById('chat-container');
            container.innerHTML = `
                <div class="flex items-start space-x-4 max-w-3xl">
                    <div class="w-8 h-8 rounded-full bg-indigo-600 flex items-center justify-center font-bold text-sm text-white shrink-0">AG</div>
                    <div class="bg-slate-900 border border-slate-800 rounded-2xl p-4 text-slate-200 text-sm shadow-sm space-y-2">
                        <p class="font-semibold text-white">Conversation cleared.</p>
                        <p>Agent memory reset successfully. Ready for new tasks!</p>
                    </div>
                </div>
            `;
        }
    </script>
</body>
</html>
"""

class ChatHandler(http.server.SimpleHTTPRequestHandler):
    def do_GET(self):
        if self.path == '/' or self.path == '':
            self.send_response(200)
            self.send_header('Content-Type', 'text/html; charset=utf-8')
            self.end_headers()
            self.wfile.write(HTML_PAGE.encode('utf-8'))
        else:
            self.send_response(404)
            self.end_headers()

    def do_POST(self):
        if self.path == '/api/chat':
            content_length = int(self.headers.get('Content-Length', 0))
            body = self.rfile.read(content_length)
            data = json.loads(body.decode('utf-8'))
            message = data.get('message', '')

            try:
                thought, response = reasoner.solve_with_agent(message, max_new_tokens=250)
                resp_data = {'thought': thought, 'response': response}
            except Exception as e:
                resp_data = {'thought': 'Error during agent execution', 'response': str(e)}

            self.send_response(200)
            self.send_header('Content-Type', 'application/json; charset=utf-8')
            self.end_headers()
            self.wfile.write(json.dumps(resp_data).encode('utf-8'))

        elif self.path == '/api/clear':
            reasoner.clear_history()
            self.send_response(200)
            self.send_header('Content-Type', 'application/json; charset=utf-8')
            self.end_headers()
            self.wfile.write(json.dumps({'status': 'cleared'}).encode('utf-8'))
        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, format, *args):
        return

def run_server(port=5000):
    server = socketserver.TCPServer(('127.0.0.1', port), ChatHandler)
    print(f"\n========================================================")
    print(f" Ashen GPT Agentic Web Interface running at: http://localhost:{port}")
    print(f" Open your browser to interact with the Agentic MoE Model!")
    print(f"========================================================\n")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down web server...")
        server.server_close()

if __name__ == '__main__':
    run_server(5000)
