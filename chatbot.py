import sys
import io
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

import torch
import torch.nn as nn
from torch.nn import functional as F
import tiktoken
import pickle
import os
import re
import subprocess
import glob as glob_module
import json
import datetime

# --- Session Storage ---
SESSIONS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'sessions_cli')
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
        self.workspace_context = ""
        self.session_id = None
        self.working_dir = os.getcwd()  # current working directory for tools

    def clear_history(self):
        self.history = []

    def change_working_dir(self, dir_path):
        """Change the working directory for all tool operations."""
        if not dir_path:
            return False
        if not os.path.isdir(dir_path):
            return False
        self.working_dir = os.path.abspath(dir_path)
        return True

    def get_working_dir(self):
        return self.working_dir

    def set_workspace_context(self, dir_path):
        if not dir_path:
            self.workspace_context = ""
        else:
            try:
                items = []
                for entry in os.scandir(dir_path):
                    prefix = "📁 " if entry.is_dir() else "📄 "
                    items.append(f"  {prefix}{entry.name}")
                self.workspace_context = f"Directory: {os.path.abspath(dir_path)}\n" + "\n".join(items)
            except Exception as e:
                self.workspace_context = f"Error scanning directory: {e}"

    def execute_tool(self, tool_name, kwargs):
        try:
            if tool_name == 'read_file':
                path = kwargs.get('file_path', '')
                full_path = path if os.path.isabs(path) else os.path.join(self.working_dir, path)
                if os.path.exists(full_path):
                    with open(full_path, 'r', encoding='utf-8', errors='ignore') as f:
                        return f.read()[:2000]
                return f"Error: File not found: {path}"

            elif tool_name == 'write_file':
                path = kwargs.get('file_path', '')
                content = kwargs.get('content', '')
                full_path = path if os.path.isabs(path) else os.path.join(self.working_dir, path)
                with open(full_path, 'w', encoding='utf-8') as f:
                    f.write(content)
                return f"Successfully wrote to {path}"

            elif tool_name == 'glob':
                pattern = kwargs.get('pattern', '*')
                old_cwd = os.getcwd()
                try:
                    os.chdir(self.working_dir)
                    matches = glob_module.glob(pattern, recursive=True)
                finally:
                    os.chdir(old_cwd)
                return str(matches[:30])

            elif tool_name == 'grep_search':
                pattern = kwargs.get('pattern', '')
                results = []
                old_cwd = os.getcwd()
                try:
                    os.chdir(self.working_dir)
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
                finally:
                    os.chdir(old_cwd)
                return "\n".join(results) if results else "No matches found."

            elif tool_name == 'run_shell_command':
                cmd = kwargs.get('command', '')
                res = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=30, cwd=self.working_dir)
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

        if self.workspace_context:
            system_instructions += f"\n### Current Workspace Context\n{self.workspace_context}\n\n"

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

                print(f"\n[Agent executing tool: {tool_name}({args_str})...]")
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

# --- Session Management Helpers ---

def _session_file(session_id):
    return os.path.join(SESSIONS_DIR, f"{session_id}.json")

def save_session(session_id, data):
    filepath = _session_file(session_id)
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

def load_session(session_id):
    filepath = _session_file(session_id)
    if not os.path.exists(filepath):
        return None
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)

def list_sessions():
    sessions = []
    for fname in os.listdir(SESSIONS_DIR):
        if not fname.endswith('.json'):
            continue
        sid = fname[:-5]
        filepath = os.path.join(SESSIONS_DIR, fname)
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
            sessions.append({
                'id': sid,
                'name': data.get('name', 'Untitled'),
                'updated': data.get('updated_at', ''),
                'message_count': len(data.get('history', [])),
                'workspace_context': data.get('workspace_context', '')
            })
        except Exception:
            pass
    sessions.sort(key=lambda s: s.get('updated', ''), reverse=True)
    return sessions

def delete_session(session_id):
    filepath = _session_file(session_id)
    if os.path.exists(filepath):
        os.remove(filepath)
        return True
    return False

def create_new_session(name=None):
    global current_session_id
    ts = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    current_session_id = f"session_{ts}"
    save_session(current_session_id, {
        'name': name or 'Untitled',
        'created_at': datetime.datetime.now().isoformat(),
        'updated_at': datetime.datetime.now().isoformat(),
        'history': [],
        'workspace_context': ''
    })
    return current_session_id

def append_to_session(session_id, user_msg, assistant_resp):
    global current_session_id
    data = load_session(session_id)
    if not data:
        return
    data['history'].append({'user': user_msg, 'assistant': assistant_resp})
    data['updated_at'] = datetime.datetime.now().isoformat()
    save_session(session_id, data)
    current_session_id = session_id

def rename_session(session_id, name):
    data = load_session(session_id)
    if data:
        data['name'] = name
        save_session(session_id, data)
        return True
    return False

reasoner.session_id = None  # Track session on reasoner

if __name__ == "__main__":
    print("\n--- Ashen GPT Fully Functional Agentic CLI Chatbot Ready ---")
    print("Capabilities: Chain-of-Thought + Tool Execution (read_file, write_file, glob, grep_search, run_shell_command)")
    print("Session & Workspace: /sessions, /new, /load <id>, /delete <id>, /rename <name>, /workspace <path>, /wctx off")
    print("Working Directory: /cd <path> (change), /cd (show), /pwd")
    print("Commands: /clear (reset conversation), /help (show help), /exit or /quit (exit)")
    
    # Auto-create first session
    current_session_id = create_new_session("Initial Session")
    reasoner.session_id = current_session_id
    
    while True:
        try:
            prompt = input("\nUser:\n> ")
            if not prompt.strip():
                continue
            cmd = prompt.strip().lower()
            
            # Session management commands
            if cmd in ['exit', 'quit', '/exit', '/quit']:
                print("Goodbye!")
                break
            if cmd == '/clear':
                # Save before clearing
                if current_session_id and load_session(current_session_id):
                    save_session(current_session_id, {
                        'name': load_session(current_session_id)['name'],
                        'history': list(reasoner.history),
                        'workspace_context': reasoner.workspace_context
                    })
                reasoner.clear_history()
                reasoner.set_workspace_context("")
                print("[Conversation history cleared]")
                continue
            if cmd == '/sessions':
                sessions = list_sessions()
                if not sessions:
                    print("\nNo sessions found.")
                else:
                    print(f"\n{'ID':<25} {'Name':<20} {'Msgs':>6} {'Workspace':>10}  Updated")
                    print("-" * 90)
                    for s in sessions:
                        wctx = "📁" if s.get('workspace_context') else "—"
                        updated = s['updated'][:16] if s['updated'] else ''
                        print(f"{s['id']:<25} {s['name']:<20} {s['message_count']:>6} {wctx:>10}  {updated}")
                    print(f"\nTotal: {len(sessions)} sessions | Active: {current_session_id or 'None'}")
                continue
            if cmd == '/new':
                current_session_id = create_new_session("New Session")
                reasoner.session_id = current_session_id
                reasoner.history = []
                reasoner.workspace_context = ""
                print(f"[Created new session: {current_session_id}]")
                continue
            if cmd.startswith('/load '):
                sid = cmd[6:].strip()
                if not sid:
                    print("[Usage: /load <session_id>]")
                    continue
                data = load_session(sid)
                if data:
                    current_session_id = sid
                    reasoner.session_id = sid
                    reasoner.history = [(m['user'], m['assistant']) for m in data.get('history', [])]
                    reasoner.set_workspace_context('')
                    if data.get('workspace_context'):
                        reasoner.workspace_context = data['workspace_context']
                    print(f"[Loaded session '{data['name']}' with {len(data['history'])} messages]")
                else:
                    print(f"[Session not found: {sid}]")
                continue
            if cmd.startswith('/delete '):
                sid = cmd[8:].strip()
                if not sid:
                    print("[Usage: /delete <session_id>]")
                    continue
                if delete_session(sid):
                    if current_session_id == sid:
                        current_session_id = None
                        reasoner.session_id = None
                        reasoner.history = []
                        reasoner.workspace_context = ""
                    print(f"[Deleted session: {sid}]")
                else:
                    print(f"[Session not found: {sid}]")
                continue
            if cmd.startswith('/rename '):
                name = cmd[8:].strip()
                if not name:
                    print("[Usage: /rename <new_name>]")
                    continue
                if rename_session(current_session_id, name):
                    print(f"[Renamed session to '{name}'")
                else:
                    print("[Failed to rename session]")
                continue
            if cmd.startswith('/workspace '):
                dir_path = cmd[11:].strip()
                if os.path.isdir(dir_path):
                    reasoner.set_workspace_context(dir_path)
                    print(f"[Workspace context set: {dir_path}]")
                    # Update session
                    if current_session_id:
                        save_session(current_session_id, {
                            'name': load_session(current_session_id)['name'] if load_session(current_session_id) else 'Untitled',
                            'history': list(reasoner.history),
                            'workspace_context': reasoner.workspace_context
                        })
                else:
                    print(f"[Invalid directory: {dir_path}]")
                continue
            if cmd == '/wctx off' or cmd == '/workspace off':
                reasoner.set_workspace_context('')
                print("[Workspace context cleared]")
                if current_session_id:
                    save_session(current_session_id, {
                        'name': load_session(current_session_id)['name'] if load_session(current_session_id) else 'Untitled',
                        'history': list(reasoner.history),
                        'workspace_context': ''
                    })
                continue
            if cmd.startswith('/cd '):
                dir_path = cmd[4:].strip()
                if not dir_path:
                    print(f"[Current working directory: {reasoner.working_dir}]")
                    continue
                if reasoner.change_working_dir(dir_path):
                    print(f"[Working directory changed to: {reasoner.working_dir}]")
                else:
                    print(f"[Invalid directory: {dir_path}]")
                continue
            if cmd == '/pwd':
                print(f"[Working directory: {reasoner.working_dir}]")
                continue
            if cmd == '/help':
                print("""Ashen GPT Agentic CLI Help:
  - Ask questions or request actions (e.g. 'Run pytest' or 'Check files').
  - /clear : Clear conversation memory.
  - /help  : Show this help message.
  - /exit  : Exit the CLI.

Session Management:
  - /sessions     : List all saved sessions.
  - /new          : Create a fresh session.
  - /load <id>    : Load a session by ID (shown in /sessions).
  - /delete <id>  : Delete a session.
  - /rename <name>: Rename current session.

Working Directory:
  - /cd <path>    : Change working directory for all tool operations.
  - /cd           : Show current working directory.
  - /pwd          : Alias for /cd (show current directory).

Workspace Context:
  - /workspace <path> : Scan a directory and inject its file tree into prompts.
  - /wctx off         : Clear workspace context.""")
                continue

            print("Invalid command. Type /help for usage.")

            GREY = "\033[90m"
            RESET = "\033[0m"
            
            # Auto-name first message in session
            if not current_session_id:
                current_session_id = create_new_session("Untitled")
                reasoner.session_id = current_session_id
                sdata = load_session(current_session_id)
                if sdata and sdata.get('name', 'Untitled') == 'Untitled':
                    preview = prompt[:40].replace('\n', ' ')
                    sdata['name'] = preview + ('...' if len(prompt) > 40 else '')
                    save_session(current_session_id, sdata)

            thought, answer = reasoner.solve_with_agent(prompt, max_new_tokens=250)
            print(f"\n{GREY}<think>\n{thought}\n</think>{RESET}")
            print(f"\nAssistant:\n{answer}")
            
            # Save to session
            append_to_session(current_session_id, prompt, f"<think>\n{thought}\n</think>\n{answer}")
        except (KeyboardInterrupt, EOFError):
            print("\nGoodbye!")
            break
