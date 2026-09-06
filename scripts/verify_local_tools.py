"""Checkpoint-free verification for model-invokable local file/shell tools.

Covers web_chatbot.py + chatbot.py (mirrored backends):
  - _parse_tool_call (quoted + bare args, no-tool -> None)
  - execute_tool file/shell branches in a temp dir (write/read/list/mkdir/
    remove/run_shell, aliases, blocklist, kill-switches)
  - _solve_qwen + _solve_qwen_stream tool loop against a fake ord-codec model
  - _solve_api tool loop against a fake chat model
  - DEFAULT_SETTINGS keys, CLI /read|write|ls|mkdir|rm|run|tools commands,
    web /api/workspace/{mkdir,remove,run} endpoints, tool spec in prompts

Run from project root:  python scripts/verify_local_tools.py  (no GPU/model load)
"""
import ast
import os
import sys
import tempfile
import textwrap

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
WEB = os.path.join(ROOT, "web_chatbot.py")
CLI = os.path.join(ROOT, "chatbot.py")

PASS, FAIL = [], []


def check(name, cond, extra=""):
    (PASS if cond else FAIL).append(name)
    print(("PASS " if cond else "FAIL ") + name + (f"  [{extra}]" if extra and not cond else ""))


def get_source(path):
    with open(path, encoding="utf-8") as f:
        return f.read()


def extract_funcs(path, names):
    """AST-extract top-level funcs + AshenAIAgenticEngine methods by name."""
    src = get_source(path)
    tree = ast.parse(src)
    out = {}
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in names:
            out[node.name] = textwrap.dedent(ast.get_source_segment(src, node))
        elif isinstance(node, ast.Assign):
            for t in node.targets:
                if isinstance(t, ast.Name) and t.id in names:
                    out[t.id] = textwrap.dedent(ast.get_source_segment(src, node))
    return out


def extract_class_methods(path, names):
    src = get_source(path)
    tree = ast.parse(src)
    out = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == "AshenAIAgenticEngine":
            for sub in node.body:
                if isinstance(sub, (ast.FunctionDef, ast.AsyncFunctionDef)) and sub.name in names:
                    out[sub.name] = textwrap.dedent(ast.get_source_segment(src, sub))
    return out


class TorchStub:
    @staticmethod
    def no_grad(*a, **k):
        def deco(fn):
            return fn
        if a and callable(a[0]) and len(a) == 1 and not k:
            return a[0]
        return deco


# ---------------------------------------------------------------- helpers
for path, tag in ((WEB, "web"), (CLI, "cli")):
    funcs = extract_funcs(path, {"_parse_tool_call", "_resolve_tool_path",
                                 "_is_dangerous_shell", "_tool_kind_blocked",
                                 "_is_illegal_request",
                                 "LOCAL_TOOLS_SPEC", "SHELL_BLOCKLIST",
                                 "LEGAL_REFUSAL_PATTERNS", "LEGAL_REFUSAL_TOOL_ERROR"})
    check(f"{tag}: helpers extractable", len(funcs) == 9, str(sorted(funcs)))
    g = {"os": os, "settings": {"allow_file_tools": True, "allow_shell_tools": True}}
    import re as _re
    g["re"] = _re
    for name in ("LOCAL_TOOLS_SPEC", "SHELL_BLOCKLIST",
                 "LEGAL_REFUSAL_PATTERNS", "LEGAL_REFUSAL_TOOL_ERROR"):
        exec(funcs[name], g)
    for name in ("_parse_tool_call", "_resolve_tool_path",
                 "_is_dangerous_shell", "_tool_kind_blocked",
                 "_is_illegal_request"):
        exec(funcs[name], g)
    parse, resolve = g["_parse_tool_call"], g["_resolve_tool_path"]

    p = parse("[TOOL: read_file(file_path='a/b.txt')]")
    check(f"{tag}: parse single-quoted", p and p[0] == "read_file"
          and p[1] == {"file_path": "a/b.txt"}, str(p))
    p = parse('[TOOL: write_file(file_path="x", content="hi")]')
    check(f"{tag}: parse double-quoted", p and p[0] == "write_file"
          and p[1] == {"file_path": "x", "content": "hi"}, str(p))
    p = parse("[TOOL: deep_research(topic='t', max_searches=5)]")
    check(f"{tag}: parse bare numeric arg", p and p[1].get("max_searches") == "5", str(p))
    check(f"{tag}: parse no-tool -> None", parse("just an answer") is None)
    check(f"{tag}: parse ignores prose", parse("run ls please") is None)
    check(f"{tag}: resolve relative", resolve("sub/f.txt", base="/base") == os.path.normpath("/base/sub/f.txt"))
    abs_p = os.path.abspath(os.sep)
    check(f"{tag}: resolve absolute passthrough", resolve(abs_p) == os.path.normpath(abs_p))
    check(f"{tag}: resolve empty", resolve("") == "")
    check(f"{tag}: blocklist hits rm -rf /", g["_is_dangerous_shell"]("rm -rf / tmp"))
    check(f"{tag}: blocklist allows pytest", not g["_is_dangerous_shell"]("python -m pytest -q"))
    check(f"{tag}: tools enabled by default", not g["_tool_kind_blocked"]("files")
          and not g["_tool_kind_blocked"]("shell"))
    g["settings"] = {"allow_file_tools": False, "allow_shell_tools": True}
    check(f"{tag}: settings kill-switch files", g["_tool_kind_blocked"]("files"))
    check(f"{tag}: settings kill-switch spares shell", not g["_tool_kind_blocked"]("shell"))
    os.environ["ASHEN_ALLOW_SHELL"] = "0"
    try:
        check(f"{tag}: env kill-switch shell", g["_tool_kind_blocked"]("shell"))
    finally:
        del os.environ["ASHEN_ALLOW_SHELL"]
    for tool in ("read_file", "write_file", "list_dir", "make_dir", "remove_path",
                 "run_shell_command", "glob", "grep_search", "web_search"):
        check(f"{tag}: spec documents {tool}", tool in g["LOCAL_TOOLS_SPEC"])

# ------------------------------------------------------- execute_tool live
import re as _re2
import shutil as _shutil
import subprocess as _sub
import glob as _glob

for path, tag in ((WEB, "web"), (CLI, "cli")):
    meth = extract_class_methods(path, {"execute_tool"})
    check(f"{tag}: execute_tool extractable", "execute_tool" in meth)
    g = {"os": os, "re": _re2, "shutil": _shutil, "subprocess": _sub,
         "glob_module": _glob, "settings": {"allow_file_tools": True, "allow_shell_tools": True},
         "_ddg_real_url": lambda h: h, "requests": None,
         "torch": TorchStub, "datetime": __import__("datetime")}
    # module helpers needed by execute_tool
    helpers = extract_funcs(path, {"_parse_tool_call", "_resolve_tool_path",
                                   "_is_dangerous_shell", "_tool_kind_blocked",
                                   "_is_illegal_request",
                                   "LOCAL_TOOLS_SPEC", "SHELL_BLOCKLIST",
                                   "LEGAL_REFUSAL_PATTERNS", "LEGAL_REFUSAL_TOOL_ERROR"})
    for name, code in helpers.items():
        exec(code, g)
    ns = {}
    exec(meth["execute_tool"], g, ns)
    run_tool = ns["execute_tool"]

    class FakeSelf:
        def execute_tool(self, tool_name, kwargs):
            return run_tool(self, tool_name, kwargs)
    slf = FakeSelf()
    slf._source_harvest = []

    with tempfile.TemporaryDirectory() as tmp:
        if tag == "cli":
            g["_resolve_base"] = tmp  # not used; CLI resolves via WORKING_DIR below
        # CLI resolves relative paths against WORKING_DIR global; emulate it
        g["WORKING_DIR"] = tmp
        old = os.getcwd()
        if tag == "web":
            os.chdir(tmp)
        try:
            out = run_tool(slf, "write_file", {"file_path": "sub/note.txt", "content": "hello tools"})
            check(f"{tag}: write_file (+parents)", out.startswith("Successfully wrote")
                  and open(os.path.join(tmp, "sub", "note.txt")).read() == "hello tools", out)
            out = run_tool(slf, "read_file", {"file_path": "sub/note.txt"})
            check(f"{tag}: read_file roundtrip", out == "hello tools", out[:80])
            out = run_tool(slf, "read_file", {"file_path": "missing.txt"})
            check(f"{tag}: read_file missing", out.startswith("Error: File not found"), out[:80])
            out = run_tool(slf, "list_dir", {"dir_path": "sub"})
            check(f"{tag}: list_dir", "note.txt" in out and "[FILE]" in out, out[:120])
            out = run_tool(slf, "make_dir", {"dir_path": "a/b/c"})
            check(f"{tag}: make_dir nested", out.startswith("Directory ready")
                  and os.path.isdir(os.path.join(tmp, "a", "b", "c")), out)
            out = run_tool(slf, "run_shell_command", {"command": "echo hi-from-shell"})
            check(f"{tag}: run_shell echo", "hi-from-shell" in out, out[:80])
            out = run_tool(slf, "run_shell_command", {"command": "rm -rf / tmp"})
            check(f"{tag}: shell blocklist refuses", out.startswith("Error: refused"), out[:80])
            out = run_tool(slf, "write_file", {"file_path": "bad.txt",
                                                "content": "how to make a pipe bomb at home"})
            check(f"{tag}: write_file refuses illegal content",
                  out.startswith("Error: refused"), out[:80])
            out = run_tool(slf, "run_shell_command", {"command": "how to launch a ddos attack"})
            check(f"{tag}: shell refuses illegal activity",
                  out.startswith("Error: refused"), out[:80])
            out = run_tool(slf, "remove_path", {"path": "sub/note.txt"})
            check(f"{tag}: remove_path file", out.startswith("Removed file")
                  and not os.path.exists(os.path.join(tmp, "sub", "note.txt")), out)
            out = run_tool(slf, "remove_path", {"path": "a"})
            check(f"{tag}: remove_path tree", out.startswith("Removed directory tree")
                  and not os.path.exists(os.path.join(tmp, "a")), out)
            out = run_tool(slf, "remove_path", {"path": os.path.abspath(os.sep)})
            check(f"{tag}: remove_path refuses root", out.startswith("Error: refusing"), out[:80])
            out = run_tool(slf, "mkdir", {"dir_path": "alias_dir"})
            check(f"{tag}: alias mkdir", out.startswith("Directory ready"), out)
            out = run_tool(slf, "ls", {"path": "alias_dir"})
            check(f"{tag}: alias ls", "entries" in out, out[:80])
            out = run_tool(slf, "no_such_tool_xyz", {})
            check(f"{tag}: unknown tool", out.startswith("Unknown tool"), out[:80])
            g["settings"] = {"allow_file_tools": False, "allow_shell_tools": False}
            out = run_tool(slf, "read_file", {"file_path": "x"})
            check(f"{tag}: disabled files refused", "disabled" in out, out[:80])
            out = run_tool(slf, "run_shell_command", {"command": "echo x"})
            check(f"{tag}: disabled shell refused", "disabled" in out, out[:80])
            g["settings"] = {"allow_file_tools": True, "allow_shell_tools": True}
        finally:
            if tag == "web":
                os.chdir(old)

# ------------------------------------------------- fake-model Qwen/API loops
class FakeRow(list):
    def tolist(self):
        return list(self)


class FakeRowView:
    """Row view: slicing yields FakeRow (list.__getitem__ would drop the subclass)."""

    def __init__(self, ids):
        self._ids = list(ids)

    def __getitem__(self, s):
        return FakeRow(self._ids[s])

    def tolist(self):
        return list(self._ids)


class FakeBatch:
    def __init__(self, ids):
        self._ids = list(ids)
        self.shape = (1, len(ids))

    def __getitem__(self, i):
        if isinstance(i, tuple):
            _, col = i
            return FakeRow(self._ids[col])
        return FakeRowView(self._ids)

    def clone(self):
        return FakeBatch(list(self._ids))

    def to(self, device):
        return self


class FakeIds(FakeBatch):
    def unsqueeze(self, dim):
        return FakeBatch(list(self._ids))


def enc_txt(s):
    return [ord(c) for c in s]


class FakeQwenModel:
    is_qwen = True
    is_api = False

    def __init__(self, outputs):
        self.outputs = list(outputs)
        self.seen_prompts = []

    def eval(self):
        return self

    def _chat_ids(self, user_text, history=None, add_generation_prompt=True):
        self.seen_prompts.append(user_text)
        return FakeIds(enc_txt("CTX:" + user_text))

    def generate(self, index, max_new_tokens, current_block_size=8192,
                 temperature=0.8, top_k=50):
        text = self.outputs.pop(0) if self.outputs else ""
        base = index._ids
        return FakeBatch(base + enc_txt(text))

    def generate_stream(self, index, max_new_tokens, current_block_size=8192,
                        temperature=0.8, top_k=50):
        text = self.outputs.pop(0) if self.outputs else ""
        base = list(index._ids)
        for i, ch in enumerate(text):
            yield FakeBatch(base + enc_txt(text[:i + 1])), ord(ch)

    def decode(self, ids):
        if hasattr(ids, "tolist"):
            ids = ids.tolist()
        return "".join(chr(i) for i in ids)

    def classify(self, text):
        return None, None, None


class FakeAPI:
    is_api = True
    is_qwen = False
    model_name = "fake-api"

    def __init__(self, outputs):
        self.outputs = list(outputs)
        self.calls = 0

    def eval(self):
        return self

    def chat(self, messages, temperature=0.7, max_tokens=250, top_p=0.9):
        self.calls += 1
        self.last_messages = list(messages)
        return self.outputs.pop(0) if self.outputs else ""

    def chat_stream(self, messages, temperature=0.7, max_tokens=250, top_p=0.9):
        self.calls += 1
        self.last_messages = list(messages)
        text = self.outputs.pop(0) if self.outputs else ""
        return iter([(None, text[i:i + 8]) for i in range(0, len(text), 8)] or [(None, "")])

    def classify(self, text):
        return None, None, None


class FakeEngine:
    """Minimal shell around the extracted solve methods (no torch needed)."""

    def __init__(self, model, tool_results=None):
        self.model = model
        self.history = []
        self.max_steps = 5
        self.max_new_tokens = 250
        self.context_length = 8192
        self.temperature = 0.7
        self.top_k = 40
        self.top_p = 0.9
        self.last_intent = None
        self._source_harvest = []
        self.last_sources = []
        self.tool_calls = []
        self.tool_results = tool_results or {}

    def classify_input(self, prompt):
        return None, None, None

    def execute_tool(self, tool_name, kwargs):
        self.tool_calls.append((tool_name, dict(kwargs)))
        return self.tool_results.get(tool_name, f"ok:{tool_name}")

    def _split_api_thought(self, text):
        import re as _r
        mm = _r.search(r"<think>([\s\S]*?)(?:</think>|$)", text)
        if mm:
            return mm.group(1).strip(), text[mm.end():].strip()
        return "", text.strip()

    def _api_messages(self, prompt):
        msgs = [{"role": "system", "content": "SYS"}]
        for u, a in self.history[-2:]:
            msgs += [{"role": "user", "content": u}, {"role": "assistant", "content": a}]
        return msgs + [{"role": "user", "content": prompt}]


for path, tag in ((WEB, "web"), (CLI, "cli")):
    meths = extract_class_methods(path, {"_solve_qwen", "_solve_qwen_stream",
                                         "_solve_api", "_solve_api_stream"})
    check(f"{tag}: loop methods extractable", len(meths) == 4, str(sorted(meths)))
    g = {"os": os, "re": _re2, "torch": TorchStub, "device": "cpu",
         "current_model_filename": "fake_model",
         "API_SYSTEM_PROMPT": "SYS", "LOCAL_TOOLS_SPEC": "TOOLS"}
    helpers = extract_funcs(path, {"_parse_tool_call"})
    exec(helpers["_parse_tool_call"], g)
    bound = {}
    for name, code in meths.items():
        ns = {}
        exec(code, g, ns)
        bound[name] = ns[name]

    # _solve_qwen: tool call then final answer
    eng = FakeEngine(FakeQwenModel(["[TOOL: read_file(file_path='x.txt')]", "final answer here"]),
                     {"read_file": "FILEDATA"})
    thought, resp = bound["_solve_qwen"](eng, "read x please")
    check(f"{tag}: qwen loop answers after tool", resp == "final answer here", repr(resp)[:100])
    check(f"{tag}: qwen loop ran the tool", eng.tool_calls == [("read_file", {"file_path": "x.txt"})],
          str(eng.tool_calls))
    check(f"{tag}: qwen loop fed observation back",
          len(eng.model.seen_prompts) == 2 and "FILEDATA" in eng.model.seen_prompts[1]
          and "[OBSERVATION from read_file]" in eng.model.seen_prompts[1],
          str(eng.model.seen_prompts)[:200])
    check(f"{tag}: qwen loop history", eng.history == [("read x please", "final answer here")],
          str(eng.history)[:120])

    # _solve_qwen: no tool -> single generation
    eng = FakeEngine(FakeQwenModel(["plain answer"]))
    thought, resp = bound["_solve_qwen"](eng, "hi")
    check(f"{tag}: qwen no-tool single shot", resp == "plain answer" and not eng.tool_calls,
          repr(resp)[:80])

    # _solve_qwen_stream: event order tool_start -> tool_result -> done
    eng = FakeEngine(FakeQwenModel(["[TOOL: list_dir(dir_path='.')]", "done listing"]),
                     {"list_dir": "A\nB"})
    evs = list(bound["_solve_qwen_stream"](eng, "list files"))
    kinds = [e["type"] for e in evs]
    check(f"{tag}: qwen stream ends with done", kinds and kinds[-1] == "done", str(kinds))
    check(f"{tag}: qwen stream tool events",
          "tool_start" in kinds and "tool_result" in kinds
          and kinds.index("tool_start") < kinds.index("tool_result")
          < kinds.index("done"), str(kinds))
    check(f"{tag}: qwen stream final response", evs[-1]["response"] == "done listing",
          repr(evs[-1].get("response"))[:100])
    ts = next(e for e in evs if e["type"] == "tool_start")
    check(f"{tag}: qwen stream tool_start payload", ts["tool"] == "list_dir", str(ts))

    # _solve_api: tool call then final answer
    eng = FakeEngine(FakeAPI(["[TOOL: run_shell_command(command='echo hi')]", "shell says hi"]),
                     {"run_shell_command": "hi"})
    thought, resp = bound["_solve_api"](eng, "run echo")
    check(f"{tag}: api loop answers after tool", resp == "shell says hi", repr(resp)[:100])
    check(f"{tag}: api loop 2 chat calls", eng.model.calls == 2, str(eng.model.calls))
    check(f"{tag}: api loop observation in messages",
          any("[OBSERVATION from run_shell_command]" in str(m) for m in eng.model.last_messages),
          str(eng.model.last_messages)[:250])

    # _solve_api_stream
    eng = FakeEngine(FakeAPI(["[TOOL: make_dir(dir_path='d')]", "made it"]),
                     {"make_dir": "Directory ready: d"})
    evs = list(bound["_solve_api_stream"](eng, "make d"))
    kinds = [e["type"] for e in evs]
    check(f"{tag}: api stream tool events + done",
          "tool_start" in kinds and "tool_result" in kinds and kinds[-1] == "done",
          str(kinds))
    check(f"{tag}: api stream final response", evs[-1]["response"] == "made it",
          repr(evs[-1].get("response"))[:100])

# ------------------------------------------------- settings / CLI / web wiring
import json as _json

for path, tag in ((WEB, "web"), (CLI, "cli")):
    src = get_source(path)
    tree = ast.parse(src)
    defaults = None
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and getattr(node.targets[0], "id", "") == "DEFAULT_SETTINGS":
            defaults = ast.literal_eval(node.value)
    check(f"{tag}: settings defaults parse", isinstance(defaults, dict))
    check(f"{tag}: settings allow_file_tools default on", defaults.get("allow_file_tools") is True)
    check(f"{tag}: settings allow_shell_tools default on", defaults.get("allow_shell_tools") is True)

cli_src = get_source(CLI)
for cmd in ("cmd_read", "cmd_write", "cmd_ls", "cmd_mkdir", "cmd_rm", "cmd_run", "cmd_tools"):
    check(f"cli: {cmd} defined", f"def {cmd}(" in cli_src)
for route in ("'/read'", "'/write'", "'/ls'", "'/mkdir'", "'/rm '", "'/run '", "'/tools'"):
    check(f"cli: dispatch {route}", route in cli_src)
check("cli: help lists /run", "/run <command" in cli_src)

web_src = get_source(WEB)
for route in ("/api/workspace/mkdir", "/api/workspace/remove", "/api/workspace/run"):
    check(f"web: endpoint {route}", route in web_src)

for ident in ("You are Ashen",):
    for p, t in ((WEB, "web"), (CLI, "cli")):
        check(f"{t}: identity intact", ident in get_source(p))

print(f"\n{len(PASS)} passed, {len(FAIL)} failed")
sys.exit(1 if FAIL else 0)
