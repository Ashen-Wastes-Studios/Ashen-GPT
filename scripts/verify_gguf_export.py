# -*- coding: utf-8 -*-
"""verify_gguf_export.py — checkpoint-free tests for the Q4_K_M GGUF export in
qwen_finetune.py. No model load, no network: the GGUF block is AST-extracted
from disk and exec'd with stubbed subprocess/urllib/zipfile.

Run from project root:  python scripts/verify_gguf_export.py
"""
import ast
import os
import sys
import tempfile

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(HERE, "qwen_finetune.py")

WANT_FUNCS = {"_is_hf_model_dir", "_gguf_run", "_gguf_pip_install",
              "_gguf_ensure_deps", "_gguf_ensure_converter",
              "_gguf_ensure_quantize_bin", "_gguf_f16_path", "export_gguf"}
WANT_VARS = {"GGUF_ENABLED", "GGUF_QUANT", "_GGUF_OUT_ENV", "GGUF_OUT",
             "GGUF_KEEP_F16", "GGUF_EVERY", "GGUF_TOOLS",
             "_GGUF_CONVERTER_URL", "_GGUF_RELEASE_API"}


def _load_tree():
    with open(SRC, encoding="utf-8") as f:
        return ast.parse(f.read())


def _exec_gguf_block(env):
    """Exec the GGUF assignments + helpers with a controlled environ.

    Returns the namespace dict. `env` is the full os.environ to use.
    """
    tree = _load_tree()
    nodes = []
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name in WANT_FUNCS:
            nodes.append(node)
        elif isinstance(node, ast.Assign):
            targets = [t.id for t in node.targets if isinstance(t, ast.Name)]
            if targets and all(t in WANT_VARS for t in targets):
                nodes.append(node)
    mod = ast.Module(body=nodes, type_ignores=[])
    code = compile(mod, SRC, "exec")
    g = {"__name__": "__gguf_test__"}
    # real stdlib handles the block only needs
    import json
    import subprocess
    import urllib.request
    import zipfile
    g.update({"os": os, "sys": sys, "json": json, "subprocess": subprocess,
              "urllib": urllib, "zipfile": zipfile,
              "HERE": HERE,
              "OUT_DIR": os.path.join(HERE, "ashen_gpt_model"),
              "CLASS_HEAD_PT": os.path.join(HERE, "ashen_gpt_model", "class_head.pt")})
    old_env = dict(os.environ)
    os.environ.clear()
    os.environ.update(env)
    try:
        exec(code, g)  # noqa: S102 — local test harness, file under test only
    finally:
        os.environ.clear()
        os.environ.update(old_env)
    return g


PASS, FAIL = 0, 0


def check(name, cond):
    global PASS, FAIL
    if cond:
        PASS += 1
        print("  ok: %s" % name)
    else:
        FAIL += 1
        print("  FAIL: %s" % name)


def _fake_hf_dir(parent):
    d = os.path.join(parent, "hf")
    os.makedirs(d, exist_ok=True)
    open(os.path.join(d, "config.json"), "w").write("{}")
    open(os.path.join(d, "model.safetensors"), "w").write("x")
    return d


def main():
    global PASS, FAIL
    print("[1] defaults (clean env)")
    g = _exec_gguf_block({"PATH": os.environ.get("PATH", ""), "SYSTEMROOT": os.environ.get("SYSTEMROOT", "")})
    check("GGUF_ENABLED defaults True", g["GGUF_ENABLED"] is True)
    check("GGUF_QUANT defaults Q4_K_M", g["GGUF_QUANT"] == "Q4_K_M")
    check("GGUF_OUT defaults to ashen_gpt_model-Q4_K_M.gguf",
          g["GGUF_OUT"].endswith("ashen_gpt_model-Q4_K_M.gguf"))
    check("GGUF_EVERY defaults 0 (final only)", g["GGUF_EVERY"] == 0)
    check("GGUF_KEEP_F16 defaults False", g["GGUF_KEEP_F16"] is False)

    print("[2] env overrides")
    g2 = _exec_gguf_block({"QWEN_GGUF": "0", "QWEN_GGUF_QUANT": "Q8_0",
                           "QWEN_GGUF_KEEP_F16": "1", "QWEN_GGUF_EVERY": "20"})
    check("QWEN_GGUF=0 disables", g2["GGUF_ENABLED"] is False)
    check("QUANT override renames default OUT",
          g2["GGUF_OUT"].endswith("ashen_gpt_model-Q8_0.gguf"))
    check("KEEP_F16 override", g2["GGUF_KEEP_F16"] is True)
    check("EVERY override", g2["GGUF_EVERY"] == 20)
    g3 = _exec_gguf_block({"QWEN_GGUF_OUT": "D:/m/custom.gguf"})
    check("explicit QWEN_GGUF_OUT wins", g3["GGUF_OUT"] == "D:/m/custom.gguf")

    print("[3] _gguf_f16_path")
    f16 = g["_gguf_f16_path"]
    check("strips -QUANT suffix",
          f16("/m/ashen_gpt_model-Q4_K_M.gguf", "Q4_K_M").replace("\\", "/") == "/m/ashen_gpt_model-f16.gguf")
    check("keeps other suffixes",
          f16("/m/ashen_gpt_model-Q8_0.gguf", "Q4_K_M").replace("\\", "/") == "/m/ashen_gpt_model-Q8_0-f16.gguf")
    check("no-dir relative path", f16("out.gguf", "Q4_K_M") == "out-f16.gguf")

    print("[4] export_gguf command shapes (stubbed tools)")
    tmp = tempfile.mkdtemp(prefix="gguf_verify_")
    hf = _fake_hf_dir(tmp)
    out = os.path.join(tmp, "ashen_gpt_model-Q4_K_M.gguf")
    cmds = []

    def fake_run(cmd):
        cmds.append(cmd)
        # simulate the tool writing its output file (2nd --outfile value / 3rd arg)
        if "--outfile" in cmd:
            p = cmd[cmd.index("--outfile") + 1]
            open(p, "w").write("fake-f16")
        else:
            open(cmd[2], "w").write("fake-q4km")

    g["GGUF_KEEP_F16"] = False
    g["_gguf_ensure_deps"] = lambda: None
    g["_gguf_ensure_converter"] = lambda: os.path.join(tmp, "convert.py")
    g["_gguf_ensure_quantize_bin"] = lambda: os.path.join(tmp, "llama-quantize.exe")
    g["_gguf_run"] = fake_run
    g["GGUF_TOOLS"] = tmp
    ret = g["export_gguf"](hf, out, "Q4_K_M")
    check("returns out path", ret == out)
    check("converter cmd shape",
          cmds[0][:3] == [sys.executable, os.path.join(tmp, "convert.py"), hf]
          and cmds[0][3:] == ["--outfile", os.path.join(tmp, "ashen_gpt_model-f16.gguf"),
                              "--outtype", "f16"])
    check("quantize cmd shape",
          cmds[1][:3] == [os.path.join(tmp, "llama-quantize.exe"),
                          os.path.join(tmp, "ashen_gpt_model-f16.gguf"), out]
          and cmds[1][3] == "Q4_K_M" and cmds[1][4].isdigit())
    check("intermediate f16 removed by default",
          not os.path.exists(os.path.join(tmp, "ashen_gpt_model-f16.gguf")))
    check("quantized file produced", os.path.isfile(out))

    print("[5] GGUF_KEEP_F16=1 keeps intermediate")
    g["GGUF_KEEP_F16"] = True
    out2 = os.path.join(tmp, "m2-Q4_K_M.gguf")
    g["export_gguf"](hf, out2, "Q4_K_M")
    check("f16 kept", os.path.isfile(os.path.join(tmp, "m2-f16.gguf")))

    print("[6] non-HF dir rejected")
    try:
        g["export_gguf"](tmp, os.path.join(tmp, "x.gguf"), "Q4_K_M")
        check("raises on non-HF dir", False)
    except RuntimeError:
        check("raises on non-HF dir", True)

    print("[7] quantize-bin picker (stubbed network)")
    import urllib.request as _urlreq
    g = _exec_gguf_block({"PATH": os.environ.get("PATH", ""),
                          "SYSTEMROOT": os.environ.get("SYSTEMROOT", "")})

    class FakeResp:
        def __init__(self, payload):
            self.payload = payload

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def read(self):
            import json as _j
            return _j.dumps(self.payload).encode()

    rel = {"assets": [
        {"name": "llama-ubuntu-x64.zip", "browser_download_url": "http://x/u.zip"},
        {"name": "llama-b1-bin-win-cpu-x64.zip", "browser_download_url": "http://x/cpu.zip"},
        {"name": "llama-b1-bin-win-cuda-12.4-x64.zip", "browser_download_url": "http://x/cuda.zip"},
    ]}
    tools2 = os.path.join(tmp, "tools2")
    os.makedirs(tools2, exist_ok=True)
    g["GGUF_TOOLS"] = tools2
    seen_urls = []
    real_urlopen, real_retrieve = _urlreq.urlopen, _urlreq.urlretrieve

    def fake_urlopen(url):
        seen_urls.append(url)
        return FakeResp(rel)

    def fake_retrieve(url, dst):
        import zipfile as _zf
        with _zf.ZipFile(dst, "w") as zf:
            zf.writestr("build/bin/llama-quantize.exe", "MZ-fake")
            zf.writestr("build/bin/llama.dll", "fake-dll")

    _urlreq.urlopen, _urlreq.urlretrieve = fake_urlopen, fake_retrieve
    try:
        got = g["_gguf_ensure_quantize_bin"]()
    finally:
        _urlreq.urlopen, _urlreq.urlretrieve = real_urlopen, real_retrieve
    check("picks cuda win asset", seen_urls == [g["_GGUF_RELEASE_API"]])
    check("exe flattened to tools root",
          got == os.path.join(tools2, "llama-quantize.exe") and os.path.isfile(got))
    check("dlls co-extracted", os.path.isfile(os.path.join(tools2, "build/bin/llama.dll"))
          or os.path.isfile(os.path.join(tools2, "llama.dll")))
    # second call: no network (exe now on disk)
    net_calls = []
    _urlreq.urlopen = lambda url: (net_calls.append(url), FakeResp(rel))[1]
    try:
        got2 = g["_gguf_ensure_quantize_bin"]()
    finally:
        _urlreq.urlopen = real_urlopen
    check("cached exe reused", got2 == got and not net_calls)

    print("[8] save_checkpoint + final hooks present in source")
    with open(SRC, encoding="utf-8") as f:
        src = f.read()
    check("periodic hook in save_checkpoint",
          "if GGUF_ENABLED and GGUF_EVERY and it % GGUF_EVERY == 0" in src)
    check("final export after last checkpoint",
          "model = save_checkpoint(model, MAX_ITERS)\nif GGUF_ENABLED:" in src)
    check("export-only CLI branch exits", '"--export-gguf-only" in sys.argv' in src
          and "sys.exit(0)" in src)
    check("export failure never kills training (try/except at final)",
          "GGUF export FAILED" in src)

    print("\n%d passed, %d failed" % (PASS, FAIL))
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
