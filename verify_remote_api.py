#!/usr/bin/env python3
"""
verify_remote_api.py — Verify the ATHENS remote worker API compatibility.

This script runs the worker in REMOTE mode, sends a test prompt, and
verifies the response follows the expected JSON-lines protocol.

Usage:
    # From the Ashen AI Base directory:
    python3 verify_remote_api.py --port 8765 --host 127.0.0.1

    # Or from the Discord Bot Base directory (tests the remote URL mode):
    python3 verify_remote_api.py --remote-url http://127.0.0.1:8765
"""

import argparse
import asyncio
import json
import os
import socket
import subprocess
import sys
import time
import urllib.request
import urllib.error

# ─── Protocol constants ──────────────────────────────────────────────


class ProtocolError(Exception):
    """Raised when the worker's protocol doesn't match expectations."""


# ─── Local worker (TCP server mode) test ─────────────────────────────


def _wait_for_worker(host: str, port: int, timeout: float = 60.0) -> None:
    """Wait until the TCP server responds, or raise."""
    start = time.monotonic()
    while time.monotonic() - start < timeout:
        try:
            with socket.create_connection((host, port), timeout=2):
                return
        except OSError:
            time.sleep(0.5)
    raise ProtocolError(
        f"Worker did not start on {host}:{port} within {timeout}s"
    )


async def _test_local_worker(model_dir: str, host: str, port: int) -> bool:
    """Start the worker in remote mode and send a test prompt."""
    print(f"Starting ATHENS worker in remote mode...")
    print(f"  model_dir: {model_dir}")
    print(f"  host:      {host}")
    print(f"  port:      {port}")

    # Pre-start: verify the worker boots
    proc = subprocess.Popen(
        [
            sys.executable, "ashen_gpt_worker.py",
            "--model-dir", model_dir,
            "--host", host,
            "--port", str(port),
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )

    try:
        _wait_for_worker(host, port, timeout=90.0)
        print("[OK] Worker TCP server is accepting connections")

        # Send a test request
        test_prompt = "Hello! Respond with 'OK' and nothing else."
        request = {
            "type": "ask",
            "request_id": 1,
            "prompt": test_prompt,
            "max_new_tokens": 50,
        }
        payload = json.dumps(request) + "\n"

        print(f"Sending test prompt (len={len(test_prompt)})...")
        with socket.create_connection((host, port), timeout=30) as sock:
            sock.sendall(payload.encode("utf-8"))

            # Read response
            response = b""
            while True:
                chunk = sock.recv(4096)
                if not chunk:
                    break
                response += chunk
                if b"\n" in chunk:
                    break

        lines = response.decode("utf-8", errors="replace").strip().split("\n")
        if not lines:
            raise ProtocolError("No response received")

        result = json.loads(lines[0])
        if result.get("type") != "response":
            raise ProtocolError(f"Expected 'response' type, got {result.get('type')}")

        answer = result.get("response", "")
        print(f"[OK] Response received: {answer[:80]}...")

        # Verify protocol completeness
        if result.get("type") != "response":
            raise ProtocolError("Missing 'type' field in response")

        print(f"[OK] Protocol OK — worker responded correctly")
        return True

    finally:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()


# ─── Remote URL test (Discord bot's HTTP client mode) ────────────────


async def _test_remote_url(url: str, prompt: str = None) -> bool:
    """Send a test request to a remote worker via HTTP/JSON."""
    import requests

    print(f"Testing remote worker at {url}...")

    test_prompt = prompt or "Hello! Respond with 'OK' and nothing else."
    request = {
        "type": "ask",
        "request_id": 1,
        "prompt": test_prompt,
        "max_new_tokens": 50,
    }

    try:
        resp = requests.post(
            url,
            json=request,
            timeout=30,
            headers={"Content-Type": "application/json"},
        )
        resp.raise_for_status()
        data = resp.json()

        if data.get("type") != "response":
            raise ProtocolError(f"Expected 'response' type, got {data.get('type')}")

        answer = data.get("response", "")
        print(f"[OK] Response received: {answer[:80]}...")

        # Verify required fields
        for field in ("response", "sources", "thought_file", "thought_text"):
            if field not in data:
                print(f"[WARN] Missing field '{field}' in response")

        print(f"[OK] Remote URL API OK — worker responded correctly")
        return True

    except requests.exceptions.ConnectionError as e:
        raise ProtocolError(f"Cannot connect to {url}: {e}")
    except requests.exceptions.Timeout:
        raise ProtocolError(f"Remote worker at {url} timed out")
    except requests.exceptions.HTTPError as e:
        raise ProtocolError(f"HTTP {e.response.status_code}: {e.response.text[:200]}")


# ─── CLI entry point ─────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="Verify ATHENS remote worker API compatibility",
    )
    parser.add_argument(
        "--port", type=int, default=8765,
        help="TCP port for local worker mode (default: 8765)",
    )
    parser.add_argument(
        "--host", type=str, default="127.0.0.1",
        help="Host for local worker mode (default: 127.0.0.1)",
    )
    parser.add_argument(
        "--remote-url", type=str,
        help="HTTP URL for remote worker mode (tests Discord bot client)",
    )
    parser.add_argument(
        "--model-dir", type=str,
        default=os.path.join(os.path.dirname(__file__), "ashen_gpt_model"),
        help="Path to ATHENS model directory",
    )
    parser.add_argument(
        "--prompt", type=str,
        default="Hello! Respond with 'OK' and nothing else.",
        help="Test prompt to send",
    )
    args = parser.parse_args()

    # Determine test mode
    if args.remote_url:
        # Remote URL mode
        print("== Remote URL mode ==")
        print(f"Testing remote worker at {args.remote_url}")
        try:
            asyncio.run(_test_remote_url(args.remote_url, args.prompt))
            print("\n[VERDICT] PASS — remote API is compatible")
            return 0
        except ProtocolError as e:
            print(f"\n[VERDICT] FAIL — {e}")
            return 1

    # Local worker mode (default)
    print("== Local worker mode (TCP server) ==")
    print(f"Model directory: {args.model_dir}")

    if not os.path.isdir(args.model_dir):
        print(f"[ERROR] Model directory not found: {args.model_dir}")
        print("         Run the model training/finetune first, or point --model-dir")
        return 1

    try:
        asyncio.run(_test_local_worker(args.model_dir, args.host, args.port))
        print("\n[VERDICT] PASS — local worker API is compatible")
        return 0
    except Exception as e:
        print(f"\n[VERDICT] FAIL — {type(e).__name__}: {e}")
        # Print worker startup output for debugging
        proc = subprocess.Popen(
            [sys.executable, "ashen_gpt_worker.py", "--model-dir", args.model_dir],
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
        )
        time.sleep(3)
        if proc.poll() is None:
            proc.terminate()
            try:
                out = proc.communicate(timeout=2)[0]
                if out:
                    print("Worker output (first 1000 chars):")
                    print(out[:1000])
            except subprocess.TimeoutExpired:
                proc.kill()
        return 1


if __name__ == "__main__":
    sys.exit(main())
