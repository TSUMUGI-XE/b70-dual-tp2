#!/usr/bin/env python3
# APC test: cross-request prefix reuse when several requests share a long system
# prefix and differ only in a short tail. Checks cached_tokens / prefill time on reuse.
import json, time, urllib.request

URL = "http://localhost:8001/v1/chat/completions"
MODEL = "qwen/qwen3.6-27b-awq"

# Long shared system prefix (generic assistant guidelines), inflated by repetition (~thousands of tokens).
BASE = """You are a careful coding assistant. Follow these rules strictly.
- Answer concisely; state any assumptions explicitly.
- Prefer correct, idiomatic code over clever code.
- Point out edge cases and failure modes.
- Give the conclusion first, then a short justification.
"""
SHARED_SYSTEM = (BASE * 30).strip()  # shared prefix of a few thousand tokens

def ask(tail, label):
    body = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": SHARED_SYSTEM + "\n\n" + tail},
            {"role": "user", "content": "Review this function: def add(a, b): return a - b"},
        ],
        "max_tokens": 40, "temperature": 0.3,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    data = json.dumps(body).encode()
    t0 = time.time()
    req = urllib.request.Request(URL, data=data, headers={"Content-Type": "application/json"})
    d = json.load(urllib.request.urlopen(req, timeout=180))
    dt = time.time() - t0
    u = d.get("usage", {})
    cached = (u.get("prompt_tokens_details") or {}).get("cached_tokens")
    print(f"[{label}] total={dt:.2f}s prompt_tokens={u.get('prompt_tokens')} cached_tokens={cached} completion={u.get('completion_tokens')}")
    return dt, u.get("prompt_tokens"), cached

print("shared system chars:", len(SHARED_SYSTEM))
print("=== #1 (focus: performance, cold) ===")
ask("Focus your review on performance.", "req#1 performance")
print("=== #2 (focus: readability, same shared prefix) ===")
ask("Focus your review on readability.", "req#2 readability")
print("=== #3 (focus: performance again, full revisit) ===")
ask("Focus your review on performance.", "req#3 performance-again")
