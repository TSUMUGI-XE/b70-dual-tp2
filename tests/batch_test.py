#!/usr/bin/env python3
# Continuous-batching scaling: compare single vs N-concurrent aggregate decode t/s.
import json, time, urllib.request
from concurrent.futures import ThreadPoolExecutor

URL = "http://localhost:8001/v1/chat/completions"
MODEL = "qwen/qwen3.6-27b-awq"
MAXTOK = 128

def one(i):
    body = {
        "model": MODEL,
        "messages": [{"role": "user", "content": f"Item {i}: give three benefits of running a local LLM, briefly."}],
        "max_tokens": MAXTOK, "temperature": 0.7,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    data = json.dumps(body).encode()
    req = urllib.request.Request(URL, data=data, headers={"Content-Type": "application/json"})
    t0 = time.time()
    d = json.load(urllib.request.urlopen(req, timeout=300))
    dt = time.time() - t0
    ct = d.get("usage", {}).get("completion_tokens", 0)
    return dt, ct

def run(n):
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=n) as ex:
        res = list(ex.map(one, range(n)))
    wall = time.time() - t0
    toks = sum(c for _, c in res)
    per = [c / dt for dt, c in res if dt > 0]
    agg = toks / wall
    print(f"concurrency={n:2d}: wall={wall:5.1f}s  completed_tokens={toks:4d}  "
          f"aggregate={agg:5.1f} t/s  per-stream avg={sum(per)/len(per):4.1f} t/s")
    return agg

print("=== warmup (JIT) ===")
one(99)
print("=== batching scaling ===")
a1 = run(1)
a8 = run(8)
print(f"\n-> 8-way aggregate / single = {a8/a1:.1f}x")
