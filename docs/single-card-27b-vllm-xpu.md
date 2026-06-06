# Running Qwen3.6-27B (hybrid/GDN) on a single Intel Arc Pro B70 with vLLM-XPU 0.21 — prefix caching + continuous batching, built from source

> *English isn't my first language, so I drafted this with AI help — but every result below was measured and verified by me on the hardware described. The patches and scripts are in this repo: reproduce it, and tell me if anything's off.*

## TL;DR

I got **Qwen3.6-27B-AWQ** — a *hybrid* (Mamba/GDN + attention) model — serving on a **single Intel Arc Pro B70 (32 GB)** under **vLLM-XPU 0.21**, with **automatic prefix caching (APC)** and **continuous batching** both working. As far as I can tell there is **no prior public report** of a Qwen3.6 hybrid model running on vLLM-XPU on Battlemage — the community consensus has been "Qwen3.6 on Arc = llama.cpp only." That consensus is now stale for the dense 27B tier.

Measured on the box:

| Metric | Result |
|---|---|
| Single-stream decode | ~8 tok/s (llama.cpp/OpenCL on the same card does ~17 tok/s — vLLM loses here) |
| 8-way concurrent decode | **~53 tok/s aggregate (6.4×)** — continuous batching wins big |
| APC: 2nd request, shared 5 KB system prompt | first **35.6 s → 3.8 s (9.4×)**, server prefix-cache-hit 48% |
| Quality (AWQ, thinking off) vs llama.cpp Q6 | within scenario variance (no degradation), 0 thinking-leak |
| Model load | 19.8 GB resident, ~3 min cold start (weights + Triton JIT) |

The takeaway: **single-stream, llama.cpp is still faster**; the reason to run vLLM-XPU is **cross-request prefix reuse + batched concurrency**, which llama.cpp's recurrent/hybrid path structurally can't do. If you fan out several agents/requests that share a long system prompt, vLLM-XPU is a large net win.

## Why this was hard (and why no one had reported it)

Two myths had to die:

1. **"The FMHA `Unsupported page size 784` wall is fundamental."** It isn't — it was an *artifact of old images*. Qwen3.6 is hybrid, so the attention page size has to relate to the Mamba page size. Old vLLM-XPU builds (≈`intel/vllm:0.17.0-xpu`, dev-Feb/Mar 2026) forced the *attention* `block_size` to a weird 784 so `attention_page_bytes == mamba_page_bytes`, and the compiled XPU FMHA kernel rejected 784. Reading vLLM 0.20/0.21 source shows the strategy changed: attention `block_size` stays at the normal value (default 16), the *Mamba* page is padded instead (`mamba_page_size_padded`), and `gpu_model_runner` decouples a backend-chosen `kernel_block_size` from the logical block size — so the FMHA kernel **never sees 784**. The wall is structurally gone in current vLLM.
2. **"35B-A3B AWQ would OOM."** Also outdated — vLLM 0.21 routes AWQ MoE through `moe_wna16` (quantized fused experts), not the old unquantized path that OOM'd. (The 35B has a *different* blocker; see below.)

The real work was a chain of **XPU runtime gaps**, each small, each fatal until fixed.

## Hardware & host

- **2× Intel Arc Pro B70** (BMG-G31, 32 GB GDDR6, ~608 GB/s each). One card per model; **no tensor parallelism** (see "What doesn't work").
- **Ubuntu 26.04**, kernel **7.0.0-15**, **xe** driver, compute-runtime **26.05** on the host.
- Model: `QuantTrio/Qwen3.6-27B-AWQ` (arch `Qwen3_5ForConditionalGeneration`, awq/qwen3_5; hybrid GDN+attention).

## Building the image

vLLM-XPU 0.21 isn't shipped in a ready B70 image (Intel's `llm-scaler` PV still bundles vLLM 0.14.0; `intel/vllm` is 0.17 — both too old for Qwen3.6). So build from source with the official Dockerfile:

```bash
git clone --depth 1 --branch v0.21.0 https://github.com/vllm-project/vllm
cd vllm
docker build -f docker/Dockerfile.xpu --target vllm-base -t vllm-xpu:0.21 .
```

Baked stack (what the image ends up with):

- oneAPI 2025.3.2, compute-runtime 25.48.36300.8 + IGC 2.24.8 + level-zero 1.26 **inside the image**
- torch **2.11.0+xpu**, `vllm 0.21.0+xpu`, `vllm_xpu_kernels` **0.1.7**

The XPU build is fast (~seconds) because it uses the prebuilt kernels wheel rather than compiling CUDA. Persist the Triton cache (`-v /tmp/triton_cache:/root/.triton`) — first run JIT-compiles ~1000 kernels.

## The four fixes

Serve command shape:

```bash
docker run --device /dev/dri \
  -e ZE_AFFINITY_MASK=0 \
  -e ZES_ENABLE_SYSMAN=1 \
  -e VLLM_USE_TRITON_AWQ=1 \
  -v /tmp/triton_cache:/root/.triton \
  vllm-xpu:0.21 \
  vllm serve /models/Qwen3.6-27B-AWQ \
    -tp 1 --enforce-eager \
    --max-model-len 16384 --gpu-memory-utilization 0.90 \
    --enable-prefix-caching
```

1. **`ZES_ENABLE_SYSMAN=1`** — without it, device/sysman init fails on XPU.
2. **`torch.xpu.mem_get_info` shim** — vLLM queries free/total device memory via `torch.xpu.mem_get_info`, which wasn't wired up; inject a small shim (e.g. via `sitecustomize.py` on `PYTHONPATH`) that returns the Level-Zero `mem_get_info` values.
3. **`VLLM_USE_TRITON_AWQ=1`** — route AWQ through the Triton kernels (the XPU path), instead of a CUDA-only dequant.
4. **Cherry-pick [PR #41995](https://github.com/vllm-project/vllm/pull/41995)** — this is the one that unlocks **APC**. With prefix caching on, Mamba align-mode runs `collect_mamba_copy_meta`, which builds src/dst pointer arrays. On XPU the device pointers can be ≥ 2⁶³, and the int64 view overflows → `OverflowError` (issue #41817). PR #41995 views the pointer arrays as `np.uint64`. Mount the patched `mamba_utils.py` over the install (or bake it). **After this patch, APC works** — and cross-request prefix reuse is the entire reason to be here.

## Benchmarks

**APC (the headline).** One ~5 KB shared system prompt, several requests that differ only in a short one-line tail:

```
1st request : 35.6 s   (cold prefill)
2nd request :  3.8 s   (9.4× — prefix reused)
3rd request :  3.7 s
server-reported prefix-cache hit: 48%
```

This is exactly the reuse llama.cpp's hybrid/recurrent KV layout can't give you across requests.

**Continuous batching.** 8 concurrent decodes aggregate to **~53 tok/s (6.4×** over single-stream ~8 tok/s**)** — plenty of headroom to fan out multiple agents at once.

**Quality.** AWQ with `enable_thinking:false`, fact-recall scenarios vs a llama.cpp Q6 baseline: difference within scenario variance (no measurable degradation), and 0 reasoning-trace leakage. Note: on this model `enable_thinking:false` (via `chat_template_kwargs`) is required — `/no_think` is a no-op on the vLLM 0.21 qwen3_5 template.

## Production

Patches baked into a pinned image (`FROM vllm-xpu:0.21` + COPY the #41995 patch + the mem shim), run under a systemd user unit on `:8001`, `Restart=on-failure`, `TimeoutStartSec=600` for the ~3-min cold load. One B70 runs this; the other runs a 35B-A3B MoE under llama.cpp/OpenCL. Endpoint contract, one model per card.

## What does NOT work (yet)

- **35B-A3B AWQ (MoE) on vLLM-XPU** — blocked by a capability gate, *not* OOM. `XPUPlatform.get_device_capability()` returns `None` by design, so `moe_wna16` computes `device_capability = -1`, and `-1 < AWQConfig.get_min_capability()` → it refuses to select the quantized MoE method. The upstream tracker for fixing this on XPU is **[RFC #37979](https://github.com/vllm-project/vllm/issues/37979)** (Intel Quantization Roadmap H1 2026): wNa16 Linear layers are merged, **MoE layers are still "Planned."** When MoE wNa16 lands, the 35B should move to vLLM and pick up the same batching win.
- **Tensor parallelism across the two B70s (TP=2)** — blocked. The vLLM-side symptom is [#41663](https://github.com/vllm-project/vllm/issues/41663) (GP-fault), and underneath it the kernel-side cross-GPU P2P bug **drm-xe #8022** (`vm_bind` returns `-EINVAL` for a peer-imported buffer on BMG-G31). #8022 was *closed* (2026-05-24, pointing at Vivek Kasireddy's P2P dma-buf RFC), but a shipping, B70-verified fix + a multi-process Level-Zero fix are still needed before TP across two cards is real. For now: **one model per card, no inter-card traffic.** This is also the *better* design on a PCIe-3.0-class host — TP decode is latency-bound on all-reduce, which the bus can't help.

## Reproduction checklist

1. Build `Dockerfile.xpu --target vllm-base` from tag `v0.21.0`.
2. Env: `ZE_AFFINITY_MASK=<card>`, `ZES_ENABLE_SYSMAN=1`, `VLLM_USE_TRITON_AWQ=1`.
3. Shim `torch.xpu.mem_get_info`.
4. Cherry-pick PR #41995 for APC.
5. Serve `-tp 1 --enforce-eager --enable-prefix-caching`; persist the Triton cache.
6. Send requests with `chat_template_kwargs={"enable_thinking": false}`.

## Open questions for anyone with the hardware

- Does the same recipe work for **35B-A3B** once `moe_wna16` MoE lands on XPU?
- Anyone reproduced cross-GPU P2P on B70 post-#8022 on a recent kernel?
- `--enforce-eager` is on for stability — has anyone gotten the XPU graph path stable for this model?

---

*Built and measured on 2× Arc Pro B70, Ubuntu 26.04 / kernel 7.0.0-15 / xe. If you're on Battlemage and chasing Qwen3.6 — it works; ping me.*
