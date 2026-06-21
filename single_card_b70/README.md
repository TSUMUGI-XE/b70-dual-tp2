# A ladder for LLM serving on Intel Arc Pro B70 — single stream → single-card concurrency → both cards

> *Most Arc-for-LLM discussion online jumps straight to multi-GPU and gets stuck on "is dual-B70 broken." But the useful thing to write down first is the **ladder**: as you go from one request, to many concurrent requests, to a model too big for one card, the *right tool changes* — and most people never climb past the first rung, leaving 1.6–2x on the table. So this is the whole ladder, every number measured by me on Arc Pro B70, reproducible from this repo, with the one idea (decode here is overhead-bound, not bandwidth-bound) that explains why each rung has a different winner.*
>
> *English isn't my first language, so I drafted this with AI help — but every result below was measured on the hardware described, and the launch scripts and patches are in this repo. Reproduce it and tell me if anything's off.*

## TL;DR — the ladder

Model throughout: **Qwen3.6-35B-A3B**, a small-active MoE (~3B active params/token) — the sweet spot for a 32 GB B70 (quality of a 35B, decode cost near a 3B).

| Rung | Question you're asking | Winner | Headline number (one B70 unless noted) |
|---|---|---|---|
| **1 — single stream** | "one user, lowest latency" | **llama.cpp** (SYCL build) | **44 → 72.6 tok/s** rebuilding OpenCL→SYCL; **123 tok/s** with the vLLM+MTP squeeze |
| **2 — single-card concurrency** | "many users / a batch, on one card" | **vLLM** (continuous batching) | **~800 tok/s aggregate @ N=64**, scales ~linearly |
| **3 — both cards** | "model too big for 32 GB, or pool capacity" | **vLLM host-staged TP=2** | **runs today** (capacity, not speed); native P2P = open frontier |

**The single most important line in this whole document is rung 1's first half: a 1.6x decode speedup from nothing but rebuilding llama.cpp against the SYCL backend instead of OpenCL.** If you do one thing, do that.

Everything else follows from one fact, established in rung 1 and reused all the way up:

> On this card, single-stream decode is **overhead/dispatch bound, not memory-bandwidth bound.** The GPU spends most of each token *waiting* on per-layer launches, not *streaming weights*.

That's why a smaller quantization barely helps, why graph capture helps a lot, why speculative decoding is the only lever that breaks the per-forward-pass ceiling, and why two cards buy *capacity* but not single-stream *speed*. Keep it in your head and you can predict each knob before you turn it.

---

## Rung 1 — single stream (one user, lowest latency) → llama.cpp

### The free 1.6x: SYCL backend, not OpenCL

llama.cpp can drive an Arc GPU two ways: the **OpenCL** backend (the zero-config default, `ONEAPI_DEVICE_SELECTOR=opencl:gpu`) and the **SYCL** backend (a separate build). They are not the same speed.

One B70, Qwen3.6-35B-A3B at Q6_K, single stream:

```
OpenCL backend : 44.5 tok/s
SYCL backend   : 72.6 tok/s   (build b9738)
```

**Same card, same model, same `-ngl 99 -fa on` — 1.6x from the backend alone.** No quantization change, no tricks. If you're on the OpenCL path (very common, because it's the no-config one), you're leaving a third of your tokens on the floor.

```bash
source /opt/intel/oneapi/setvars.sh
cmake -B build-sycl -DGGML_SYCL=ON -DCMAKE_C_COMPILER=icx -DCMAKE_CXX_COMPILER=icpx
cmake --build build-sycl --config Release -j
ONEAPI_DEVICE_SELECTOR=level_zero:0 \
  ./build-sycl/bin/llama-server -m <model>.gguf -ngl 99 -fa on --port 8000
```

(Scripts: `scripts/llama_sycl_build.sh`, `scripts/llama_serve_1card.sh`.)

### Quantization barely helps — and that tells you *why* decode is slow

The natural next move is Q6 → Q4. Same card, same SYCL build:

```
Q6_K   : 72.6 tok/s   (≈6.5 bit)
Q4_K_M : 77.4 tok/s   (≈4.5 bit)
```

**+6.6%** for a model **32% smaller on disk.** If single-stream decode were **bandwidth bound**, 1.32x smaller weights would decode ~1.32x faster. It doesn't — **+6.6% ≠ +32%** — so decode here is **overhead/dispatch bound.** That's the fact the whole ladder rests on. The practical corollary: the ceiling for "make each forward pass cheaper" is ~**77 tok/s** on this card for 35B-A3B (4-bit and 6.5-bit both flat-line near it). To pass it you need a *different* lever, not a smaller model.

### The single-stream ceiling-breaker (advanced): vLLM cudagraph + speculative decode

llama.cpp at 72.6 is the simple answer. If you want the absolute single-stream floor, vLLM-XPU can go further — but it's more setup, so treat this as the "squeeze every last token" tier.

- **First, get the XPU graph on.** vLLM in default eager mode decodes 35B-A3B at only ~10–11 tok/s — pure per-layer dispatch tax, the overhead-bound regime laid bare. Enabling XPU cudagraph (`VLLM_XPU_ENABLE_XPU_GRAPH=1` + a one-line torch device-sync fix, pytorch#187277, since fixed) takes it to **~68–72 tok/s (≈4.4x over eager)** — on par with llama.cpp SYCL. *Without the env flag the graph is silently skipped.*
- **Then add MTP (speculative / multi-token decode).** In an overhead-bound regime where the GPU mostly idles, verifying several drafted tokens per forward pass is nearly free. The catch on Intel: MTP and *FULL* cudagraph capture crash at init (dynamic spec shapes vs. static graph). The fix is the **`cudagraph_mode: PIECEWISE`** capture mode, which keeps the dynamic part eager and graphs the rest:

```
FULL cudagraph, no MTP              : 72.0 tok/s
PIECEWISE, no MTP                   : 71.3 tok/s   (≈ FULL, ~1% overhead)
PIECEWISE + MTP(n=3) + prefix cache : 123.5 tok/s  (1.71x)
```

**123.5 tok/s single-stream on one B70 for a 35B-class model, coherent, prefix caching live.** Honest caveats: n-sweep (draft length) is noisy at single-shot — n2:103 / n3:123 / n4:104 / n5:118 / n6:114, so pick **n≈3** and don't read structure into the wiggle. And **MTP is a latency lever only** — it *lowers* concurrent throughput (it reserves token budget, dropping the concurrency cap ~26→7), which is exactly why rung 2 turns it off.

(Patch stack for this tier: cudagraph enablers above + cherry-picks #45614 and #44986 for MTP×prefix-cache coexistence + a 2-line XPU pointer signed-wrap fix in `mamba_utils.py`. MXFP4 MoE on XPU also needs self-converted weights — no off-the-shelf MXFP4 on HF — and an `ssm16` graft keeping the SSM gate projections at BF16, or long context drifts. The runnable serve command (the piecewise + MTP + prefix-cache flags) is `scripts/vllm_serve_1card_mtp.sh`; the image itself is a custom oneAPI 2026 / vLLM 0.23 build with the patches above, not a one-file Dockerfile — build it separately.)

---

## Rung 2 — single-card concurrency (many users / a batch) → vLLM

The moment you have *more than one request in flight*, the metric changes from latency to **aggregate throughput**, and the winner flips from llama.cpp to **vLLM with continuous batching.**

llama.cpp serves one stream at a time well, but doesn't multiplex — a second request waits. vLLM batches concurrent requests into each forward pass, so aggregate throughput climbs ~linearly with concurrency until the card saturates. On one B70, 35B-A3B (cudagraph, **MTP off**):

```
N=8   : ~85 tok/s aggregate
N=16  : ~80–160 tok/s aggregate (scaling)
N=64  : ~800 tok/s aggregate
```

That ~**800 tok/s at N=64 on a single 32 GB card** is the rung-2 headline — about 11x the single-stream rate, because the overhead-bound idle time that hurts batch-of-one is exactly what concurrency fills. This is the "shared endpoint / eval sweep / offline batch" rung. Turn MTP **off** here: speculative decode buys single-stream latency at the cost of concurrent capacity, so on this rung it's a net loss.

So the rung-1-vs-rung-2 decision is not "which is faster" but "which axis is my workload on":

| Workload | Rung | Stack | Why |
|---|---|---|---|
| One user, interactive chat | 1 | llama.cpp SYCL (72.6), or vLLM+MTP (123) | single-stream latency |
| Many users / batch / eval | 2 | vLLM continuous batching, MTP off | aggregate throughput scales with N |
| Model too big for 32 GB | 3 | both cards (below) | capacity, not speed |

---

## Rung 3 — both cards (a model too big for 32 GB, or pooling) → vLLM host-staged TP=2

If the model doesn't fit in 32 GB, or you want to pool both cards' capacity, you need tensor parallelism across the two B70 — `vllm serve -tp 2`. This is where the internet says "dual-B70 is broken." The honest state:

- **`vllm serve -tp 2` hangs** out of the box — the cross-GPU collective tries a direct GPU↔GPU peer copy, which a consumer root complex refuses (`0x70000003`, a clean error, not silent corruption).
- **It runs today via host-staged collectives** — route the all-reduce through host RAM (gloo, or oneCCL's host path) instead of peer copy. A model that fits on *no* single 32 GB card then loads and serves across the pair. This is the **capacity unlock**, and it's reproducible.
- **It is *not* a single-stream speed win.** Host-staged TP=2 single-stream is *slower* than one card alone (the extra host round-trip), e.g. 35B-A3B ~7.5 tok/s TP=2 vs ~11 tok/s on one card. Use rung 3 for **capacity / pooling**, never to make one request faster. For latency, stay on rung 1.

### The open frontier: native peer-to-peer (PLX, unverified)

Host-staged works but pays a host round-trip. The *clean* fast path is native GPU↔GPU P2P — which on this consumer box is refused at the PCIe root complex (the two cards hang off separate root ports with no shared switch; the kernel's `pci_p2pdma` only whitelists server/Xeon host bridges, not consumer Intel). The expected physical fix is to put **both cards under one PCIe switch (a PLX/PEX board)** so the peer copy never traverses the root complex — but **I haven't verified that yet** (switch on the way). Until then, host-staged is the answer that works.

The full diagnostic — the hang teardown, the `0x70000003` probe, the three working host-staged routes, the AMD-vs-Intel kernel whitelist, and the PLX plan — is its own (much deeper) writeup:

→ **[dual-B70 tensor-parallel writeup](../README.md)**

If you have one card, you never touch any of this. Start at rung 1.

---

## Why each rung has a different winner — one paragraph

Single-stream decode on this card is **overhead/dispatch bound, not bandwidth bound** (32% smaller weights bought only 6.6% more tokens). So:

- **Rung 1 (latency):** kill per-launch overhead. SYCL > OpenCL backend (1.6x); cudagraph replays many launches as one (4.4x over eager); MTP emits multiple tokens per pass — the only lever past the "cheaper forward" ceiling (+71%).
- **Rung 2 (throughput):** that same idle-while-waiting time is *free capacity* — continuous batching fills it, so aggregate scales ~linearly to ~800 tok/s @ N=64. (MTP off: it trades this capacity for single-stream latency.)
- **Rung 3 (capacity):** two cards add VRAM, not single-stream speed — the host-staged path even costs a little latency. It's for models that don't fit, not requests that are slow.

---

*Measured on Arc Pro B70 (32 GB), `xe` driver, recent Linux kernel. Single-stream numbers are batch-of-one; n-sweep figures are single-shot, ±~10% noisy. Rung-2 aggregates are continuous-batching, MTP off. Rung-3 TP=2 is host-staged (capacity, not speed); native P2P is unverified pending a PCIe switch. Host is a consumer platform (single root complex, PCIe Gen4 x8 to each card) — single-card decode is overhead bound so the host link isn't the rung-1/2 limiter. Corrections welcome.*
