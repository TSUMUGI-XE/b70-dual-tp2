# Tensor-parallel LLM serving across two Intel Arc Pro B70 — three working routes, and why the "it's broken upstream" answer was wrong

> *I spent a long time fighting this — and given how often "Battlemage multi-GPU is just broken, wait for upstream" comes up, I figure others are stuck in the same place. So this is the writeup of that struggle: what actually failed, what I tried, and the three routes that finally worked — collected so the next person doesn't have to start from zero.*
>
> *English isn't my first language, so I drafted this with AI help — but every result below was measured by me on the hardware described, and the patches, probes and launch scripts are in this repo. Reproduce it and tell me if anything's off.*

## TL;DR

Two **Intel Arc Pro B70 (32 GB each, ~$1,900 for the pair — $949 list per card)** are widely assumed to be unable to do tensor parallelism: `vllm serve -tp 2` hangs, and the standing answer in the community has been *"cross-GPU on Battlemage is broken, wait for upstream."*

That answer is wrong — or rather, it diagnoses the wrong layer. After taking the hang apart I have **three independent ways to run a single model across both B70**, all reproducible from this repo:

| Route | How | Single-stream | Peak aggregate | Power / temp | Depends on |
|---|---|---|---|---|---|
| **native vLLM TP=2** | legacy L0 V1 + oneCCL host collectives + topo-check off | 5.4 tok/s | **180 tok/s @ N=64** (33×) | 346 W / 78 °C | a few env flags |
| **fusion (this repo)** | drop oneCCL, route the all-reduce through host gloo | 4.4 tok/s | **84 tok/s @ N=32** (21×) | ~70 °C | **gloo only — HW/driver independent** |
| **llama.cpp layer-split** | `-sm layer` across both cards, host-staged dev2dev | ~50 tok/s | 44 tok/s (single-slot) | — | automatic fallback |

**Read the single-stream column first.** None of these makes one request *faster* — peer latency is the floor (see below). What they buy you is **throughput** (batched/offline jobs) and **capacity** (a 70B-class model that does not fit on one 32 GB card). If you want low-latency chat, run one model per card; that's still the right design. If you want to push a big batch through, or run a model too big for one card, on hardware that cost a quarter of the equivalent NVIDIA box — that's what this is for.

**Concretely:** a **Qwen2.5-72B-AWQ** that fits on *no* single 32 GB card loads and serves across the pair at **~119 tok/s aggregate** — the capacity unlock in one number. What that does and doesn't buy you is the next section.

## The diagnostic reversal (this is the actually-useful part)

If you only take one thing from this repo, take the debugging arc, not the numbers.

1. **Symptom.** `vllm serve -tp 2` hangs at the oneCCL OFI/ATL handshake. Swapping the host CPU/board (Z390 → Z490M, separate root ports, Gen4 x8/x8, separate IOMMU groups) reproduces the hang at the *exact same point* — a clean negative control that rules out PCIe bandwidth and root-port layout as the cause.
2. **The wrong conclusion.** "oneCCL is broken on Battlemage, nothing to do but wait." This is where most reports stop.
3. **The re-diagnosis.** It isn't oneCCL the *library*. The thing that's special about vLLM TP is that it runs **one process per GPU**, so every collective crosses an L0/xe device boundary *between processes*. llama.cpp, which puts both GPUs in **one process** with per-device contexts, doesn't hit the hang. So the fault is in the cross-device path, not the transport library.
4. **The probe.** [`repro/b70_p2p_copy_probe.cpp`](repro/b70_p2p_copy_probe.cpp) is ~80 lines: open both GPUs with per-device L0 contexts (the faithful single-process multi-GPU pattern), then attempt a direct device-to-device `zeCommandListAppendMemoryCopy`. It returns **`0x70000003`** — a *clean error, not a hang* — while a host-staged copy of the same buffers passes. Same result under both the L0 V2 adapter and legacy V1 (`SYCL_UR_USE_LEVEL_ZERO_V2=0`). So: **direct peer copy fails; host-staged copy works.**
5. **The bypass.** Every working route above is the same trick — *don't do the peer copy.* Route the reduction through host memory (gloo, or oneCCL's host collectives, or llama.cpp's automatic host-staged dev2dev). The hang was never inevitable; it was the stack insisting on a peer path that this configuration won't serve.

The whole thing in one picture:

```
  Peer path — what every TP stack tries first:
      [B70 #0] ──✗──► [B70 #1]    zeCommandListAppendMemoryCopy → 0x70000003
                                   (kernel P2PDMA refuses: no shared bridge / not whitelisted)

  Host-staged path — what all three working routes do instead:
      [B70 #0] ──► host RAM ──► [B70 #1]    ✓  gloo · oneCCL-host · llama.cpp fallback
                                   (an extra round-trip; batching amortizes it)
```

### What the root cause of the peer-copy failure is — settled: the kernel refuses the peer path, and it isn't my host config

It is tempting to file the `0x70000003` as an xe driver bug and call it a day. Before doing that I had one confound to rule out: this box runs `pcie_acs_override=downstream,multifunction` on the kernel command line (added to split IOMMU groups for an unrelated VFIO GPU passthrough). That override makes the kernel treat the two B70 as ACS-isolated, and `pci_p2pdma_distance()` would then refuse peer paths between them — *exactly* this signature (host-staged works, peer fails). All earlier probe runs had that override active, so I couldn't tell self-inflicted from genuine.

**So I re-ran the probe on a control boot with `pcie_acs_override` removed.** The result ([`repro/acs_probe_acsOFF_result.log`](repro/acs_probe_acsOFF_result.log)): with the override confirmed absent, the direct peer copy **still returns `0x70000003`**, and host-staged still passes — identical to the override-on baseline. The override was **not** the cause. And the kernel says, in its own words, exactly why:

```
xe 0000:07:00.0: cannot be used for peer-to-peer DMA as the client and provider
(0000:03:00.0) do not share an upstream bridge or whitelisted host bridge
```

So this is a **clean, reproducible refusal at the kernel P2PDMA layer**, not an artifact of my cmdline. The two cards hang off **separate PCIe root ports with no shared switch between them**, so a peer copy has to traverse the root complex / host bridge — and the kernel only permits host-bridge-traversing P2P when that host bridge is on its `pci_p2pdma_whitelist`. This Intel client host bridge isn't on it, so the path is refused and compute-runtime surfaces it as `OUT_OF_DEVICE_MEMORY`. (Consistent corroboration: even with ACS off the two GPUs stay in **separate IOMMU groups**, and neither exposes a `p2pdma` sysfs node — xe never registered as a P2PDMA provider here.)

The honest open question is no longer *"is it my config?"* (it isn't) but *"can this silicon route P2P through the root complex at all?"* — i.e. is this a **whitelist gap that should be fixed** (BMG + this host bridge genuinely support it), or a **consumer-topology limit that should be documented** (it doesn't, and the reliable physical fix is putting both cards under one PCIe switch). Either way the host-staged routes above sidestep it — which is exactly why they're the practical answer today.

This is the honest state: **the workarounds are solid and measured; the root cause is now settled to the kernel P2PDMA layer with the driver's own diagnostic, and the upstream report (now framed as a capability/eligibility question, not a bug claim) is ready to file, not held.**

## The three routes in detail

### Route 1 — llama.cpp layer-split (capacity, best single-stream)

`-sm layer` splits the model's layers across both cards; the only cross-card traffic is one activation hand-off per token, and llama.cpp's `dev2dev_memcpy` (ggml-sycl) already falls back to host-staging automatically when the L0 peer copy errors (force it with `GGML_SYCL_ENABLE_LEVEL_ZERO=0`). A 35B-A3B-Q6_K runs across both B70 at **~50 tok/s single-stream** and answers correctly. This is the route for *one model that doesn't fit on one card* when you care about single-stream latency. It does **not** scale with concurrency (llama.cpp is single-slot here). Launcher: [`scripts/run_llama_tp2.sh`](scripts/).
*A 72B-Q4 GGUF loads and runs (~6 tok/s) but currently produces degenerate output — still debugging; 35B is correct, so it's 72B-specific (likely FA / head-split). Use the AWQ + vLLM path for 70B for now.*

### Route 2 — fusion: host-staged all-reduce in vLLM (this repo's contribution)

The goal: keep vLLM's prefix caching + continuous batching + real tensor parallelism, but stop using oneCCL. vLLM already maintains a **`cpu_group` (gloo)** alongside the GPU `device_group` (xccl, the hang source). The patch ([`fusion_patch/`](fusion_patch/), gated behind `VLLM_XPU_HOSTSTAGED_TP=1` — *off = byte-identical to upstream*, so it's safe to carry) does three things:

1. `parallel_state.py` — force the device group and world init to **gloo** (never create an xccl group).
2. `xpu_communicator.py` — make collectives **GPU → CPU → gloo → GPU** host-staged, upcasting reduces to fp32 (gloo can't reduce fp16/bf16).
3. `xpu_worker.py` — replace the worker-init oneCCL warm-up all-reduce (the literal hang trigger) with a gloo CPU sanity check; add the missing `all_gather` override.

Result (Qwen3.6-27B-AWQ): **zero hangs**, `Application startup complete`, correct output, and N=1→32 scaling **4.4 → 83.7 tok/s (21×)** at only ~70 °C — the host round-trip is per-step, so batching amortizes it. The big property: **it depends on gloo only, so it works regardless of driver/P2P state** — the universal safety net.

### Route 3 — native vLLM TP=2 (fastest aggregate)

vLLM issue [#41663](https://github.com/vllm-project/vllm/issues/41663) (same dual-B70 hardware) posted a working env stack I hadn't tried *as a set*: legacy L0 V1 (`SYCL_UR_USE_LEVEL_ZERO_V2=0`) + oneCCL host collectives (`CCL_ENABLE_SYCL_KERNELS=0`) + `CCL_TOPO_FABRIC_VERTEX_CONNECTION_CHECK=0`. With all three, the ATL handshake clears in ~180 s (delay, not hang) and TP=2 loads on the stock image — no patch. On Qwen3.6-27B-AWQ: **N=64 → 180 tok/s (33×)**, 346 W, 78 °C. Note this "native" success is *also* host-routed reduce (`SYCL_KERNELS=0`) — it's the cousin of the fusion route, not raw P2P. Launcher: [`scripts/run_vllm_tp2_native.sh`](scripts/).

This same native route is what carries a **70B-class** model across the pair — the headline capability, so it gets its own section next.

## What this is actually for: a 70B-class model on two $949 cards

The headline isn't the 27B — it's that a model **too big for any single 32 GB card** runs across the pair at all. A **Qwen2.5-72B-AWQ** loads and serves correctly on both B70 via the native route (route 3), independently measured at **N=1→64 = 4.5 → 119 tok/s aggregate** (the knee is around N=32 ≈ 96 tok/s; it saturates earlier than the 27B because the model is compute-heavier). Idle is ~106 W for the pair; single-stream is a power sink (~440 W for 4.5 tok/s), but batched it lands a **~27× better tokens/joule**. If you want single-stream instead of throughput, llama.cpp layer-split (route 1) runs a 35B across both cards at ~50 tok/s.

**Here's the honest catch — and where I'm still fighting.** This is genuinely a work in progress on the *quality* axis, not a finished product:

- **It's a throughput/capacity play, not a latency one.** A 70B here only pays for itself (perf and power) under batching; single-stream is slow (see the floor discussion above — whether it ever lifts is the open P2P question).
- **Only the AWQ path is solid at 70B** *(same GGUF issue flagged under Route 1 — restated here because it's load-bearing for the 70B picture)*. The 72B-Q4 **GGUF** loads but currently produces **degenerate output** — still debugging (35B-GGUF is correct, so it's 72B-specific, likely FA / head-split). AWQ + vLLM is the working 70B route for now.
- **Capacity is not quality.** In my own side-by-side, this *old* 72B (Qwen2.5) actually **lost** to the *current* 27B (Qwen3.6) on hard synthesis — see **Honest framing** below. So the value isn't "bigger = smarter."

Put together: the real unlock is **being able to run a *current-generation* 70B the day you need one**, on hardware that cost a quarter of the NVIDIA equivalent — and I'm still pushing on the rough edges (single-stream floor, GGUF correctness, picking models that actually earn the capacity). If you're chasing the same thing, the repro is here — tell me what you find.

## Which route for what

- **Low-latency chat / interactive** → don't use TP=2 at all. One model per card; single-stream stays high. (The single-card 27B vLLM-XPU writeup — APC + continuous batching on one B70 — is in [`docs/single-card-27b-vllm-xpu.md`](docs/single-card-27b-vllm-xpu.md).)
- **Offline / batched generation** (bulk drafting, dataset processing, analytics) → **native (route 3)**, 180 tok/s aggregate. Per-item latency doesn't matter, total throughput does.
- **A model too big for one 32 GB card (70B-class)** → **llama.cpp layer-split (route 1)** for single-stream, or **native/fusion vLLM** for batched. This is the capacity unlock.
- **A portable safety net that doesn't care about driver state** → **fusion (route 2)**.

## Why anyone should care: the price/power ladder

To run a 70B-class model *locally* today, grounded prices (2026):

| Option | VRAM | Holds 70B? | ~Price | Power |
|---|---|---|---|---|
| RTX PRO 6000 Blackwell | 96 GB | yes | ~$8,000–9,200 | ~600 W (rated TBP) |
| RTX 5090 ×1 | 32 GB | **no** | ~$2,000 | 575 W (rated TBP) |
| **2× Arc Pro B70** | **64 GB** | **yes (TP=2)** | **~$1,900** ($949 list ×2) | **~440 W** (measured, 72B) |

The NVIDIA wattages are rated board power (spec sheet); the B70 figure is measured wall draw on a 72B run, so they aren't strictly apples-to-apples. All prices are list/MSRP (the B70 pair at $949×2 is reproducible, not a one-off deal). Same capacity at **roughly 1/4 to 1/5 the price** (~$1,900 vs ~$8,000–9,200) **and lower power** — and the pair costs *less than a single RTX 5090* that can't hold a 70B at all. *If* you can live with the floor numbers above and a non-turnkey setup. Which brings us to the honest caveat.

## Honest framing (so this doesn't blow up on you)

- **This is reproducible and documented; it is not turnkey.** Two consumer B70 + magic env flags (route 3) or a 3-file patch (route 2). It works, it's written down, you can redo it — but you will be editing flags, not running an installer.
- **The single-stream floor (4–5 tok/s on vLLM TP=2) is a host-round-trip floor, not a thermal/power ceiling.** The cards sit at 70–78 °C and well under their power budget during these runs — they're idle-waiting on the host-staged reduce, not compute-bound. Whether that floor ever lifts depends on the open question above: if BMG + this host bridge turns out to be a `pci_p2pdma_whitelist` gap, real peer P2P could land in software and these numbers would rise with no change here; if it's a genuine cross-root-port topology limit, the lift needs physical reconfiguration (both cards under one PCIe switch), not a driver update. I'm not promising the floor rises — I'm saying the cards are idle-waiting, so there's headroom *if* the peer path opens.
- **Negative controls are kept on purpose** (HW swap → same hang point; L0 V1 and V2 → same peer error). The point of this repo is to be the map for the next person touching Battlemage multi-GPU, not a benchmark flex.
- **Capacity is real; "bigger model = better answers" is not the claim.** In side-by-side quality testing on this box, a *current-generation* 27B (Qwen3.6) actually beat the *older-generation* 72B (Qwen2.5) on hard synthesis tasks — but that comparison confounds size with a two-generation model gap, so it proves neither "smaller is better" nor "bigger is better." The honest lesson: **the value of fitting a 70B is being able to run a current-generation 70B when you need that capacity** — don't expect an old 72B to out-reason a new 27B. Pick the model by generation first, size second.
- **No security hardening here — this is a hobbyist writeup, not a deployment guide.** The serving endpoints have no auth, no TLS, no network isolation; keep them on `localhost`/a trusted LAN and don't point them at the open internet. The flip side is the reason I'm comfortable publishing it: nothing here touches your system in a risky way. Every route is **config + batched serving**, not kernel patches or driver hacks — the fusion patch is env-gated and byte-identical to upstream when off, and host-staging is just an extra memory round-trip. So treat this as something to **tinker with for the fun of running a 70B at home**, at your own discretion — the workarounds are safe to try, the security posture is on you. And to be plain: this is shared as-is, for free, with **no warranty and no support** — I can't take responsibility for what it does on your hardware. If it eats a weekend, trips your setup, or your model talks nonsense, that's on you, not me. (The LICENSE says the same in legalese.) Have fun with it.

## Repo layout

```
README.md                     ← this file
docs/single-card-27b-vllm-xpu.md   ← single-B70 27B serving (APC + batching), the prod baseline
fusion_patch/                 ← route 2: host-staged all-reduce (VLLM_XPU_HOSTSTAGED_TP=1)
scripts/                      ← launchers for all three routes
repro/b70_p2p_copy_probe.cpp  ← the ~80-line peer-copy probe
repro/acs_probe_acsOFF_result.log  ← control run with pcie_acs_override REMOVED (peer still fails → not the override)
repro/ISSUE-*-draft.md        ← upstream report draft — capability/eligibility question (acs_override control done; ready to file)
patches/                      ← the single-card XPU fixes (mem-shim, AWQ, #41995 for APC)
Dockerfile.prod               ← pinned image
```

## Open questions for anyone with the hardware

- ~~Does the peer copy still return `0x70000003` without `pcie_acs_override`?~~ **Answered: yes, it does** ([`repro/acs_probe_acsOFF_result.log`](repro/acs_probe_acsOFF_result.log)) — the override was not the cause; the kernel refuses the path at the P2PDMA layer (no shared upstream bridge / host bridge not whitelisted).
- Is BMG + a consumer Intel host bridge eligible for the kernel's `pci_p2pdma_whitelist` (i.e. does the silicon route P2P through the root complex), or is cross-root-port P2P genuinely unsupported here? **This is the open question now.**
- Has anyone routed full-bandwidth P2P between two B70 under a single PCIe switch?
- Does xe expose a `p2pdma` sysfs node on any BMG kernel yet? (None here, even on the ACS-off boot — a hint the driver hasn't registered as a P2PDMA provider.)

---

*Built and measured on 2× Arc Pro B70, Ubuntu 26.04 / kernel 7.0.0-22 / xe. If you're chasing multi-GPU on Battlemage — it works, three ways; open an issue or discussion and I'll help.*
