# fusion_patch — host-staged all-reduce for vLLM-XPU TP=2 (Route 2)

These four files overlay the corresponding files in an installed vLLM-XPU 0.21. They let
`vllm serve -tp 2` run across two Arc Pro B70 **without oneCCL**, by routing every collective
through host memory over gloo. The whole thing is gated behind one env var:

```
VLLM_XPU_HOSTSTAGED_TP=1
```

**With the flag unset, these files are byte-identical to upstream** — so the overlay is safe to
carry in any image; it only changes behavior when you opt in.

## Why this works

vLLM's TP hang on Battlemage is not really a oneCCL bug — it's that the per-GPU-process model
forces collectives across an L0/xe device boundary, and the direct peer copy fails
(`0x70000003`; see `../repro/`). vLLM already builds a **`cpu_group` (gloo)** alongside the GPU
`device_group` (xccl). gloo runs entirely host-side and never touches the broken peer path, so
we move the reduction there.

## What each file changes (all under `if VLLM_XPU_HOSTSTAGED_TP`)

| File | Change |
|---|---|
| `parallel_state.py` | Force the device group's `new_group` and the world `init_process_group` to **gloo** — never create an xccl group. |
| `xpu_communicator.py` | Make `all_reduce` / `all_gather` / etc. **GPU → CPU → gloo → GPU** host-staged. Reduces are **upcast to fp32** (gloo can't reduce fp16/bf16). |
| `xpu_worker.py` | Replace the worker-init oneCCL warm-up all-reduce (the literal hang trigger) with a gloo CPU sanity check. |
| `base_device_communicator.py` | Reference for the un-overridden `all_gather` that the XPU communicator now provides. |

## Caveats

- The host round-trip is a **per-step** cost, so it hurts single-stream latency but amortizes
  under batching (N=1→32: 4.4 → 83.7 tok/s measured on 27B-AWQ).
- The mount paths in `../scripts/run_vllm_tp2_fusion.sh` assume the dist-packages layout of the
  reference image; adjust `VL=` if your image installs vLLM elsewhere.
- This is a workaround, not an upstream fix. If/when xe ships real BMG peer P2P, the native
  path should outperform this. It's the **portable safety net** — it depends on gloo only.

## License / attribution

All four `.py` files are **vLLM source (Apache-2.0)**, modified here for B70 host-staged TP=2.
Their original SPDX/copyright headers are retained, each carries a top-of-file modification
notice, and the in-file changes are marked `[B70 TP=2 fusion]`. They remain under Apache-2.0 —
see the repository [`NOTICE`](../NOTICE). `parallel_state.py` also keeps vLLM's own upstream
attribution (adapted from NVIDIA/Megatron-LM).
