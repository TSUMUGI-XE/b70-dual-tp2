# Experimental: 35B-A3B (MoE) AWQ on vLLM-XPU — currently blocked

This does **not** serve in production. It's the in-progress attempt to run
`Qwen3.6-35B-A3B-AWQ` (MoE) on a single Arc Pro B70 under vLLM-XPU.

The blocker is a **capability gate, not OOM**: `XPUPlatform.get_device_capability()`
returns `None`, so `moe_wna16` computes `device_capability = -1 < AWQConfig.get_min_capability()`
and refuses the quantized MoE method. The two patches in `../patches/` relax that gate and
force the triton MoeWNA16 path. **Even with the gate open, the triton wNa16 MoE kernel
correctness/perf on XPU is unverified.** The upstream item to watch is
[RFC #37979](https://github.com/vllm-project/vllm/issues/37979) (MoE wNa16 on XPU = "Planned").

Files:
- `Dockerfile.prod-35b` — bakes the two patches on top of `vllm-xpu-27b:prod`.
- `start_vllm_35b_poc.sh` — launch on card 1 (`ZE_AFFINITY_MASK=1`) at `:8003`.
