# XPU patches

Small diffs against vLLM 0.21.0 (`+xpu`), applied to the installed package
(`$SITE_PACKAGES/vllm/...`). Apply with e.g.:

    patch -p1 -d /opt/venv/lib/python3.12/site-packages < mamba_utils-pr41995.patch

| patch | file | what / why |
|---|---|---|
| `mamba_utils-pr41995.patch` | `vllm/v1/worker/mamba_utils.py` | **Upstream [PR #41995](https://github.com/vllm-project/vllm/pull/41995).** View mamba copy-metadata device pointers as `uint64` so XPU pointers ≥ 2⁶³ don't overflow. **Enables APC** (needed for the 27B). If your base vLLM already includes #41995, skip it. |
| `moe_wna16-xpu-capability-gate.patch` | `.../quantization/moe_wna16.py` | (35B only) XPU `get_device_capability()` is `None`; the AWQ min-capability gate is CUDA SM semantics, N/A on XPU — treat as capable. |
| `awq-xpu-force-wna16.patch` | `.../quantization/awq.py` | (35B only) In the FusedMoE branch, Marlin is CUDA-only; on XPU always take the MoeWNA16 (triton) path. |

The `mamba_utils` change is upstream vLLM (Apache-2.0; see ../NOTICE). The other two
are my own modifications of vLLM source files, likewise under Apache-2.0. They are
**experimental** — triton wNa16 MoE on XPU is unverified (see ../README.md).
