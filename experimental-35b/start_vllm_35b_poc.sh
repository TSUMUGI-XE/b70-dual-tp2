#!/bin/bash
# [EXPERIMENTAL / not production] Launch 35B-A3B-AWQ (MoE) on vLLM-XPU to test the capability-gate patches.
# Prereq (manual): free a card. The prod 27B (:8001) is on ZE_AFFINITY_MASK=0; this PoC uses card 1 (:8003).
# Image: vllm-xpu-35b:prod (PR #41995 + mem shim + moe_wna16/awq capability patches baked in).
# Goal: (1) get past the capability gate and load? (2) MoE forward runs on XPU?
#       (3) does MoE + continuous batching hold up? (llama.cpp's MoE parallel decode degraded for us)
# Caveat: VRAM tight (AWQ ~25.5GB) -> modest max-model-len. UNVERIFIED (triton wna16 MoE on XPU not guaranteed).
set -u
MODEL_DIR="${MODEL_DIR:-$HOME/models}"   # expects $MODEL_DIR/Qwen3.6-35B-A3B-AWQ
/usr/bin/docker rm -f vllm-35b-poc 2>/dev/null || true
exec /usr/bin/docker run --name vllm-35b-poc --rm \
  --device /dev/dri \
  -e ZE_AFFINITY_MASK=1 -e ZES_ENABLE_SYSMAN=1 -e VLLM_USE_TRITON_AWQ=1 \
  -e VLLM_WORKER_MULTIPROC_METHOD=spawn \
  -v "${MODEL_DIR}":/models \
  -p 8003:8003 \
  vllm-xpu-35b:prod \
  bash -c 'source /root/.bashrc >/dev/null 2>&1; source /opt/intel/oneapi/setvars.sh --force >/dev/null 2>&1; exec vllm serve /models/Qwen3.6-35B-A3B-AWQ --served-model-name qwen/qwen3.6-35b-a3b-awq --host 0.0.0.0 --port 8003 -tp 1 --enforce-eager --max-model-len 8192 --gpu-memory-utilization 0.95 --enable-prefix-caching'
