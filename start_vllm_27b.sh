#!/bin/bash
# ExecStart wrapper for the vLLM-XPU 27B server.
# Serves Qwen3.6-27B-AWQ on one Arc Pro B70 (ZE_AFFINITY_MASK=0) at :8001 with APC.
# The prod image bakes in PR #41995 + the mem_get_info shim (see Dockerfile.prod).
set -u
MODEL_DIR="${MODEL_DIR:-$HOME/models}"   # expects $MODEL_DIR/Qwen3.6-27B-AWQ
/usr/bin/docker rm -f vllm-27b 2>/dev/null || true
exec /usr/bin/docker run --name vllm-27b --rm \
  --device /dev/dri \
  -e ZE_AFFINITY_MASK=0 -e ZES_ENABLE_SYSMAN=1 -e VLLM_USE_TRITON_AWQ=1 \
  -e VLLM_WORKER_MULTIPROC_METHOD=spawn \
  -v "${MODEL_DIR}":/models \
  -p 8001:8001 \
  vllm-xpu-27b:prod \
  bash -c 'source /root/.bashrc >/dev/null 2>&1; source /opt/intel/oneapi/setvars.sh --force >/dev/null 2>&1; exec vllm serve /models/Qwen3.6-27B-AWQ --served-model-name qwen/qwen3.6-27b-awq --host 0.0.0.0 --port 8001 -tp 1 --enforce-eager --max-model-len 16384 --gpu-memory-utilization 0.90 --enable-prefix-caching'
