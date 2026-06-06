#!/bin/bash
# Route 3 — native vLLM TP=2 across two Arc Pro B70 (no patch).
# Uses the env stack reported in vLLM issue #41663: legacy Level-Zero V1 + oneCCL host
# collectives + topology check off. The ATL handshake clears in ~180s (delay, not hang).
# This is host-routed reduce (CCL_ENABLE_SYCL_KERNELS=0), NOT raw peer P2P.
#
#   IMAGE=vllm-xpu:0.21 MODEL_DIR=$HOME/models MODEL=Qwen2.5-72B-Instruct-AWQ \
#     ./run_vllm_tp2_native.sh
set -u
IMAGE="${IMAGE:-vllm-xpu:0.21}"          # a vLLM-XPU 0.21 image with the single-card patches baked (see patches/)
MODEL_DIR="${MODEL_DIR:-$HOME/models}"
MODEL="${MODEL:-Qwen2.5-72B-Instruct-AWQ}"
PORT="${PORT:-8003}"
SMN="${SMN:-qwen/$(echo "$MODEL" | tr 'A-Z' 'a-z')}"
MAXLEN="${MAXLEN:-8192}"

exec docker run --name vllm-tp2-native --rm --device /dev/dri \
  -e ZE_AFFINITY_MASK=0,1 \
  -e ZES_ENABLE_SYSMAN=1 \
  -e VLLM_USE_TRITON_AWQ=1 \
  -e VLLM_WORKER_MULTIPROC_METHOD=spawn \
  -e SYCL_UR_USE_LEVEL_ZERO_V2=0 \
  -e CCL_ENABLE_SYCL_KERNELS=0 \
  -e CCL_TOPO_FABRIC_VERTEX_CONNECTION_CHECK=0 \
  -v "${MODEL_DIR}":/models -p "$PORT":"$PORT" "$IMAGE" \
  bash -c "source /root/.bashrc >/dev/null 2>&1; source /opt/intel/oneapi/setvars.sh --force >/dev/null 2>&1; \
    exec vllm serve /models/${MODEL} --served-model-name ${SMN} --host 0.0.0.0 --port ${PORT} \
      -tp 2 --enforce-eager --max-model-len ${MAXLEN} --gpu-memory-utilization 0.90"
