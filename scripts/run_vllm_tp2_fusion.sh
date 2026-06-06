#!/bin/bash
# Route 2 — fusion: host-staged all-reduce in vLLM (this repo's contribution).
# Drops oneCCL entirely and routes collectives through host gloo. Gated behind
# VLLM_XPU_HOSTSTAGED_TP=1; with the flag OFF the mounted files are byte-identical to
# upstream, so carrying the mount is safe. Depends on gloo only => works regardless of
# driver / P2P state. The four files in ../fusion_patch/ overlay the installed vLLM.
#
#   IMAGE=vllm-xpu:0.21 MODEL_DIR=$HOME/models MODEL=Qwen2.5-72B-Instruct-AWQ \
#     ./run_vllm_tp2_fusion.sh
set -u
IMAGE="${IMAGE:-vllm-xpu:0.21}"
MODEL_DIR="${MODEL_DIR:-$HOME/models}"
MODEL="${MODEL:-Qwen2.5-72B-Instruct-AWQ}"
PORT="${PORT:-8003}"
SMN="${SMN:-qwen/$(echo "$MODEL" | tr 'A-Z' 'a-z')}"
MAXLEN="${MAXLEN:-8192}"
PATCH="$(cd "$(dirname "$0")/../fusion_patch" && pwd)"
# Where vLLM's distributed code lives inside the image (adjust if your image differs):
VL=/usr/local/lib/python3.12/dist-packages/vllm/distributed

exec docker run --name vllm-tp2-fusion --rm --device /dev/dri \
  -e ZE_AFFINITY_MASK=0,1 \
  -e ZES_ENABLE_SYSMAN=1 \
  -e VLLM_USE_TRITON_AWQ=1 \
  -e VLLM_WORKER_MULTIPROC_METHOD=spawn \
  -e VLLM_XPU_HOSTSTAGED_TP=1 \
  -v "${MODEL_DIR}":/models \
  -v "${PATCH}/parallel_state.py":${VL}/parallel_state.py:ro \
  -v "${PATCH}/xpu_communicator.py":${VL}/device_communicators/xpu_communicator.py:ro \
  -v "${PATCH}/base_device_communicator.py":${VL}/device_communicators/base_device_communicator.py:ro \
  -v "${PATCH}/xpu_worker.py":/usr/local/lib/python3.12/dist-packages/vllm/v1/worker/xpu_worker.py:ro \
  -p "$PORT":"$PORT" "$IMAGE" \
  bash -c "source /root/.bashrc >/dev/null 2>&1; source /opt/intel/oneapi/setvars.sh --force >/dev/null 2>&1; \
    exec vllm serve /models/${MODEL} --served-model-name ${SMN} --host 0.0.0.0 --port ${PORT} \
      -tp 2 --enforce-eager --max-model-len ${MAXLEN} --gpu-memory-utilization 0.90"
