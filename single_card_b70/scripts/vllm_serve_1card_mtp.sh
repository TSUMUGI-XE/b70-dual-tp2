#!/usr/bin/env bash
# Rung 1 "squeeze every last token" tier: vLLM-XPU on ONE Arc GPU with
# PIECEWISE cudagraph + MTP (speculative decode) + prefix caching.
# On Arc Pro B70 + Qwen3.6-35B-A3B (MXFP4/ssm16) this lands ~123 tok/s single-stream.
#
# NOTE: this is the advanced tier and is NOT plug-and-play. It needs a custom
# vLLM-XPU image (oneAPI 2026 + vLLM 0.23 + the patches listed in the README:
# XPU cudagraph enablers, cherry-picks #45614 / #44986, the mamba_utils signed-wrap
# fix) and a self-converted MXFP4 model with the SSM gate left at BF16. The flags
# below are the runnable core; build the image separately. See README "Rung 1
# (advanced)" for the patch stack.
set -euo pipefail

IMG="${VLLM_XPU_IMAGE:?set VLLM_XPU_IMAGE=<your custom vllm-xpu image>}"
MODEL="${MODEL:?set MODEL=/models/<your-mxfp4-ssm16-model>}"
MODELS_DIR="${MODELS_DIR:-$HOME/models}"
SERVED_NAME="${SERVED_NAME:-qwen3.6-35b-a3b}"
PORT="${PORT:-8000}"
CARD="${CARD:-0}"          # ZE_AFFINITY_MASK: which Arc GPU (0 = first)
NAME="${NAME:-vllm-1card-mtp}"

docker run -d --name "$NAME" \
  --device /dev/dri -v /dev/dri/by-path:/dev/dri/by-path:ro \
  --cap-add SYS_PTRACE --security-opt seccomp=unconfined --ipc=host --shm-size=16g \
  -e ZE_AFFINITY_MASK="$CARD" \
  -e ZES_ENABLE_SYSMAN=1 \
  -e VLLM_WORKER_MULTIPROC_METHOD=spawn \
  -e VLLM_XPU_ENABLE_XPU_GRAPH=1 \
  -v "$MODELS_DIR":/models \
  -p "$PORT":"$PORT" \
  "$IMG" vllm serve "$MODEL" \
    --served-model-name "$SERVED_NAME" \
    --host 0.0.0.0 --port "$PORT" \
    -tp 1 --language-model-only \
    --max-model-len 8192 --gpu-memory-utilization 0.90 \
    --enable-prefix-caching \
    --compilation-config '{"cudagraph_mode":"PIECEWISE"}' \
    --speculative-config '{"method":"mtp","num_speculative_tokens":3}'

echo "Serving on :$PORT (model $SERVED_NAME on Arc GPU $CARD)."
echo "Wait for health:  curl -s localhost:$PORT/health"
echo "Logs:             docker logs -f $NAME"
