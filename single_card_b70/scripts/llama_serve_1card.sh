#!/usr/bin/env bash
# Serve a model on ONE Intel Arc GPU with the SYCL build of llama.cpp.
# This is rung 1 (single-stream latency). On Arc Pro B70 + Qwen3.6-35B-A3B Q6_K
# this lands ~72 tok/s single-stream (vs ~44 tok/s on the OpenCL backend).
set -euo pipefail

ONEAPI_ENV="${ONEAPI_ENV:-/opt/intel/oneapi/setvars.sh}"
BIN="${BIN:-./build-sycl/bin}"          # output of llama_sycl_build.sh
MODEL="${MODEL:?set MODEL=/path/to/model.gguf}"
PORT="${PORT:-8000}"
DEVICE="${DEVICE:-level_zero:0}"          # pick the GPU; level_zero:0 = first Arc

# shellcheck disable=SC1090
source "$ONEAPI_ENV" >/dev/null 2>&1 || true

export ONEAPI_DEVICE_SELECTOR="$DEVICE"
export ZES_ENABLE_SYSMAN=1               # lets llama report VRAM use

# -ngl 99: all layers on GPU.  -fa on: flash attention.
exec "$BIN/llama-server" -m "$MODEL" -ngl 99 -fa on --host 0.0.0.0 --port "$PORT"
