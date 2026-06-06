#!/bin/bash
# Route 1 — llama.cpp layer-split across two Arc Pro B70 (best single-stream, capacity).
# `-sm layer` splits layers across the cards; the only cross-card traffic is one activation
# hand-off per token. llama.cpp's dev2dev_memcpy auto-falls-back to host-staging when the
# L0 peer copy errors; GGML_SYCL_ENABLE_LEVEL_ZERO=0 forces the host path explicitly.
# Build llama.cpp with the SYCL backend first (see https://github.com/ggml-org/llama.cpp).
#
#   LLAMA_BIN=$HOME/llama.cpp/build-sycl/bin/llama-server \
#   MODEL=$HOME/models/Qwen2.5-35B-A3B-Q6_K.gguf ./run_llama_tp2.sh
set -u
LLAMA_BIN="${LLAMA_BIN:-$HOME/llama.cpp/build-sycl/bin/llama-server}"
MODEL="${MODEL:?set MODEL=/path/to/model.gguf}"
PORT="${PORT:-8003}"

export ONEAPI_DEVICE_SELECTOR="level_zero:*"
export ZES_ENABLE_SYSMAN=1
export GGML_SYCL_ENABLE_LEVEL_ZERO=0      # force host-staged dev2dev (skip the failing L0 peer copy)

exec "$LLAMA_BIN" \
  --model "$MODEL" \
  --device SYCL0,SYCL1 -sm layer \
  -ngl 99 -fa on \
  --host 0.0.0.0 --port "$PORT"
