#!/usr/bin/env bash
# Build llama.cpp with the SYCL backend for Intel Arc (the "free 1.6x" over OpenCL).
# Tested on Arc Pro B70, oneAPI 2026.0, llama.cpp build ~b9738.
#
# Prereq: Intel oneAPI Base Toolkit (provides icx/icpx + the SYCL runtime).
#   https://www.intel.com/content/www/us/en/developer/tools/oneapi/base-toolkit.html
set -euo pipefail

ONEAPI_ENV="${ONEAPI_ENV:-/opt/intel/oneapi/setvars.sh}"
SRC="${1:-$PWD}"   # path to a llama.cpp checkout (default: current dir)

# shellcheck disable=SC1090
source "$ONEAPI_ENV"

cmake -S "$SRC" -B "$SRC/build-sycl" \
  -DGGML_SYCL=ON \
  -DCMAKE_C_COMPILER=icx \
  -DCMAKE_CXX_COMPILER=icpx \
  -DCMAKE_BUILD_TYPE=Release

cmake --build "$SRC/build-sycl" --config Release -j"$(nproc)"

echo
echo "Done. Binaries in: $SRC/build-sycl/bin/"
echo "Sanity check the device is the SYCL/Level-Zero path (not OpenCL):"
echo "  ONEAPI_DEVICE_SELECTOR=level_zero:0 $SRC/build-sycl/bin/llama-bench -m <model>.gguf -ngl 99 --list-devices"
