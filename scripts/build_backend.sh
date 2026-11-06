#!/bin/bash
# Build libnvqir-metal.dylib against a pip-installed cudaq (Apple Silicon only).
#
# The cudaq macOS wheel (cuda-quantum-cu13) provides the runtime libraries; the
# matching C++ headers are fetched from the cuda-quantum source tag, since the
# wheel stopped shipping them in 0.16. No CUDA-Q toolkit installer needed.
#
# Usage: scripts/build_backend.sh [PYTHON]
#   PYTHON  interpreter whose environment has cudaq installed (default: python3)
# Output: build/cudaq-<version>/libnvqir-metal.dylib
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PY="${1:-${PYTHON:-python3}}"
source "$ROOT/scripts/_cudaq_env.sh"
cudaq_env "$PY"
echo ">> cudaq $CUDAQ_VERSION ($CUDAQ_LIB_DIR)"

HEADERS="$ROOT/toolchain/cudaq-headers-$CUDAQ_VERSION"
"$PY" "$ROOT/scripts/fetch_cudaq_headers.py" "$CUDAQ_VERSION" "$HEADERS"

# Prefer Ninja when available (CMake reads CMAKE_GENERATOR from the env).
if [ -z "${CMAKE_GENERATOR:-}" ] && command -v ninja >/dev/null; then
  export CMAKE_GENERATOR=Ninja
fi
BUILD="$ROOT/build/cudaq-$CUDAQ_VERSION"
cmake -S "$ROOT" -B "$BUILD" -DCMAKE_BUILD_TYPE=Release \
  -DCUDAQ_HEADERS_DIR="$HEADERS" -DCUDAQ_LIB_DIR="$CUDAQ_LIB_DIR" >/dev/null
cmake --build "$BUILD"
echo ">> built $BUILD/libnvqir-metal.dylib"
