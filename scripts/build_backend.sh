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

# Locate cudaq without importing it; print version and lib dir on two lines
# (two reads, so paths containing spaces survive).
{ read -r CUDAQ_VERSION; read -r CUDAQ_LIB_DIR; } < <("$PY" - <<'EOF'
import importlib.metadata as md, importlib.util, os, re
spec = importlib.util.find_spec("cudaq")
if spec is None or not spec.origin:
    raise SystemExit("cudaq is not installed for this Python")
ver = md.version("cuda-quantum-cu13")
print(re.match(r"\d+\.\d+\.\d+", ver).group(0))  # source tag, e.g. 0.16.0
print(os.path.join(os.path.dirname(os.path.dirname(spec.origin)), "lib"))
EOF
)
echo ">> cudaq $CUDAQ_VERSION ($CUDAQ_LIB_DIR)"

HEADERS="$ROOT/toolchain/cudaq-headers-$CUDAQ_VERSION"
"$PY" "$ROOT/scripts/fetch_cudaq_headers.py" "$CUDAQ_VERSION" "$HEADERS"

BUILD="$ROOT/build/cudaq-$CUDAQ_VERSION"
GEN=()
command -v ninja >/dev/null && GEN=(-G Ninja)
cmake -S "$ROOT" -B "$BUILD" "${GEN[@]}" -DCMAKE_BUILD_TYPE=Release \
  -DCUDAQ_HEADERS_DIR="$HEADERS" -DCUDAQ_LIB_DIR="$CUDAQ_LIB_DIR" >/dev/null
cmake --build "$BUILD"
echo ">> built $BUILD/libnvqir-metal.dylib"
