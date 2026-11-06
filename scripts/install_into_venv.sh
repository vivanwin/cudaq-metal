#!/bin/bash
# Dev helper: copy a freshly built backend straight into a Python env's cudaq so
# `cudaq.set_target("metal")` picks it up, without reinstalling the package.
# Run scripts/build_backend.sh first.
#
# Usage: scripts/install_into_venv.sh [PYTHON]   (default: python3)
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PY="${1:-${PYTHON:-python3}}"
{ read -r VER; read -r LIB; } < <("$PY" - <<'EOF'
import importlib.metadata as md, importlib.util, os, re
spec = importlib.util.find_spec("cudaq")
print(re.match(r"\d+\.\d+\.\d+", md.version("cuda-quantum-cu13")).group(0))
print(os.path.join(os.path.dirname(os.path.dirname(spec.origin)), "lib"))
EOF
)
TARGETS="$(dirname "$LIB")/targets"
mkdir -p "$TARGETS"
cp "$ROOT/build/cudaq-$VER/libnvqir-metal.dylib" "$LIB/"
cp "$ROOT/targets/metal.yml" "$TARGETS/"
echo "installed metal backend (cudaq $VER) into $LIB and $TARGETS"
