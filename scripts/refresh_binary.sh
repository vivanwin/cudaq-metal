#!/bin/bash
# Maintainer helper: rebuild the C++ backend, copy the fresh dylib + target
# config into the pip package's bundled data, then build the wheel into dist/.
# Run after changing anything under src/ or targets/, or when moving to a new
# CUDA-Q version (also bump COMPATIBLE_CUDAQ in cudaq_metal/_install.py and the
# cuda-quantum-cu13 pin in pyproject.toml).
#
# Usage: scripts/refresh_binary.sh [PYTHON]   (env must have cudaq installed)
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PY="${1:-${PYTHON:-python3}}"

echo ">> building backend ..."
bash "$ROOT/scripts/build_backend.sh" "$PY"
VER="$("$PY" -c 'import importlib.metadata as m, re; print(re.match(r"\d+\.\d+\.\d+", m.version("cuda-quantum-cu13")).group(0))')"

echo ">> refreshing bundled artifacts ..."
cp "$ROOT/build/cudaq-$VER/libnvqir-metal.dylib" "$ROOT/cudaq_metal/data/"
cp "$ROOT/targets/metal.yml" "$ROOT/cudaq_metal/data/"

echo ">> building wheel ..."
rm -f "$ROOT"/dist/cudaq_metal-*.whl
# setuptools reuses build/lib between builds; clear it so files deleted from the
# package (or stray local files) can't leak into the wheel.
rm -rf "$ROOT/build/lib" "$ROOT"/build/bdist.* "$ROOT"/*.egg-info
"$PY" -m pip wheel "$ROOT" --no-deps -w "$ROOT/dist" >/dev/null
ls -la "$ROOT"/dist/cudaq_metal-*.whl
echo "done. Commit cudaq_metal/data/ so 'pip install git+<repo>' ships the update."
