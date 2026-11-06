#!/bin/bash
# Maintainer helper: rebuild the C++ backend, copy the fresh dylib into the pip
# package's bundled data, then build the wheel into dist/. Run after changing
# anything under src/, or when moving to a new CUDA-Q version (also bump
# COMPATIBLE_CUDAQ in cudaq_metal/_install.py and the cuda-quantum-cu13 pin in
# pyproject.toml).
#
# With an editable install (`pip install -e .`), the next `import cudaq_metal`
# copies the refreshed dylib into cudaq automatically.
#
# Usage: scripts/refresh_binary.sh [PYTHON]   (env must have cudaq installed)
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PY="${1:-${PYTHON:-python3}}"
source "$ROOT/scripts/_cudaq_env.sh"
cudaq_env "$PY"

echo ">> building backend ..."
bash "$ROOT/scripts/build_backend.sh" "$PY"

echo ">> refreshing bundled artifacts ..."
cp "$ROOT/build/cudaq-$CUDAQ_VERSION/libnvqir-metal.dylib" "$ROOT/cudaq_metal/data/"

echo ">> building wheel ..."
rm -f "$ROOT"/dist/cudaq_metal-*.whl
# setuptools reuses build/lib between builds; clear it so files deleted from the
# package (or stray local files) can't leak into the wheel.
rm -rf "$ROOT/build/lib" "$ROOT"/build/bdist.* "$ROOT"/*.egg-info
"$PY" -m pip wheel "$ROOT" --no-deps -w "$ROOT/dist" >/dev/null
ls -la "$ROOT"/dist/cudaq_metal-*.whl
echo "done. Commit cudaq_metal/data/ so 'pip install git+<repo>' ships the update."
