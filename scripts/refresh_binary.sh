#!/bin/bash
# Maintainer helper: rebuild the C++ backend and copy the fresh dylib + target
# config into the pip package's bundled data, then rebuild the wheel.
# Run after changing anything under src/ or targets/.
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"   # scripts/
ROOT="$(cd "$HERE/.." && pwd)"          # repo root
CQ="${CUDAQ_INSTALL:-$HOME/.cudaq-metal/cudaq}"

echo ">> building backend ..."
bash "$HERE/build_backend.sh" >/dev/null

echo ">> refreshing bundled artifacts ..."
cp "$CQ/lib/libnvqir-metal.dylib" "$ROOT/cudaq_metal/data/"
cp "$CQ/targets/metal.yml" "$ROOT/cudaq_metal/data/"

echo ">> building wheel ..."
python3 -m pip wheel "$ROOT" --no-deps -w "$ROOT/dist" >/dev/null
ls -la "$ROOT/dist"/cudaq_metal-*.whl
echo "done. Commit cudaq_metal/data/ so 'pip install git+<repo>' ships the update."
