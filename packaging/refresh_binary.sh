#!/bin/bash
# Maintainer helper: rebuild the C++ backend and copy the fresh dylib + target
# config into the pip package's bundled data, then (optionally) rebuild the wheel.
# Run this after changing anything under ../src or ../targets.
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/.." && pwd)"
CQ="${CUDAQ_INSTALL:-$HOME/.cudaq-metal/cudaq}"

echo ">> building backend ..."
bash "$ROOT/scripts/build_backend.sh" >/dev/null

echo ">> refreshing bundled artifacts ..."
cp "$CQ/lib/libnvqir-metal.dylib" "$HERE/src/cudaq_metal/data/"
cp "$CQ/targets/metal.yml" "$HERE/src/cudaq_metal/data/"

echo ">> building wheel ..."
python3 -m pip wheel "$HERE" --no-deps -w "$HERE/dist" >/dev/null
ls -la "$HERE/dist"/cudaq_metal-*.whl
echo "done. Install elsewhere with: pip install $HERE/dist/cudaq_metal-*.whl"
