#!/bin/bash
# Copy the freshly built Metal backend into the Python cudaq (venv) so that
# `cudaq.set_target("metal")` picks it up. Run scripts/build_backend.sh first.
set -euo pipefail
CQ="${CUDAQ_INSTALL:-$HOME/.cudaq-metal/cudaq}"
SP="$HOME/.cudaq-metal/venv/lib/python3.12/site-packages"
cp "$CQ/lib/libnvqir-metal.dylib" "$SP/lib/"
cp "$CQ/targets/metal.yml" "$SP/targets/"
echo "installed metal backend into venv: $SP/{lib,targets}"
