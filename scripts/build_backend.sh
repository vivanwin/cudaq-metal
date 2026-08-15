#!/bin/bash
# Build and install the Metal NVQIR backend into the CUDA-Q tree.
#
# Runs through a space-free symlink to the project because the CUDA-Q toolchain
# (nvq++, and some CMake/Ninja rules) mishandle spaces in paths.
set -euo pipefail

CQ="${CUDAQ_INSTALL:-$HOME/.cudaq-metal/cudaq}"
PROJ_REAL="/Users/vincent/Documents/projects.nosync/Demo v2/cuda-q metal"
PROJ="$HOME/.cudaq-metal/proj"           # space-free symlink -> PROJ_REAL
ln -sfn "$PROJ_REAL" "$PROJ"

source "$CQ/set_env.sh"
export CUDAQ_SYSROOT="$(xcrun --show-sdk-path)"

cd "$PROJ"
cmake -B build -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DNVQIR_DIR="$CQ/lib/cmake/nvqir"
cmake --build build
cmake --install build

echo "=== installed backend artifacts ==="
ls -la "$CQ/lib/libnvqir-metal.dylib" "$CQ/targets/metal.yml"
