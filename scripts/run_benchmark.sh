#!/bin/bash
# Compile the CUDA-Q GHZ benchmark for both qpp-cpu and metal, then run both.
# Usage: run_benchmark.sh [minQ] [maxQ] [shots] [repeats]
set -euo pipefail

CQ="${CUDAQ_INSTALL:-$HOME/.cudaq-metal/cudaq}"
PROJ_REAL="/Users/vincent/Documents/projects.nosync/Demo v2/cuda-q metal"
PROJ="$HOME/.cudaq-metal/proj"
ln -sfn "$PROJ_REAL" "$PROJ"

source "$CQ/set_env.sh"
export CUDAQ_SYSROOT="$(xcrun --show-sdk-path)"
cd "$PROJ"

MINQ="${1:-10}"; MAXQ="${2:-20}"; SHOTS="${3:-1000}"; REPS="${4:-5}"

echo ">> compiling benchmark for qpp-cpu and metal ..."
nvq++ --target qpp-cpu -O2 benchmark/bench_cudaq.cpp -o build_bench_qpp
nvq++ --target metal   -O2 benchmark/bench_cudaq.cpp -o build_bench_metal

echo; echo "############ qpp-cpu (CUDA-Q native CPU baseline) ############"
./build_bench_qpp   "$MINQ" "$MAXQ" "$SHOTS" "$REPS"
echo; echo "############ metal (Apple GPU) ############"
./build_bench_metal "$MINQ" "$MAXQ" "$SHOTS" "$REPS"
