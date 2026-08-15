"""Minimal Classiq -> CUDA-Q -> Metal feasibility test.

Requires Classiq auth first (run once):
    ~/.cudaq-metal/venv/bin/python -c "import classiq; classiq.authenticate()"

Tests whether `qprog_to_cudaq_kernel` (documented Linux-only) works on macOS.
Synthesizes a tiny circuit with Classiq, converts to a cudaq kernel, and runs
it on qpp-cpu and metal.
"""
import math
import cudaq
from classiq import (
    qfunc, CReal, QBit, Output, allocate, RX, synthesize, qprog_to_cudaq_kernel,
)


@qfunc
def main(theta: CReal, q: Output[QBit]):
    allocate(q)
    RX(theta, q)


if __name__ == "__main__":
    print("Synthesizing with Classiq (cloud)...")
    qprog = synthesize(main)
    print("Converting qprog -> cudaq kernel...")
    kernel = qprog_to_cudaq_kernel(qprog)
    print(cudaq.draw(kernel, math.pi / 3))
    for tgt in ["qpp-cpu", "metal"]:
        cudaq.set_target(tgt)
        counts = cudaq.sample(kernel, math.pi / 3, shots_count=4000)
        print(f"[{tgt}] RX(pi/3) ->", {k: counts.count(k) for k in counts})
        cudaq.reset_target()
    print("Classiq -> CUDA-Q -> Metal pipeline WORKS on macOS.")
