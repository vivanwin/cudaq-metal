"""Scaling benchmark: the same Classiq-synthesized circuit on metal (GPU) vs
qpp-cpu (CPU), via qprog_to_cudaq_kernel.

The circuit is examples/brickwork_classiq.py's hardware-efficient ansatz (RY
layer + CX-RZ-CX brickwork), observed against Σ Z_i. Only `cudaq.observe(...)`
is timed; synthesis, kernel construction and a warmup run happen first.
qpp-cpu stops at n=17 (~2 s per observe and doubling); metal runs up to n=30
to show the curve up to the statevector ceiling (sizes that don't fit are
reported and skipped). Results go to results/examples.csv, replaced only once
the whole sweep has finished. Requires ``classiq.authenticate()`` once.
"""
import functools
import operator
import os
import time
from functools import reduce

import numpy as np

import cudaq_metal  # noqa: F401  (registers the target; must precede cudaq)
import cudaq
from classiq import (qfunc, QArray, QBit, Output, allocate,
                     RY, RZ, CX, synthesize, qprog_to_cudaq_kernel,
                     pauli_operator_to_cudaq_spin_op, Pauli)

DEPTH, TRIALS = 5, 3
THETA, PHI = 0.5, 0.7
SIZES = {"qpp-cpu": range(8, 18), "metal": range(8, 31)}
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", "examples.csv")


@functools.cache  # synthesize once per size, reuse for both targets
def _kernel_and_hamiltonian(N):
    @qfunc
    def bw(q: QArray[QBit, N]):
        for layer in range(DEPTH):
            for i in range(N):
                RY(THETA, q[i])
            for j in range(layer % 2, N - 1, 2):
                CX(q[j], q[j + 1])
                RZ(PHI, q[j + 1])
                CX(q[j], q[j + 1])

    @qfunc
    def main(q: Output[QArray[QBit, N]]):
        allocate(q)
        bw(q)

    kernel = qprog_to_cudaq_kernel(synthesize(main))
    H = pauli_operator_to_cudaq_spin_op(reduce(operator.add, (Pauli.Z(i) for i in range(N))))
    return kernel, H


def observe_times(N, target):
    kernel, H = _kernel_and_hamiltonian(N)
    cudaq.set_target(target)
    cudaq.observe(kernel, H).expectation()  # warmup
    ts = []
    for _ in range(TRIALS):
        t0 = time.perf_counter()
        cudaq.observe(kernel, H).expectation()
        ts.append(time.perf_counter() - t0)
    return ts


def main():
    rows = ["n,depth,target,mean,std"]
    for target, sizes in SIZES.items():
        for n in sizes:
            try:
                ts = observe_times(n, target)
            except Exception as e:
                print(f"n={n} {target}: FAILED ({e})", flush=True)
                continue
            m, s = float(np.mean(ts)), float(np.std(ts))
            rows.append(f"{n},{DEPTH},{target},{m:.4f},{s:.4f}")
            print(f"n={n:2d} {target:8s}: {m:8.4f}s ± {s:.4f}", flush=True)
    tmp = OUT + ".part"
    with open(tmp, "w") as f:
        f.write("\n".join(rows) + "\n")
    os.replace(tmp, OUT)
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
