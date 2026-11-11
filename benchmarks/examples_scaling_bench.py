"""Scaling benchmark: the same Classiq-synthesized circuit on metal (GPU) vs
qpp-cpu (CPU), via qprog_to_cudaq_kernel.

The circuit is examples/brickwork_classiq.py's hardware-efficient ansatz (RY
layer + CX-RZ-CX brickwork), observed against Σ Z_i. Only `cudaq.observe(...)`
is timed; synthesis, kernel construction and a warmup run happen first.
qpp-cpu stops at n=17 (~2 s per observe and doubling); metal runs up to n=30
to show the curve up to the statevector ceiling (sizes that don't fit are
reported and skipped). Results go to results/examples.csv, replaced only once
the whole sweep has finished; pass target names (e.g. ``metal``) to re-measure
only those and keep the other rows. Requires ``classiq.authenticate()`` once.
"""
import functools
import operator
import os
import sys
import time
from functools import reduce

import numpy as np

import cudaq_metal  # noqa: F401  (registers the target; must precede cudaq)
import cudaq
from classiq import (qfunc, QArray, QBit, Output, allocate,
                     RY, RZ, CX, synthesize, qprog_to_cudaq_kernel,
                     pauli_operator_to_cudaq_spin_op, Pauli)

DEPTH = 5
# Report the median of at least MIN_TRIALS observes, repeating fast points
# until they add up to MIN_SECONDS: a mean of a few millisecond-scale calls is
# dominated by scheduling noise. MAX_TRIALS only stops sub-0.25 ms points.
MIN_TRIALS, MIN_SECONDS, MAX_TRIALS = 5, 0.5, 2000
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
    while len(ts) < MIN_TRIALS or (sum(ts) < MIN_SECONDS and len(ts) < MAX_TRIALS):
        t0 = time.perf_counter()
        cudaq.observe(kernel, H).expectation()
        ts.append(time.perf_counter() - t0)
    return ts


def main():
    # `python examples_scaling_bench.py metal` re-measures only metal and keeps
    # the other targets' rows from the existing CSV.
    targets = sys.argv[1:] or list(SIZES)
    unknown = set(targets) - set(SIZES)
    if unknown:
        sys.exit(f"unknown target(s) {sorted(unknown)}; choose from {list(SIZES)}")
    rows = ["n,depth,target,median,std"]
    if os.path.exists(OUT):
        with open(OUT) as f:
            rows += [r for r in f.read().splitlines()[1:]
                     if r and r.split(",")[2] not in targets]
    for target in targets:
        for n in SIZES[target]:
            try:
                ts = observe_times(n, target)
            except Exception as e:
                print(f"n={n} {target}: FAILED ({e})", flush=True)
                continue
            m, s = float(np.median(ts)), float(np.std(ts))
            # Microsecond resolution: metal's small sizes take ~0.5-1 ms.
            rows.append(f"{n},{DEPTH},{target},{m:.6f},{s:.6f}")
            print(f"n={n:2d} {target:8s}: {m:8.4f}s ± {s:.4f}", flush=True)
    tmp = OUT + ".part"
    with open(tmp, "w") as f:
        f.write("\n".join(rows) + "\n")
    os.replace(tmp, OUT)
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
