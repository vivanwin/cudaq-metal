"""Scaling benchmark: the same Classiq-synthesized circuit on metal (GPU) vs
qpp-cpu (CPU), via qprog_to_cudaq_kernel.

The circuit is examples/brickwork_classiq.py's hardware-efficient ansatz (RY
layer + CX-RZ-CX brickwork), observed against Σ Z_i, at 5 layers (~12 gates
per qubit) and 50 layers (~120 gates per qubit). Only `cudaq.observe(...)` is
timed; synthesis, kernel construction and a warmup run happen first.
qpp-cpu stops at n=17 (~2 s per observe at 5 layers, ~20 s at 50, doubling
per qubit); metal runs up to n=30 to show the curve up to the statevector
ceiling (sizes that don't fit are reported and skipped; 50 layers at n=30 take
~2.5 minutes per observe). Results go to results/examples.csv, replaced only
once the whole sweep has finished. Pass target names (e.g. ``metal``) and/or
``--depth D`` to re-measure only those and keep the other rows. Requires
``classiq.authenticate()`` once.
"""
import argparse
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

DEPTHS = (5, 50)
# Report the median of at least MIN_TRIALS observes, repeating fast points
# until they add up to MIN_SECONDS: a mean of a few millisecond-scale calls is
# dominated by scheduling noise. MAX_TRIALS only stops sub-0.25 ms points.
MIN_TRIALS, MIN_SECONDS, MAX_TRIALS = 5, 0.5, 2000
SYNTH_ATTEMPTS = 3
THETA, PHI = 0.5, 0.7
SIZES = {"qpp-cpu": range(8, 18), "metal": range(8, 31)}
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", "examples.csv")


@functools.cache  # synthesize once per (size, depth), reuse for both targets
def _kernel_and_hamiltonian(N, depth):
    @qfunc
    def bw(q: QArray[QBit, N]):
        for layer in range(depth):
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

    for attempt in range(1, SYNTH_ATTEMPTS + 1):
        try:
            qprog = synthesize(main)
            break
        except Exception as e:  # cloud call: timeouts are transient
            if attempt == SYNTH_ATTEMPTS:
                raise
            print(f"n={N} depth={depth}: synthesis failed ({type(e).__name__}), "
                  f"retry {attempt}/{SYNTH_ATTEMPTS - 1}", flush=True)
    kernel = qprog_to_cudaq_kernel(qprog)
    H = pauli_operator_to_cudaq_spin_op(reduce(operator.add, (Pauli.Z(i) for i in range(N))))
    return kernel, H


def observe_times(N, depth, target):
    kernel, H = _kernel_and_hamiltonian(N, depth)
    cudaq.set_target(target)
    cudaq.observe(kernel, H).expectation()  # warmup
    ts = []
    while len(ts) < MIN_TRIALS or (sum(ts) < MIN_SECONDS and len(ts) < MAX_TRIALS):
        t0 = time.perf_counter()
        cudaq.observe(kernel, H).expectation()
        ts.append(time.perf_counter() - t0)
    return ts


def main():
    # `python examples_scaling_bench.py metal --depth 50` re-measures only that
    # (target, depth) pair and keeps every other row from the existing CSV.
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("targets", nargs="*", help=f"default: all of {list(SIZES)}")
    ap.add_argument("--depth", type=int, action="append",
                    help=f"layers to run (repeatable); default: {list(DEPTHS)}")
    args = ap.parse_args()
    targets, depths = args.targets or list(SIZES), args.depth or list(DEPTHS)
    unknown = set(targets) - set(SIZES)
    if unknown:
        ap.error(f"unknown target(s) {sorted(unknown)}; choose from {list(SIZES)}")
    redo = {(t, str(d)) for t in targets for d in depths}
    rows = ["n,depth,target,median,std"]
    if os.path.exists(OUT):
        with open(OUT) as f:
            for r in f.read().splitlines()[1:]:
                if r and (r.split(",")[2], r.split(",")[1]) not in redo:
                    rows.append(r)
    for target in targets:
        for depth in depths:
            for n in SIZES[target]:
                try:
                    ts = observe_times(n, depth, target)
                except Exception as e:
                    print(f"n={n} depth={depth} {target}: FAILED "
                          f"({type(e).__name__}: {e})", flush=True)
                    continue
                m, s = float(np.median(ts)), float(np.std(ts))
                # Microsecond resolution: metal's small sizes take ~0.5-1 ms.
                rows.append(f"{n},{depth},{target},{m:.6f},{s:.6f}")
                print(f"n={n:2d} depth={depth:2d} {target:8s}: {m:8.4f}s ± {s:.4f}",
                      flush=True)
    tmp = OUT + ".part"
    with open(tmp, "w") as f:
        f.write("\n".join(rows) + "\n")
    os.replace(tmp, OUT)
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
