"""Benchmark the two examples across qubit counts.

For each N, times one `cudaq.observe(...).expectation()` call — the exact code
the example scripts run. Same brickwork circuit (random SU(4), depth=5) for
both, so the only variable is the engine: native @cudaq.kernel on qpp-cpu vs
Classiq synthesize + qprog_to_cudaq_kernel on metal.

Writes benchmarks/results/examples.csv; plot_examples.py renders the PNG.
"""
import operator
import time
from functools import reduce

import numpy as np
from scipy.stats import unitary_group

import cudaq_metal   # auto-registers metal + transparent mode
import cudaq
from cudaq import spin
from classiq import (qfunc, QArray, QBit, Output, allocate, unitary, repeat,
                     synthesize, qprog_to_cudaq_kernel,
                     pauli_operator_to_cudaq_spin_op, Pauli)

SU4 = unitary_group.rvs(4, random_state=42)
SU4_LIST = SU4.tolist()
DEPTH = 5
TRIALS = 2
N_MIN, N_MAX_NATIVE, N_MAX_CLASSIQ = 8, 18, 20

cudaq.register_operation("su4", SU4)


@cudaq.kernel
def native_brickwork(n: int, depth: int):
    q = cudaq.qvector(n)
    for layer in range(depth):
        j = layer % 2
        while j + 1 < n:
            su4(q[j], q[j + 1])
            j += 2


def time_native(N, depth):
    cudaq.set_target("qpp-cpu")
    H = reduce(operator.add, (spin.z(i) for i in range(N)))
    # warmup once — first call pays JIT cost
    cudaq.observe(native_brickwork, H, N, depth)
    ts = []
    for _ in range(TRIALS):
        t0 = time.perf_counter()
        cudaq.observe(native_brickwork, H, N, depth).expectation()
        ts.append(time.perf_counter() - t0)
    return ts


def make_classiq_qprog(N, depth):
    # `qfunc` closes over N & depth at decoration time — build fresh per N.
    @qfunc
    def bw(q: QArray[QBit, N]):
        repeat(depth, lambda layer: [
            unitary(SU4_LIST, [q[j], q[j + 1]])
            for j in range(0, N - 1, 2)
        ])

    @qfunc
    def main(q: Output[QArray[QBit, N]]):
        allocate(q)
        bw(q)

    return synthesize(main)


def time_classiq(N, depth):
    cudaq.set_target("metal")
    qprog = make_classiq_qprog(N, depth)
    kernel = qprog_to_cudaq_kernel(qprog)
    H = pauli_operator_to_cudaq_spin_op(
        reduce(operator.add, (Pauli.Z(i) for i in range(N))))
    cudaq.observe(kernel, H)  # warmup
    ts = []
    for _ in range(TRIALS):
        t0 = time.perf_counter()
        cudaq.observe(kernel, H).expectation()
        ts.append(time.perf_counter() - t0)
    return ts


def main():
    out = "benchmarks/results/examples.csv"
    with open(out, "w") as f:
        f.write("n,depth,kind,mean,std\n")
        for n in range(N_MIN, max(N_MAX_NATIVE, N_MAX_CLASSIQ) + 1):
            if n <= N_MAX_NATIVE:
                ts = time_native(n, DEPTH)
                m, s = float(np.mean(ts)), float(np.std(ts))
                f.write(f"{n},{DEPTH},native_qpp,{m:.4f},{s:.4f}\n"); f.flush()
                print(f"n={n} native_qpp:    {m:6.2f}s ± {s:.2f}", flush=True)
            if n <= N_MAX_CLASSIQ:
                ts = time_classiq(n, DEPTH)
                m, s = float(np.mean(ts)), float(np.std(ts))
                f.write(f"{n},{DEPTH},classiq_metal,{m:.4f},{s:.4f}\n"); f.flush()
                print(f"n={n} classiq_metal: {m:6.2f}s ± {s:.2f}", flush=True)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
