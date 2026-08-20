"""Full 2x2 sweep: {native @cudaq.kernel, Classiq qprog_to_cudaq_kernel} × {qpp-cpu, metal}.

Native+metal spawns a subprocess per measurement — native kernels aren't tracked
by cudaq_metal's transparent layer, so each fresh process gets an independent
chance of JIT-poisoning; we retry with a new subprocess if the result matches
the poisoned |0..0> outcome (Σ Zᵢ = N exactly).
"""
import operator
import os
import subprocess
import sys
import time
from functools import reduce

import numpy as np
from scipy.stats import unitary_group

import cudaq_metal
import cudaq
from cudaq import spin
from classiq import (qfunc, QArray, QBit, Output, allocate, unitary, repeat,
                     synthesize, qprog_to_cudaq_kernel,
                     pauli_operator_to_cudaq_spin_op, Pauli)

SU4 = unitary_group.rvs(4, random_state=42)
SU4_LIST = SU4.tolist()
DEPTH = 5
TRIALS = 2
HERE = os.path.dirname(os.path.abspath(__file__))
WORKER = os.path.join(HERE, "_native_worker.py")

cudaq.register_operation("su4", SU4)


@cudaq.kernel
def native_bw(n: int, depth: int):
    q = cudaq.qvector(n)
    for layer in range(depth):
        j = layer % 2
        while j + 1 < n:
            su4(q[j], q[j + 1])
            j += 2


def _native_in_process(N, depth, target):
    cudaq.set_target(target)
    H = reduce(operator.add, (spin.z(i) for i in range(N)))
    cudaq.observe(native_bw, H, N, depth)                # warmup
    ts = []
    for _ in range(TRIALS):
        t0 = time.perf_counter()
        cudaq.observe(native_bw, H, N, depth).expectation()
        ts.append(time.perf_counter() - t0)
    return ts


def _native_subprocess(N, depth, target, max_retries=10):
    """Fresh subprocess per trial — JIT-isolation for native+metal."""
    ts = []
    for _ in range(TRIALS):
        for attempt in range(max_retries):
            r = subprocess.run(
                [sys.executable, WORKER, str(N), str(depth), target],
                capture_output=True, text=True, timeout=120,
            )
            line = r.stdout.strip().split("\n")[-1]
            parts = line.split()
            t, v = float(parts[0]), float(parts[1])
            if abs(v - N) > 1e-6:                        # not poisoned |0..0>
                ts.append(t)
                break
        else:
            raise RuntimeError(f"[N={N}] JIT-poison every attempt")
    return ts


def _classiq_qprog(N, depth):
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


def _classiq_in_process(N, depth, target):
    cudaq.set_target(target)
    qprog = _classiq_qprog(N, depth)
    kernel = qprog_to_cudaq_kernel(qprog)
    H = pauli_operator_to_cudaq_spin_op(
        reduce(operator.add, (Pauli.Z(i) for i in range(N))))
    cudaq.observe(kernel, H)                             # warmup
    ts = []
    for _ in range(TRIALS):
        t0 = time.perf_counter()
        cudaq.observe(kernel, H).expectation()
        ts.append(time.perf_counter() - t0)
    return ts


CONFIGS = [
    # (kind, callable, n_range)
    ("native_qpp",    lambda n: _native_in_process(n, DEPTH, "qpp-cpu"),  range(8, 18)),
    ("native_metal",  lambda n: _native_subprocess(n, DEPTH, "metal"),    range(8, 21)),
    ("classiq_qpp",   lambda n: _classiq_in_process(n, DEPTH, "qpp-cpu"), range(8, 18)),
    ("classiq_metal", lambda n: _classiq_in_process(n, DEPTH, "metal"),   range(8, 21)),
]

out = os.path.join(HERE, "results", "examples.csv")
with open(out, "w") as f:
    f.write("n,depth,kind,mean,std\n")
    for kind, fn, r in CONFIGS:
        for n in r:
            try:
                ts = fn(n)
            except Exception as e:
                print(f"n={n} {kind}: FAILED ({e})", flush=True); continue
            m, s = float(np.mean(ts)), float(np.std(ts))
            f.write(f"{n},{DEPTH},{kind},{m:.4f},{s:.4f}\n"); f.flush()
            print(f"n={n:2d} {kind:14s}: {m:7.3f}s ± {s:.3f}", flush=True)
print(f"\nwrote {out}")
