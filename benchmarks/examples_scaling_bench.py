"""Full 2x2 sweep with matched gate set — {native @cudaq.kernel, Classiq
qprog_to_cudaq_kernel} × {qpp-cpu, metal}. Both paths emit the same primitive
gate sequence (RY layer + CX-RZ-CX brickwork), so the CPU-vs-GPU comparison
isolates the runtime engine cleanly.

Native + metal spawns a subprocess per measurement. That isolation was needed
on CUDA-Q <= 0.15 (Darwin JIT bug, see BUG_INVESTIGATION.md) and is kept so the
published numbers stay reproducible; on CUDA-Q 0.16 it is no longer required.

Only `cudaq.observe(...)` is timed; synth / kernel-construction / warmup
happen before t0.
"""
import operator, os, subprocess, sys, time
from functools import reduce

import numpy as np

import cudaq_metal
import cudaq
from cudaq import spin
from classiq import (qfunc, QArray, QBit, Output, allocate,
                     RY, RZ, CX, synthesize, qprog_to_cudaq_kernel,
                     pauli_operator_to_cudaq_spin_op, Pauli)

DEPTH, TRIALS = 5, 2
THETA, PHI = 0.5, 0.7
HERE = os.path.dirname(os.path.abspath(__file__))
WORKER = os.path.join(HERE, "_native_worker.py")


@cudaq.kernel
def native_bw(n: int, depth: int, theta: float, phi: float):
    q = cudaq.qvector(n)
    for layer in range(depth):
        for i in range(n):
            ry(theta, q[i])
        offset = layer % 2
        j = offset
        while j + 1 < n:
            x.ctrl(q[j], q[j + 1])
            rz(phi, q[j + 1])
            x.ctrl(q[j], q[j + 1])
            j += 2


def _native_in_process(N, depth, target):
    cudaq.set_target(target)
    H = reduce(operator.add, (spin.z(i) for i in range(N)))
    cudaq.observe(native_bw, H, N, depth, THETA, PHI)
    ts = []
    for _ in range(TRIALS):
        t0 = time.perf_counter()
        cudaq.observe(native_bw, H, N, depth, THETA, PHI).expectation()
        ts.append(time.perf_counter() - t0)
    return ts


def _native_subprocess(N, depth, target, max_retries=10):
    ts = []
    for _ in range(TRIALS):
        for attempt in range(max_retries):
            r = subprocess.run(
                [sys.executable, WORKER, str(N), str(depth), target],
                capture_output=True, text=True, timeout=180,
            )
            line = r.stdout.strip().split("\n")[-1]
            parts = line.split()
            t, v = float(parts[0]), float(parts[1])
            if abs(v - N) > 1e-6:                    # not poisoned |0..0>
                ts.append(t); break
        else:
            raise RuntimeError(f"[N={N}] JIT-poison every attempt")
    return ts


def _classiq_qprog(N, depth):
    @qfunc
    def bw(q: QArray[QBit, N]):
        for layer in range(depth):
            for i in range(N):
                RY(THETA, q[i])
            offset = layer % 2
            for j in range(offset, N - 1, 2):
                CX(q[j], q[j + 1])
                RZ(PHI, q[j + 1])
                CX(q[j], q[j + 1])

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
    cudaq.observe(kernel, H)
    ts = []
    for _ in range(TRIALS):
        t0 = time.perf_counter()
        cudaq.observe(kernel, H).expectation()
        ts.append(time.perf_counter() - t0)
    return ts


CONFIGS = [
    ("native_qpp",    lambda n: _native_in_process(n, DEPTH, "qpp-cpu"),  range(8, 18)),
    ("native_metal",  lambda n: _native_subprocess(n, DEPTH, "metal"),    range(8, 26)),
    ("classiq_qpp",   lambda n: _classiq_in_process(n, DEPTH, "qpp-cpu"), range(8, 18)),
    ("classiq_metal", lambda n: _classiq_in_process(n, DEPTH, "metal"),   range(8, 26)),
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
            print(f"n={n:2d} {kind:14s}: {m:8.4f}s ± {s:.4f}", flush=True)
print(f"\nwrote {out}")
