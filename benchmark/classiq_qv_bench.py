"""Quantum Volume scalability benchmark, authored in Classiq, run on Metal.

Pipeline per n:
  build QV(n) model  ->  synthesize() (Classiq cloud)  ->  qprog_to_cudaq_kernel()
  ->  cudaq.sample() on metal (and qpp-cpu), timed.

QV construction follows the Classiq QV notebook: n layers, each a random pairing
of the n qubits with a Haar-random SU(4) between each pair.

Requires (once):  ~/.cudaq-metal/venv/bin/python -c "import classiq; classiq.authenticate()"
Usage: classiq_qv_bench.py [minQ] [maxQ] [shots] [repeats] [seed] [targets]
"""
import math
import random
import sys
import time

import numpy as np
from numpy.linalg import qr
from scipy.stats import unitary_group

import cudaq
from classiq import qfunc, QArray, Output, allocate, repeat, unitary, synthesize
from classiq import qprog_to_cudaq_kernel, Preferences


def haar(n):
    z = unitary_group.rvs(n) + 1j * unitary_group.rvs(n)
    q, r = qr(z)
    lam = np.diag([r[i, i] / abs(r[i, i]) for i in range(n)])
    return np.dot(q, lam)


def qv_model(N):
    @qfunc
    def qv_layer(target: QArray):
        order = list(range(N))
        random.shuffle(order)
        for idx in range(N // 2):
            a = order[idx]
            b = order[N // 2 + idx]
            unitary(haar(4).tolist(), [target[a], target[b]])

    @qfunc
    def main(target: Output[QArray]):
        allocate(N, target)
        repeat(N, lambda _: qv_layer(target))

    return main


def make_kernel(n, seed, synth_timeout):
    random.seed(seed)
    np.random.seed(seed)
    qprog = synthesize(qv_model(n), preferences=Preferences(timeout_seconds=synth_timeout))
    res = qprog_to_cudaq_kernel(qprog)
    if isinstance(res, tuple):
        kernel, *fargs = res
    else:
        kernel, fargs = res, []
    return kernel, fargs, qprog


def _run_once(metric, kernel, fargs, shots):
    if metric == "state":
        s = cudaq.get_state(kernel, *fargs)  # pure statevector, no sampling
        del s
    else:
        cudaq.sample(kernel, *fargs, shots_count=shots)


def bench(target, kernel, fargs, shots, repeats, last_ms, metric):
    cudaq.set_target(target)
    # Skip a warmup run for already-slow targets (avoids a second long run;
    # JIT is a fixed ms-scale cost, negligible next to a multi-second sim).
    if last_ms is None or last_ms < 1000.0:
        _run_once(metric, kernel, fargs, shots)  # warmup
        n_meas = repeats
    else:
        n_meas = 1
    best = 1e18
    for _ in range(n_meas):
        t0 = time.perf_counter()
        _run_once(metric, kernel, fargs, shots)
        best = min(best, (time.perf_counter() - t0) * 1e3)
    cudaq.reset_target()
    return best


if __name__ == "__main__":
    minQ = int(sys.argv[1]) if len(sys.argv) > 1 else 4
    maxQ = int(sys.argv[2]) if len(sys.argv) > 2 else 26
    shots = int(sys.argv[3]) if len(sys.argv) > 3 else 1000
    repeats = int(sys.argv[4]) if len(sys.argv) > 4 else 2
    seed = int(sys.argv[5]) if len(sys.argv) > 5 else 42
    targets = sys.argv[6].split(",") if len(sys.argv) > 6 else ["qpp-cpu", "metal"]
    budget_s = float(sys.argv[7]) if len(sys.argv) > 7 else 30.0
    metric = sys.argv[8] if len(sys.argv) > 8 else "sample"  # "sample" | "state"
    budget_ms = budget_s * 1000.0
    # Stop-AFTER-exceed: run each n, and retire a target only once its measured
    # time actually exceeds the budget (so the last run per target is over-budget).
    synth_to = max(int(budget_s), 60)

    print(f"# Classiq QV, metric={metric}, shots={shots}, seed={seed}, "
          f"budget={budget_s:.0f}s/run (retire after a run exceeds budget)")
    cols = ["synth(s)"] + [t + "(ms)" for t in targets]
    hdr = "n    | " + " | ".join(f"{c:>13}" for c in cols)
    print(hdr)
    print("-" * len(hdr))

    active = {t: True for t in targets}
    last = {t: None for t in targets}
    for n in range(minQ, maxQ + 1):
        if not any(active.values()):
            print("# all targets retired (over budget); stopping.")
            break
        try:
            t0 = time.perf_counter()
            kernel, fargs, qprog = make_kernel(n, seed + n, synth_to)
            synth_s = time.perf_counter() - t0
        except Exception as e:
            print(f"{n:<4} | synth/convert failed ({type(e).__name__}); stopping. {e}")
            break
        cells = [f"{synth_s:13.2f}"]
        for t in targets:
            if not active[t]:
                cells.append(f"{'skip':>13}")
                continue
            ms = bench(t, kernel, fargs, shots, repeats, last[t], metric)
            last[t] = ms
            over = " *" if ms > budget_ms else ""
            cells.append(f"{ms:11.2f}{over:>2}")
            if ms > budget_ms:  # last run exceeded budget -> retire this target
                active[t] = False
        print(f"{n:<4} | " + " | ".join(cells), flush=True)
    print(f"# '*' marks the over-budget run that retired a target (>{budget_s:.0f}s).")
