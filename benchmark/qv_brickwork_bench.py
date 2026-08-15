"""Scalability benchmark with 2-qubit gates: a brickwork of random SU(4) gates
(a quantum-volume-style random circuit). Exercises the general k-target Metal
kernel at scale and compares metal vs qpp-cpu. Timing is independent of the
matrix values, so a single fixed random SU(4) is reused for all gates.

Usage: qv_brickwork_bench.py [minQ] [maxQ] [shots] [repeats] [targets]
  targets: comma list, default "qpp-cpu,metal"
"""
import sys
import time
import numpy as np
import cudaq

rng = np.random.default_rng(7)


def random_unitary(dim):
    z = rng.standard_normal((dim, dim)) + 1j * rng.standard_normal((dim, dim))
    q, r = np.linalg.qr(z)
    d = np.diagonal(r)
    return q * (d / np.abs(d))


cudaq.register_operation("su4", random_unitary(4))


@cudaq.kernel
def brickwork(n: int, depth: int):
    q = cudaq.qvector(n)
    for layer in range(depth):
        offset = layer % 2
        i = offset
        while i + 1 < n:
            su4(q[i], q[i + 1])
            i += 2
    mz(q)


def bench(target, n, depth, shots, repeats):
    cudaq.set_target(target)
    cudaq.sample(brickwork, n, depth, shots_count=shots)  # warmup
    best = 1e18
    for _ in range(repeats):
        t0 = time.perf_counter()
        cudaq.sample(brickwork, n, depth, shots_count=shots)
        best = min(best, (time.perf_counter() - t0) * 1e3)
    cudaq.reset_target()
    return best


if __name__ == "__main__":
    minQ = int(sys.argv[1]) if len(sys.argv) > 1 else 8
    maxQ = int(sys.argv[2]) if len(sys.argv) > 2 else 16
    shots = int(sys.argv[3]) if len(sys.argv) > 3 else 1000
    repeats = int(sys.argv[4]) if len(sys.argv) > 4 else 3
    targets = sys.argv[5].split(",") if len(sys.argv) > 5 else ["qpp-cpu", "metal"]

    print(f"# brickwork SU(4), depth=n, shots={shots}, repeats={repeats}")
    hdr = "n    | depth | " + " | ".join(f"{t+'(ms)':>14}" for t in targets)
    print(hdr)
    print("-" * len(hdr))
    for n in range(minQ, maxQ + 1):
        depth = n
        row = f"{n:<4} | {depth:<5} | "
        row += " | ".join(f"{bench(t, n, depth, shots, repeats):14.2f}" for t in targets)
        print(row, flush=True)
