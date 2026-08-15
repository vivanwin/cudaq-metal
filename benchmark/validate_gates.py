"""Validate the Metal backend's multi-qubit-target gate handling against qpp-cpu.

Strategy: drive *identical* cudaq operations on both backends and compare the
sampled distributions. This sidesteps any absolute matrix-convention reasoning —
we only require Metal to agree with the reference for the same frontend ops.
Covers SWAP (deterministic) and a random SU(4) custom op (as in quantum volume),
including non-adjacent target qubits.
"""
import sys
import numpy as np
import cudaq

rng = np.random.default_rng(1234)


def random_unitary(dim: int) -> np.ndarray:
    z = (rng.standard_normal((dim, dim)) + 1j * rng.standard_normal((dim, dim)))
    q, r = np.linalg.qr(z)
    # Fix phases so q is Haar-distributed (not essential for the test).
    d = np.diagonal(r)
    return q * (d / np.abs(d))


SU4 = random_unitary(4)
cudaq.register_operation("su4", SU4)


@cudaq.kernel
def swap_basis(bits: list[int]):
    q = cudaq.qvector(2)
    for i in range(2):
        if bits[i] == 1:
            x(q[i])
    swap(q[0], q[1])
    mz(q)


@cudaq.kernel
def su4_adjacent():
    q = cudaq.qvector(3)
    h(q[0])
    h(q[1])
    su4(q[0], q[1])  # 2-qubit-target gate on adjacent qubits
    mz(q)


@cudaq.kernel
def su4_nonadjacent():
    q = cudaq.qvector(3)
    h(q[0])
    h(q[2])
    su4(q[0], q[2])  # target qubits are NOT adjacent — tests bit placement
    mz(q)


def dist(target, kernel, *args, shots=40000):
    cudaq.set_target(target)
    counts = cudaq.sample(kernel, *args, shots_count=shots)
    total = sum(counts.count(k) for k in counts)
    d = {k: counts.count(k) / total for k in counts}
    cudaq.reset_target()
    return d


def tvd(a, b):
    keys = set(a) | set(b)
    return 0.5 * sum(abs(a.get(k, 0.0) - b.get(k, 0.0)) for k in keys)


def compare(name, kernel, *args, tol=0.03):
    dq = dist("qpp-cpu", kernel, *args)
    dm = dist("metal", kernel, *args)
    d = tvd(dq, dm)
    ok = d < tol
    print(f"[{'OK ' if ok else 'BAD'}] {name:16s} TVD={d:.4f}")
    if not ok:
        print("    qpp:", {k: round(v, 3) for k, v in sorted(dq.items())})
        print("    mtl:", {k: round(v, 3) for k, v in sorted(dm.items())})
    return ok


if __name__ == "__main__":
    ok = True
    # SWAP is deterministic on basis states -> distributions are point masses.
    for bits in ([0, 1], [1, 0], [1, 1]):
        ok &= compare(f"swap{bits}", swap_basis, bits, tol=0.01)
    ok &= compare("su4_adjacent", su4_adjacent)
    ok &= compare("su4_nonadjacent", su4_nonadjacent)
    print("ALL PASS" if ok else "FAILURES PRESENT")
    sys.exit(0 if ok else 1)
