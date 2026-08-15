"""Validate cudaq.get_state() on the Metal backend vs qpp-cpu.

Compares the full statevector for several circuits (including non-trivial
phases and 2-qubit SU(4) gates). Reports state fidelity |<qpp|metal>|^2 and the
max per-amplitude difference after aligning global phase.
"""
import numpy as np
import cudaq

rng = np.random.default_rng(0)
SU4 = np.linalg.qr(rng.standard_normal((4, 4)) + 1j * rng.standard_normal((4, 4)))[0]
cudaq.register_operation("su4", SU4)


@cudaq.kernel
def ghz(n: int):
    q = cudaq.qvector(n)
    h(q[0])
    for i in range(n - 1):
        x.ctrl(q[i], q[i + 1])


@cudaq.kernel
def phasey():
    q = cudaq.qvector(3)
    h(q[0]); h(q[1]); h(q[2])
    rx(0.7, q[0]); ry(1.3, q[1]); rz(0.9, q[2])
    t(q[0]); s(q[1])
    x.ctrl(q[0], q[2])
    rz(2.1, q[1])


@cudaq.kernel
def su4brick(n: int):
    q = cudaq.qvector(n)
    for i in range(n):
        h(q[i])
    for layer in range(3):
        offset = layer % 2
        j = offset
        while j + 1 < n:
            su4(q[j], q[j + 1])
            j += 2


def sv(target, kernel, *args):
    cudaq.set_target(target)
    s = np.array(cudaq.get_state(kernel, *args), copy=True).astype(np.complex128)
    cudaq.reset_target()
    return s


def compare(name, kernel, *args):
    a = sv("qpp-cpu", kernel, *args)
    b = sv("metal", kernel, *args)
    fid = abs(np.vdot(a, b)) ** 2 / (np.vdot(a, a).real * np.vdot(b, b).real)
    phase = np.vdot(a, b)
    phase = phase / abs(phase) if abs(phase) > 1e-12 else 1.0
    maxdiff = np.max(np.abs(a - b * np.conj(phase)))
    ok = fid > 0.9999 and maxdiff < 1e-4
    print(f"[{'OK ' if ok else 'BAD'}] {name:12s} dim={a.size:<6d} "
          f"fidelity={fid:.6f}  max|Δamp|={maxdiff:.2e}")
    return ok


if __name__ == "__main__":
    ok = True
    ok &= compare("ghz(4)", ghz, 4)
    ok &= compare("phasey", phasey)
    ok &= compare("su4brick(6)", su4brick, 6)
    print("ALL PASS" if ok else "FAILURES")
