"""Measure gate-fusion impact across circuit families (not just QV).

Times cudaq.get_state (pure simulation) on the metal backend for several
representative circuits. Run twice to A/B fusion:
    python benchmarks/fusion_impact.py [n]                         # fusion ON
    MSV_DISABLE_FUSION=1 python benchmarks/fusion_impact.py [n]    # fusion OFF

Circuits:
  trotter_tfim  : Hamiltonian-sim step, rzz = CX-Rz-CX + rx  (fuses very well)
  hw_efficient  : VQE/QML ansatz, Ry layer then CX ladder     (order-limited)
  brickwork_su4 : dense random SU(4) on pairs                  (nothing to absorb)
"""
import os
import sys
import time

import numpy as np
import cudaq_metal  # noqa: F401  (registers the target; must precede cudaq)
import cudaq

rng = np.random.default_rng(3)
cudaq.register_operation(
    "su4", np.linalg.qr(rng.standard_normal((4, 4)) + 1j * rng.standard_normal((4, 4)))[0]
)


@cudaq.kernel
def trotter_tfim(n: int, layers: int):
    q = cudaq.qvector(n)
    for _ in range(layers):
        for i in range(n):
            rx(0.3, q[i])
        for i in range(n - 1):  # rzz(0.5) = CX . Rz . CX  (fuses to one 2q block)
            x.ctrl(q[i], q[i + 1])
            rz(0.5, q[i + 1])
            x.ctrl(q[i], q[i + 1])


@cudaq.kernel
def hw_efficient(n: int, layers: int):
    q = cudaq.qvector(n)
    for _ in range(layers):
        for i in range(n):
            ry(0.4, q[i])       # all rotations first ...
        for i in range(n - 1):  # ... then the entangler ladder
            x.ctrl(q[i], q[i + 1])


@cudaq.kernel
def brickwork_su4(n: int, layers: int):
    q = cudaq.qvector(n)
    for layer in range(layers):
        offset = layer % 2
        j = offset
        while j + 1 < n:
            su4(q[j], q[j + 1])
            j += 2


def bench(kernel, n, layers, repeats=3):
    cudaq.set_target("metal")
    s = cudaq.get_state(kernel, n, layers); del s  # warmup
    best = 1e18
    for _ in range(repeats):
        t0 = time.perf_counter()
        s = cudaq.get_state(kernel, n, layers); del s
        best = min(best, (time.perf_counter() - t0) * 1e3)
    cudaq.reset_target()
    return best


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 20
    tag = "OFF" if os.environ.get("MSV_DISABLE_FUSION") else "ON "
    circuits = [
        ("trotter_tfim", trotter_tfim, 12),
        ("hw_efficient", hw_efficient, 24),
        ("brickwork_su4", brickwork_su4, 20),
    ]
    print(f"# fusion={tag}  n={n}  metal get_state (ms, best-of)")
    for name, k, layers in circuits:
        print(f"  {name:16s} layers={layers:<3d} {bench(k, n, layers):9.2f}", flush=True)
