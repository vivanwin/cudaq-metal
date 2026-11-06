"""Hardware-efficient ansatz — native CUDA-Q kernel.

Per layer: RY(θ) on every qubit, then a brickwork of CX-RZ(φ)-CX on
alternating even/odd pairs. Same primitive gate sequence Classiq emits after
decomposition, so `brickwork_classiq.py` and this script apply the *same*
circuit — engine × target timings are then directly comparable.

Runs on the Apple GPU via the `metal` target; change TARGET to "qpp-cpu" to
compare against CUDA-Q's CPU simulator.
"""
import cudaq_metal  # noqa: F401  ← the one metal-related line (registers the target)
import cudaq
from cudaq import spin
from functools import reduce
import operator

N = 16
DEPTH = 5
TARGET = "metal"
THETA, PHI = 0.5, 0.7


@cudaq.kernel
def brickwork(n: int, depth: int, theta: float, phi: float):
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


H = reduce(operator.add, (spin.z(i) for i in range(N)))

cudaq.set_target(TARGET)
print(f"<Σ Zᵢ> on {N}-qubit brickwork (depth={DEPTH}, {TARGET}): "
      f"{cudaq.observe(brickwork, H, N, DEPTH, THETA, PHI).expectation():+.4f}")
