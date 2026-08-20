"""Hardware-efficient ansatz — native CUDA-Q kernel.

Per layer: RY(θ) on every qubit, then a brickwork of CX-RZ(φ)-CX on
alternating even/odd pairs. Same primitive gate sequence Classiq emits after
decomposition, so `brickwork_classiq.py` and this script apply the *same*
circuit — engine × target timings are then directly comparable.

Defaults to `qpp-cpu`. See docstring caveat in the top-level README about
native kernels on `--target metal` (they bypass cudaq_metal's transparent
reliability layer).
"""
import cudaq_metal        # ← the one metal-related line
import cudaq
from cudaq import spin
from functools import reduce
import operator

N = 16
DEPTH = 5
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

cudaq.set_target("qpp-cpu")                      # swap to "metal" per docstring
print(f"<Σ Zᵢ> on {N}-qubit brickwork (depth={DEPTH}): "
      f"{cudaq.observe(brickwork, H, N, DEPTH, THETA, PHI).expectation():+.4f}")
