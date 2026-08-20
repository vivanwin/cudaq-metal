"""Random-SU(4) brickwork circuit — native CUDA-Q kernel.

Uses `cudaq.set_target("qpp-cpu")` by default. You can switch to "metal" but
be aware: native `@cudaq.kernel` objects aren't tracked by cudaq_metal's
transparent reliability layer (only `qprog_to_cudaq_kernel`-derived kernels
are). On Darwin arm64 you'll occasionally see the |0..0> outcome and need to
re-run — see BUG_INVESTIGATION.md. For fully reliable Metal execution, use
the Classiq path in `brickwork_classiq.py`.
"""
import cudaq_metal        # ← the one metal-related line (harmless on qpp-cpu)
import cudaq
from cudaq import spin
from scipy.stats import unitary_group

N = 16                                              # qubits — scale me
DEPTH = N                                           # QV-shape: depth == width

cudaq.register_operation("su4", unitary_group.rvs(4, random_state=42))


@cudaq.kernel
def brickwork(n: int, depth: int):
    q = cudaq.qvector(n)
    for layer in range(depth):
        j = layer % 2
        while j + 1 < n:
            su4(q[j], q[j + 1])
            j += 2


H = sum(spin.z(i) for i in range(N))                # Σ Zᵢ — global magnetization

cudaq.set_target("qpp-cpu")                          # swap to "metal" — see docstring
print(f"<Σ Zᵢ> on {N}-qubit brickwork (depth={DEPTH}): "
      f"{cudaq.observe(brickwork, H, N, DEPTH).expectation():+.4f}")
