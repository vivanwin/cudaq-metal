"""Same hardware-efficient ansatz, synthesized through Classiq.

Emits the same primitive gates as `brickwork_native.py` (RY on every qubit,
then CX-RZ-CX brickwork per layer), so timings are directly comparable.
Runs on Metal via qprog_to_cudaq_kernel.

Requires ``classiq.authenticate()`` once.
"""
from functools import reduce
import operator

import cudaq_metal        # ← the one metal-related line
import cudaq
from classiq import (qfunc, QArray, QBit, Output, allocate,
                     RY, RZ, CX, synthesize, qprog_to_cudaq_kernel,
                     pauli_operator_to_cudaq_spin_op, Pauli)

N = 16
DEPTH = 5
THETA, PHI = 0.5, 0.7


@qfunc
def brickwork(q: QArray[QBit, N]):
    for layer in range(DEPTH):
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
    brickwork(q)


qprog  = synthesize(main)
kernel = qprog_to_cudaq_kernel(qprog)
H      = pauli_operator_to_cudaq_spin_op(
             reduce(operator.add, (Pauli.Z(i) for i in range(N))))

cudaq.set_target("metal")
print(f"<Σ Zᵢ> on {N}-qubit Classiq brickwork (depth={DEPTH}): "
      f"{cudaq.observe(kernel, H).expectation():+.4f}")
