"""Random-SU(4) brickwork circuit — Classiq synthesis + CUDA-Q Metal execution.

The Classiq compiler synthesizes the ansatz, ``qprog_to_cudaq_kernel`` converts
it into a CUDA-Q kernel, and ``cudaq.observe`` runs on the Apple GPU. Fully
reliable across arbitrary parametric loops thanks to cudaq_metal's transparent
reliability layer (see the top-level README).

Requires ``classiq.authenticate()`` once.
"""
from functools import reduce
import operator

import cudaq_metal        # ← the one metal-related line
import cudaq
from scipy.stats import unitary_group
from classiq import (qfunc, QArray, QBit, Output, allocate, unitary, repeat,
                     synthesize, qprog_to_cudaq_kernel,
                     pauli_operator_to_cudaq_spin_op, Pauli)

N = 15                                              # qubits — scale me
DEPTH = 5

SU4 = unitary_group.rvs(4, random_state=42).tolist()


@qfunc
def brickwork(q: QArray[QBit, N]):
    repeat(DEPTH, lambda layer: [
        unitary(SU4, [q[j], q[j + 1]])
        for j in range(0, N - 1, 2)
    ])


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
