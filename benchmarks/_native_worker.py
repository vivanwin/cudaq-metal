"""Single-observe worker for the native + metal path — subprocess-per-measurement."""
import operator, sys, time
from functools import reduce

import cudaq_metal
import cudaq
from cudaq import spin

N     = int(sys.argv[1])
DEPTH = int(sys.argv[2])
TGT   = sys.argv[3]
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


cudaq.set_target(TGT)
H = reduce(operator.add, (spin.z(i) for i in range(N)))
cudaq.observe(brickwork, H, N, DEPTH, THETA, PHI)             # warmup
t0 = time.perf_counter()
v = cudaq.observe(brickwork, H, N, DEPTH, THETA, PHI).expectation()
print(f"{time.perf_counter() - t0} {v}")
