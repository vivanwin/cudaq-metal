import operator, sys, time
from functools import reduce
from scipy.stats import unitary_group

import cudaq_metal  # harmless on non-metal targets
import cudaq
from cudaq import spin

N     = int(sys.argv[1])
DEPTH = int(sys.argv[2])
TGT   = sys.argv[3]

cudaq.register_operation("su4", unitary_group.rvs(4, random_state=42))


@cudaq.kernel
def brickwork(n: int, depth: int):
    q = cudaq.qvector(n)
    for layer in range(depth):
        j = layer % 2
        while j + 1 < n:
            su4(q[j], q[j + 1])
            j += 2


cudaq.set_target(TGT)
H = reduce(operator.add, (spin.z(i) for i in range(N)))
cudaq.observe(brickwork, H, N, DEPTH)                    # warmup
t0 = time.perf_counter()
v = cudaq.observe(brickwork, H, N, DEPTH).expectation()
print(f"{time.perf_counter() - t0} {v}")
