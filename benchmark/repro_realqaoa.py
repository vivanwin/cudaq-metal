"""Minimal reproducer for the Metal backend + kernel-builder deferred-exception bug.

Run this 4-5 times to see the non-determinism: `<H>=-0.000000` means a
deferred-exception failure (the loud `[metal] WARNING: sample() ...` banner
appears on stderr). Any non-zero value is a correct evaluation. See
../BUG_INVESTIGATION.md for the root cause and mitigation.
"""
import cudaq
import numpy as np
import networkx as nx
from classiq import (qfunc, CReal, CArray, QArray, QBit, Output, RY, RZ,
                     allocate, phase, repeat, synthesize, qprog_to_cudaq_kernel)
from cudaq import spin

NUM_NODES = 15
NUM_LAYERS = 3
GRAPH_SEED = 42

_G = nx.gnp_random_graph(NUM_NODES, p=0.15, seed=GRAPH_SEED)
_r = np.random.default_rng(GRAPH_SEED)
EDGES = [(i, j, float(_r.uniform(0.5, 2.0))) for i, j in _G.edges()]
TW = sum(w for _, _, w in EDGES)


def maxcut_cost(v, edges=EDGES, tw=TW):
    return sum(w * (v[i] + v[j] - 2 * v[i] * v[j]) for i, j, w in edges) / tw


np.random.seed(42)
THETAS = np.random.uniform(0.3, 1.5, NUM_NODES).tolist()


@qfunc
def warm_start_init(v: Output[QArray[QBit, NUM_NODES]]) -> None:
    allocate(v)
    for i in range(NUM_NODES):
        RY(THETAS[i], v[i])


@qfunc
def cost_layer(gamma: CReal, v: QArray[QBit]) -> None:
    phase(-maxcut_cost(v), gamma)


@qfunc
def warm_start_mixer_layer(beta: CReal, v: QArray[QBit]) -> None:
    for i in range(NUM_NODES):
        RY(THETAS[i], v[i])
        RZ(-2 * beta, v[i])
        RY(-THETAS[i], v[i])


@qfunc
def main(params: CArray[CReal, NUM_LAYERS * 2],
         v: Output[QArray[QBit, NUM_NODES]]) -> None:
    warm_start_init(v)
    gammas = params[0:NUM_LAYERS]
    betas = params[NUM_LAYERS:2 * NUM_LAYERS]
    repeat(NUM_LAYERS,
           lambda i: [cost_layer(gammas[i], v), warm_start_mixer_layer(betas[i], v)])


qprog = synthesize(main)
kernel = qprog_to_cudaq_kernel(qprog)

H = None
for i, j, w in EDGES:
    term = (w / TW) * 0.5 * (spin.z(i) * spin.z(j) - spin.i(0))
    H = term if H is None else H + term

cudaq.set_target("metal")
np.random.seed(0)
for k in range(4):
    p = list(np.random.uniform(0.1, 3.14, 2 * NUM_LAYERS))
    e = cudaq.observe(kernel, H, *p).expectation()
    print(f"call {k}: <H>={e:+.6f}"
          + ("  <-- deferred-exception failure" if abs(e) < 1e-9 else ""))
