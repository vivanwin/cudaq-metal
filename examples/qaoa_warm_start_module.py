"""User-provided module: defines build() returning (kernel, hamiltonian)."""
import numpy as np, networkx as nx
from classiq import *
from cudaq import spin

NUM_NODES = 15
NUM_LAYERS = 3
GRAPH_SEED = 42

_G = nx.gnp_random_graph(NUM_NODES, p=0.15, seed=GRAPH_SEED)
_r = np.random.default_rng(GRAPH_SEED)
EDGES = [(i, j, float(_r.uniform(0.5, 2.0))) for i, j in _G.edges()]
TW = sum(w for _, _, w in EDGES)

def maxcut_cost(v, edges=EDGES, tw=TW):
    return sum(w*(v[i]+v[j]-2*v[i]*v[j]) for i,j,w in edges)/tw

np.random.seed(42)
THETAS = np.random.uniform(0.3, 1.5, NUM_NODES).tolist()

@qfunc
def wi(v: Output[QArray[QBit, NUM_NODES]]) -> None:
    allocate(v);  [RY(THETAS[i], v[i]) for i in range(NUM_NODES)]
@qfunc
def cl(g: CReal, v: QArray[QBit]) -> None: phase(-maxcut_cost(v), g)
@qfunc
def wm(b: CReal, v: QArray[QBit]) -> None:
    for i in range(NUM_NODES):
        RY(THETAS[i], v[i]); RZ(-2*b, v[i]); RY(-THETAS[i], v[i])
@qfunc
def main(params: CArray[CReal, NUM_LAYERS * 2], v: Output[QArray[QBit, NUM_NODES]]) -> None:
    wi(v)
    gs = params[0:NUM_LAYERS]; bs = params[NUM_LAYERS:2*NUM_LAYERS]
    repeat(NUM_LAYERS, lambda i: [cl(gs[i], v), wm(bs[i], v)])

def build():
    qprog = synthesize(main)
    kernel = qprog_to_cudaq_kernel(qprog)
    H = None
    for i,j,w in EDGES:
        t = (w/TW)*0.5*(spin.z(i)*spin.z(j) - spin.i(0))
        H = t if H is None else H+t
    return kernel, H
