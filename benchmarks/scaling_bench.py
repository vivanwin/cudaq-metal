"""Reproducer for benchmarks/results/scaling.csv.

Sweeps a warm-start QAOA optimization across a range of qubit counts, timing
each engine (Classiq cloud, cudaq qpp-cpu, cudaq metal) over `NUM_TRIALS`
30-iter COBYLA runs, and appends mean ± std to the CSV. Uses the transparent
`cudaq.observe(kernel, H, *params).expectation()` syntax — no ReliableObserver.

Run:  python benchmarks/scaling_bench.py --min 13 --max 24 --trials 3

Regenerate the plot afterward:  python benchmarks/plot_scaling.py
"""
import argparse
import os
import statistics
import sys
import time

import numpy as np
import networkx as nx
import scipy.optimize
from functools import reduce
import operator

# The one line that makes cudaq.observe reliable on --target metal:
import cudaq_metal
import cudaq
from classiq import (
    qfunc, CReal, CArray, QBit, QArray, Output, allocate, RY, RZ, phase, repeat,
    synthesize, qprog_to_cudaq_kernel, pauli_operator_to_cudaq_spin_op, Pauli,
    variational_minimize,
)


def build_problem(n_nodes: int, layers: int = 3, seed: int = 42):
    """Weighted-graph MaxCut instance + warm-start-derived THETAS + ansatz."""
    g = nx.gnp_random_graph(n_nodes, p=0.15, seed=seed)
    rng = np.random.default_rng(seed)
    edges = [(i, j, float(rng.uniform(0.5, 2.0))) for i, j in g.edges()]
    total_w = sum(w for _, _, w in edges)

    def maxcut_cost(v, edges=edges, tw=total_w):
        return -sum(w * (v[i] * (1 - v[j]) + v[j] * (1 - v[i]))
                    for (i, j, w) in edges) / tw

    def continuous_cut(c, edges=edges):
        return sum(w * (c[i] + c[j] - 2 * c[i] * c[j]) for (i, j, w) in edges)

    r2 = np.random.default_rng(seed=42)
    best_c, best_val = None, -np.inf
    for _ in range(30):
        x0 = r2.uniform(0.0, 1.0, size=n_nodes)
        result = scipy.optimize.minimize(
            fun=lambda c: -continuous_cut(c), x0=x0,
            bounds=[(0.0, 1.0)] * n_nodes, method="L-BFGS-B")
        if -result.fun > best_val:
            best_val, best_c = -result.fun, result.x
    thetas = (2 * np.arcsin(np.sqrt(np.clip(best_c, 0.15, 0.85)))).tolist()

    @qfunc
    def wi(v: Output[QArray[QBit, n_nodes]]) -> None:
        allocate(v)
        for i in range(n_nodes): RY(thetas[i], v[i])

    @qfunc
    def cl(gamma: CReal, v: QArray[QBit]) -> None:
        phase(-maxcut_cost(v), gamma)

    @qfunc
    def wm(beta: CReal, v: QArray[QBit]) -> None:
        for i in range(n_nodes):
            RY(thetas[i], v[i]); RZ(-2 * beta, v[i]); RY(-thetas[i], v[i])

    @qfunc
    def main(params: CArray[CReal, layers * 2],
             v: Output[QArray[QBit, n_nodes]]) -> None:
        wi(v)
        gs = params[0:layers]; bs = params[layers:2 * layers]
        repeat(layers, lambda i: [cl(gs[i], v), wm(bs[i], v)])

    def ham():
        zz_terms = [(w / (2 * total_w)) * (Pauli.Z(i) * Pauli.Z(j)) for (i, j, w) in edges]
        return reduce(operator.add, zz_terms) + (
            -sum(w for _, _, w in edges) / (2 * total_w)) * Pauli.I(n_nodes - 1)

    return main, ham, maxcut_cost, layers


def _time_run(fn, *args, **kwargs):
    t0 = time.perf_counter(); fn(*args, **kwargs); return time.perf_counter() - t0


def bench_classiq(main, maxcut_cost, layers, max_iter=30):
    qprog = synthesize(main)
    gammas = np.linspace(0.1, np.pi, layers).tolist()
    betas = np.linspace(0.05, 0.3, layers).tolist()
    return _time_run(variational_minimize, qprog=qprog,
                     cost_function=lambda v: maxcut_cost(v),
                     initial_params={"params": gammas + betas},
                     max_iteration=max_iter)


def bench_cudaq(target, main, ham_fn, layers, max_iter=30):
    """cudaq.observe COBYLA loop against the given target — vanilla syntax."""
    cudaq.set_target(target)
    qprog = synthesize(main)
    kernel = qprog_to_cudaq_kernel(qprog)
    H = pauli_operator_to_cudaq_spin_op(ham_fn())
    gammas = np.linspace(0.1, np.pi, layers).tolist()
    betas = np.linspace(0.05, 0.3, layers).tolist()

    def objective(params):
        return cudaq.observe(kernel, H, *params).expectation()

    optimizer = cudaq.optimizers.COBYLA()
    optimizer.initial_parameters = gammas + betas
    optimizer.max_iterations = max_iter
    t0 = time.perf_counter()
    optimizer.optimize(dimensions=len(gammas + betas), function=objective)
    return time.perf_counter() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--min", type=int, default=13)
    ap.add_argument("--max", type=int, default=18)
    ap.add_argument("--trials", type=int, default=3)
    ap.add_argument("--engines", default="qpp-cpu,metal",
                    help="comma-separated: classiq, qpp-cpu, metal")
    ap.add_argument("--qpp-cutoff", type=int, default=17,
                    help="skip qpp-cpu above this qubit count (too slow)")
    ap.add_argument("--out", default=os.path.join(
        os.path.dirname(__file__), "results", "scaling.csv"))
    args = ap.parse_args()

    engines = set(args.engines.split(","))
    header_needed = not os.path.exists(args.out)
    with open(args.out, "a") as f:
        if header_needed:
            f.write("n,depth,classiq_mean,classiq_std,qpp_cpu_mean,qpp_cpu_std,"
                    "metal_reliable_mean,metal_reliable_std,"
                    "metal_transparent_mean,metal_transparent_std\n")

        for n in range(args.min, args.max + 1):
            print(f"\n=== n={n} ===", flush=True)
            main_fn, ham_fn, mc, layers = build_problem(n)
            depth = "?"  # qprog.depth is expensive to compute; leave for post-run
            cell = {}
            for eng in ["classiq", "qpp-cpu", "metal"]:
                if eng not in engines: cell[eng] = (None, None); continue
                if eng == "qpp-cpu" and n > args.qpp_cutoff: cell[eng] = (None, None); continue
                ts = []
                for t in range(args.trials):
                    if eng == "classiq":
                        s = _time_run(bench_classiq, main_fn, mc, layers)
                    else:
                        s = _time_run(bench_cudaq, eng, main_fn, ham_fn, layers)
                    ts.append(s); print(f"  {eng} trial {t}: {s:.1f}s", flush=True)
                cell[eng] = (statistics.mean(ts),
                             statistics.stdev(ts) if len(ts) > 1 else 0.0)
            def fmt(x): return f"{x:.2f}" if x is not None else ""
            f.write(f"{n},{depth},{fmt(cell['classiq'][0])},{fmt(cell['classiq'][1])},"
                    f"{fmt(cell['qpp-cpu'][0])},{fmt(cell['qpp-cpu'][1])},"
                    f",,"          # metal_reliable columns left blank; transparent is what matters
                    f"{fmt(cell['metal'][0])},{fmt(cell['metal'][1])}\n")
            f.flush()
    print(f"\nappended results to {args.out}")


if __name__ == "__main__":
    main()
