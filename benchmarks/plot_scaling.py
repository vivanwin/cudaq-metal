"""Generate the scaling benchmark plot from measured data.

Reads benchmarks/results/scaling.csv (columns: n, depth, and mean/std for each
engine) and writes benchmarks/results/scaling.png. The README embeds that PNG.

Usage:  python benchmarks/plot_scaling.py
Regenerate the plot after re-running the benchmark and updating the CSV.
"""
import csv
import os
import sys

import matplotlib
matplotlib.use("Agg")   # headless — no GUI backend needed
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(HERE, "results", "scaling.csv")
PNG_PATH = os.path.join(HERE, "results", "scaling.png")


def read_csv(path):
    rows = []
    with open(path) as f:
        for row in csv.DictReader(f):
            def f_or_none(k):
                v = row.get(k, "")
                return float(v) if v not in ("", None) else None
            rows.append({
                "n": int(row["n"]),
                "classiq": (f_or_none("classiq_mean"), f_or_none("classiq_std")),
                "qpp": (f_or_none("qpp_cpu_mean"), f_or_none("qpp_cpu_std")),
                "metal": (f_or_none("metal_transparent_mean"),
                          f_or_none("metal_transparent_std")),
            })
    return rows


def plot(rows, out_path):
    fig, ax = plt.subplots(figsize=(8, 5))

    def series(key, label, color, marker):
        ns, ms, ss = [], [], []
        for r in rows:
            m, s = r[key]
            if m is None: continue
            ns.append(r["n"]); ms.append(m); ss.append(s or 0)
        ax.errorbar(ns, ms, yerr=ss, label=label, color=color,
                    marker=marker, linewidth=1.8, markersize=6, capsize=3)

    series("qpp",     "cudaq --target qpp-cpu",             "#D95E4A", "^")
    series("metal",   "cudaq --target metal (this repo)",   "#2A78D6", "o")

    ax.set_xlabel("Qubits (n)")
    ax.set_ylabel("30-iter COBYLA loop, wall time (s)")
    ax.set_yscale("log")
    ax.set_title("Warm-start QAOA — CUDA-Q Metal (Apple GPU) vs qpp-cpu")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(loc="lower right", framealpha=0.95)

    # Annotate: qpp-cpu ran out of budget past n=17
    ax.annotate("qpp-cpu: not\nfeasible past n=17",
                xy=(17, 110.65), xytext=(15.5, 260),
                fontsize=9, color="#993C1D",
                arrowprops=dict(arrowstyle="->", color="#993C1D", lw=1))

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"wrote {out_path}")


if __name__ == "__main__":
    rows = read_csv(CSV_PATH)
    plot(rows, PNG_PATH)
