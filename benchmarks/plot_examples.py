"""Plot results/examples.csv -> results/examples.png and examples-dark.png.

One colour per target; the shallowest depth is drawn solid with filled
markers, deeper circuits dashed with hollow markers.
"""
import csv
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import FixedLocator, NullLocator  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
CSV = os.path.join(HERE, "results", "examples.csv")

THEMES = {
    "examples.png": dict(surface="#fcfcfb", ink="#0b0b0b", ink2="#52514e", muted="#898781",
                         grid="#e1e0d9", axis="#c3c2b7", metal="#2a78d6", qpp="#eb6834"),
    "examples-dark.png": dict(surface="#1a1a19", ink="#f0efec", ink2="#c3c2b7", muted="#898781",
                              grid="#2c2c2a", axis="#383835", metal="#3987e5", qpp="#d95926"),
}
# target -> (label, color key)
SERIES = {"metal": ("metal (Apple GPU)", "metal"), "qpp-cpu": ("qpp-cpu (CPU)", "qpp")}
DASHED = (0, (4, 2.5))
TICKS = [1e-3, 1e-2, 1e-1, 1, 10, 100]
TICK_LABELS = ["1 ms", "10 ms", "100 ms", "1 s", "10 s", "100 s"]


def load():
    """(target, depth) -> {n: median seconds}."""
    data = {}
    with open(CSV) as f:
        for row in csv.DictReader(f):
            key = (row["target"], int(row["depth"]))
            data.setdefault(key, {})[int(row["n"])] = float(row["median"])
    return data


def plot(data, out, t):
    # Per-glyph fallback: DejaVu Sans supplies glyphs Helvetica lacks.
    plt.rcParams["font.family"] = ["Helvetica Neue", "DejaVu Sans"]
    fig, ax = plt.subplots(figsize=(8, 5.4), dpi=200)
    fig.patch.set_facecolor(t["surface"])
    ax.set_facecolor(t["surface"])

    depths = sorted({d for _, d in data})
    for target, (label, color) in SERIES.items():
        for depth in depths:
            pts = sorted(data.get((target, depth), {}).items())
            if not pts:
                continue
            solid = depth == depths[0]
            ax.plot([n for n, _ in pts], [s for _, s in pts], color=t[color],
                    linewidth=1.5, linestyle="-" if solid else DASHED,
                    solid_capstyle="round", dash_capstyle="round",
                    marker="o", markersize=5.5 if solid else 5,
                    markerfacecolor=t[color] if solid else t["surface"],
                    markeredgewidth=1.1 if solid else 1.3,
                    markeredgecolor=t["surface"] if solid else t[color], zorder=3)
            n, s = pts[-1]  # direct label at the line's end
            ax.annotate(f"{label.split(' (')[0]}, {depth} layers", (n, s), xytext=(8, 0),
                        textcoords="offset points", va="center", fontsize=9.5,
                        color=t["ink"], fontweight="bold" if solid else "normal")

    # The headline: metal's speedup at the largest size both engines ran, one
    # bracket per depth in that depth's line style. The shallow bracket sits
    # just left of n with its label on the left, the deep one just right.
    speedup_ns = set()
    for i, depth in enumerate(depths[:2]):
        gpu_pts, cpu_pts = data.get(("metal", depth), {}), data.get(("qpp-cpu", depth), {})
        common = set(gpu_pts) & set(cpu_pts)
        if not common:
            continue
        n = max(common)
        speedup_ns.add(n)
        cpu, gpu = cpu_pts[n], gpu_pts[n]
        x, side = (n - 0.15, -1) if i == 0 else (n + 0.15, 1)
        ax.plot([x, x], [gpu * 1.35, cpu / 1.35], color=t["muted"], linewidth=0.9,
                linestyle="-" if i == 0 else DASHED, zorder=2)
        ax.annotate(f"{cpu / gpu:,.0f}×", (x, (cpu * gpu) ** 0.5), xytext=(5 * side, 0),
                    textcoords="offset points", ha="left" if side > 0 else "right",
                    va="center", fontsize=9.5, color=t["ink2"], fontweight="bold")

    ax.set_yscale("log")
    ax.yaxis.set_major_locator(FixedLocator(TICKS))
    ax.yaxis.set_ticklabels(TICK_LABELS)
    ax.yaxis.set_minor_locator(NullLocator())
    ns = sorted({n for d in data.values() for n in d})
    ax.set_xticks(range(min(ns), max(ns) + 1, 2))
    ax.set_xlim(min(ns) - 0.5, max(ns) + 5.5)
    ax.set_xlabel("Qubits", color=t["ink2"], fontsize=10)
    ax.set_ylabel("Time per cudaq.observe()", color=t["ink2"], fontsize=10)
    ax.tick_params(colors=t["muted"], labelcolor=t["ink2"], labelsize=9, length=0)
    ax.grid(True, which="major", color=t["grid"], linewidth=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(t["axis"])

    # Each layer is one RY per qubit plus ~1.5 brickwork gates per qubit.
    def layers(d):
        return f"{d} layers (~{2.4 * d:.0f} gates per qubit)"

    key = f"Solid: {layers(depths[0])}"
    if len(depths) > 1:
        key += " · dashed: " + ", ".join(layers(d) for d in depths[1:])
    if len(speedup_ns) == 1:
        key += f" · × = metal's speedup at n={speedup_ns.pop()}"
    fig.text(0.075, 0.95, "Classiq circuit on Apple GPU vs CPU",
             fontsize=13, fontweight="bold", color=t["ink"])
    fig.text(0.075, 0.91, "Hardware-efficient ansatz (RY + CX·RZ·CX brickwork) via "
             r"qprog_to_cudaq_kernel · observe $\langle \Sigma_i\, Z_i \rangle$ · median of ≥5 runs",
             fontsize=9, color=t["ink2"])
    fig.text(0.075, 0.875, key, fontsize=9, color=t["ink2"])

    fig.subplots_adjust(left=0.1, right=0.97, top=0.83, bottom=0.1)
    fig.savefig(os.path.join(HERE, "results", out), facecolor=t["surface"])
    plt.close(fig)
    print(f"wrote results/{out}")


if __name__ == "__main__":
    data = load()
    for out, theme in THEMES.items():
        plot(data, out, theme)
