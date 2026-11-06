"""Plot results/examples.csv -> results/examples.png and examples-dark.png."""
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
# target -> (legend label, color key)
SERIES = {"metal": ("metal (Apple GPU)", "metal"), "qpp-cpu": ("qpp-cpu (CPU)", "qpp")}
TICKS = [1e-3, 1e-2, 1e-1, 1, 10]
TICK_LABELS = ["1 ms", "10 ms", "100 ms", "1 s", "10 s"]


def load():
    data, depths = {k: {} for k in SERIES}, set()
    with open(CSV) as f:
        for row in csv.DictReader(f):
            data[row["target"]][int(row["n"])] = float(row["mean"])
            depths.add(row["depth"])
    return data, "/".join(sorted(depths))


def plot(data, depth, out, t):
    # Per-glyph fallback: DejaVu Sans supplies glyphs Helvetica lacks.
    plt.rcParams["font.family"] = ["Helvetica Neue", "DejaVu Sans"]
    fig, ax = plt.subplots(figsize=(8, 5), dpi=200)
    fig.patch.set_facecolor(t["surface"])
    ax.set_facecolor(t["surface"])

    for target, (label, color) in SERIES.items():
        pts = sorted(data[target].items())
        ax.plot([n for n, _ in pts], [s for _, s in pts], label=label, color=t[color],
                linewidth=1.5, solid_capstyle="round", marker="o", markersize=5.5,
                markeredgewidth=1.1, markeredgecolor=t["surface"], zorder=3)
        n, s = pts[-1]  # direct label at the line's end
        ax.annotate(label.split(" (")[0], (n, s), xytext=(8, 0), textcoords="offset points",
                    va="center", fontsize=9.5, color=t["ink"], fontweight="bold")

    # The headline: metal's speedup at the largest size both engines ran.
    common = set(data["metal"]) & set(data["qpp-cpu"])
    if common:
        n = max(common)
        cpu, gpu = data["qpp-cpu"][n], data["metal"][n]
        ax.plot([n, n], [gpu * 1.4, cpu / 1.4], color=t["muted"], linewidth=0.8, zorder=2)
        ax.annotate(f"{cpu / gpu:,.0f}× faster\nat n={n}", (n, (cpu * gpu) ** 0.5),
                    xytext=(8, 0), textcoords="offset points", ha="left", va="center",
                    fontsize=9, color=t["ink2"])

    ax.set_yscale("log")
    ax.yaxis.set_major_locator(FixedLocator(TICKS))
    ax.yaxis.set_ticklabels(TICK_LABELS)
    ax.yaxis.set_minor_locator(NullLocator())
    ns = sorted({n for d in data.values() for n in d})
    ax.set_xticks(range(min(ns), max(ns) + 1, 2))
    ax.set_xlim(min(ns) - 0.5, max(ns) + 3)
    ax.set_xlabel("Qubits", color=t["ink2"], fontsize=10)
    ax.set_ylabel("Time per cudaq.observe()", color=t["ink2"], fontsize=10)
    ax.tick_params(colors=t["muted"], labelcolor=t["ink2"], labelsize=9, length=0)
    ax.grid(True, which="major", color=t["grid"], linewidth=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(t["axis"])

    fig.text(0.075, 0.95, "Classiq circuit on Apple GPU vs CPU",
             fontsize=13, fontweight="bold", color=t["ink"])
    fig.text(0.075, 0.905, f"Hardware-efficient ansatz (RY + CX·RZ·CX brickwork, depth {depth}) "
             r"via qprog_to_cudaq_kernel · observe $\langle \Sigma_i\, Z_i \rangle$ · mean of 3 runs",
             fontsize=9, color=t["ink2"])
    leg = ax.legend(loc="upper left", frameon=False, fontsize=9, handlelength=2.2)
    for txt in leg.get_texts():
        txt.set_color(t["ink2"])

    fig.subplots_adjust(left=0.1, right=0.97, top=0.85, bottom=0.11)
    fig.savefig(os.path.join(HERE, "results", out), facecolor=t["surface"])
    plt.close(fig)
    print(f"wrote results/{out}")


if __name__ == "__main__":
    data, depth = load()
    for out, theme in THEMES.items():
        plot(data, depth, out, theme)
