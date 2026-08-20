import csv, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
CSV  = os.path.join(HERE, "results", "examples.csv")
PNG  = os.path.join(HERE, "results", "examples.png")

series = {k: {"n": [], "m": [], "s": []} for k in
          ["native_qpp", "native_metal", "classiq_qpp", "classiq_metal"]}
with open(CSV) as f:
    for row in csv.DictReader(f):
        d = series[row["kind"]]
        d["n"].append(int(row["n"])); d["m"].append(float(row["mean"])); d["s"].append(float(row["std"]))

fig, ax = plt.subplots(figsize=(8.5, 5.2))
styles = {
    "native_qpp":    ("native cudaq  →  qpp-cpu",   "#D95E4A", "^", "-"),
    "native_metal":  ("native cudaq  →  metal",     "#EB6834", "s", "--"),
    "classiq_qpp":   ("Classiq  →  qpp-cpu",       "#8A8A8A", "v", "--"),
    "classiq_metal": ("Classiq  →  metal",          "#2A78D6", "o", "-"),
}
for k, (label, color, marker, ls) in styles.items():
    d = series[k]
    if not d["n"]: continue
    ax.errorbar(d["n"], d["m"], yerr=d["s"], label=label, color=color,
                marker=marker, linestyle=ls, linewidth=1.8, markersize=6, capsize=3)

ax.set_xlabel("Qubits (n)")
ax.set_ylabel("Single observe(), wall time (s)")
ax.set_yscale("log")
ax.set_title("Random-SU(4) brickwork (depth=5) — engine × target matrix")
ax.grid(True, which="both", alpha=0.25)
ax.legend(loc="lower right", framealpha=0.95, fontsize=10)
fig.tight_layout(); fig.savefig(PNG, dpi=150, bbox_inches="tight")
print(f"wrote {PNG}")
