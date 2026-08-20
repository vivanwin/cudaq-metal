"""Plot benchmarks/results/examples.csv -> examples.png (embedded in README)."""
import csv, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
CSV  = os.path.join(HERE, "results", "examples.csv")
PNG  = os.path.join(HERE, "results", "examples.png")

series = {"native_qpp": {"n": [], "m": [], "s": []},
          "classiq_metal": {"n": [], "m": [], "s": []}}
with open(CSV) as f:
    for row in csv.DictReader(f):
        d = series[row["kind"]]
        d["n"].append(int(row["n"])); d["m"].append(float(row["mean"])); d["s"].append(float(row["std"]))

fig, ax = plt.subplots(figsize=(8, 5))
ax.errorbar(series["native_qpp"]["n"], series["native_qpp"]["m"],
            yerr=series["native_qpp"]["s"], label="native @cudaq.kernel on qpp-cpu",
            color="#D95E4A", marker="^", linewidth=1.8, markersize=6, capsize=3)
ax.errorbar(series["classiq_metal"]["n"], series["classiq_metal"]["m"],
            yerr=series["classiq_metal"]["s"], label="Classiq → cudaq on metal",
            color="#2A78D6", marker="o", linewidth=1.8, markersize=6, capsize=3)
ax.set_xlabel("Qubits (n)")
ax.set_ylabel("Single observe(), wall time (s)")
ax.set_yscale("log")
ax.set_title("Random-SU(4) brickwork (depth=5) — the example scripts scaled")
ax.grid(True, which="both", alpha=0.25)
ax.legend(loc="lower right", framealpha=0.95)
fig.tight_layout(); fig.savefig(PNG, dpi=150, bbox_inches="tight")
print(f"wrote {PNG}")
