"""Smoke test: run GHZ on the Metal backend via Python cudaq, compare to qpp-cpu."""
import sys
import cudaq


@cudaq.kernel
def ghz(n: int):
    q = cudaq.qvector(n)
    h(q[0])
    for i in range(n - 1):
        x.ctrl(q[i], q[i + 1])
    mz(q)


def run(target: str, n: int, shots: int):
    cudaq.set_target(target)
    counts = cudaq.sample(ghz, n, shots_count=shots)
    d = {k: v for k, v in counts.items()}
    cudaq.reset_target()
    return d


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 8
    shots = int(sys.argv[2]) if len(sys.argv) > 2 else 2000
    for tgt in ("qpp-cpu", "metal"):
        print(f"[{tgt}] GHZ({n}) ->", run(tgt, n, shots))
