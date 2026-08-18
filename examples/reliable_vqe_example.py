"""Reliable N-iteration observe loop on Metal (Darwin arm64 workaround).

Uses cudaq_metal.ReliableObserver to guarantee every observe() returns a real
result — silent |0..0> failures cause the worker subprocess to be restarted
transparently. See BUG_INVESTIGATION.md for the underlying cudaq bug.

Requires: your build module must live somewhere importable in the *worker's*
Python. Either put it next to this script and pass its path via
`extra_pythonpath=[here]`, or install it as a package.
"""
import os
import sys
import numpy as np

import cudaq_metal

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)  # so the parent can import too


def main():
    np.random.seed(0)
    N = 30
    params_list = [list(np.random.uniform(0.1, 3.14, 6)) for _ in range(N)]

    # Point `python` at whichever venv has cudaq + classiq + cudaq_metal installed.
    with cudaq_metal.ReliableObserver(
        build_module="qaoa_warm_start_module",
        build_func="build",
        target="metal",
        max_worker_restarts=100,
        extra_pythonpath=[HERE],  # so the worker can import our module
    ) as obs:
        for k, p in enumerate(params_list):
            expv = obs.observe(*p)
            print(f"iter {k:2d}: <H>={expv:+.6f}  (worker restarts so far: {obs.restart_count})")
        print(f"\ntotal worker restarts: {obs.restart_count}")


if __name__ == "__main__":
    main()
