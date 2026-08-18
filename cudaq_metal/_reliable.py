"""Reliable observe/sample via a persistent worker subprocess.

Motivation: on Darwin arm64, each Python process's cudaq JIT state is
independently either "healthy" (all observes work) or "poisoned" (silent
|0..0> on every call, no in-process recovery). We wrap each user session in
a worker subprocess: as long as the worker's healthy we reuse it for many
observes with no startup overhead; the moment its retries give up we kill it
and spawn a fresh one. Net effect for the caller: unlimited observe/sample
calls that reliably return real results.

Usage:

    # user_script.py (contains the model + hamiltonian factory)
    def build():
        # ... your Classiq synthesis + qprog_to_cudaq_kernel + Hamiltonian construction ...
        return kernel, H

    # driver:
    import cudaq_metal
    obs = cudaq_metal.ReliableObserver("user_script", "build")
    for params in optimizer.iterate():
        expv = obs.observe(*params)   # <-- always returns a real value or raises with detail
    obs.close()
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from typing import Optional


# The worker script we spawn. Reads (op, params) from stdin as JSON lines and
# writes back JSON lines. Exits with code 2 on "poisoned" state (parent sees
# EOF, decides to restart).
_WORKER_SOURCE = r'''
import importlib
import json
import sys
import cudaq
import cudaq_metal

module_name = sys.argv[1]
build_name = sys.argv[2]
target = sys.argv[3] if len(sys.argv) > 3 else "metal"
max_in_process_retries = int(sys.argv[4]) if len(sys.argv) > 4 else 10

mod = importlib.import_module(module_name)
build = getattr(mod, build_name)

cudaq.set_target(target)
kernel, hamiltonian = build()

print(json.dumps({"status": "ready"}), flush=True)

for raw in sys.stdin:
    raw = raw.strip()
    if not raw:
        continue
    try:
        msg = json.loads(raw)
    except Exception as e:
        print(json.dumps({"status": "error", "reason": f"bad JSON: {e}"}), flush=True)
        continue
    op = msg.get("op")
    args = msg.get("args", [])
    try:
        if op == "observe":
            r = cudaq_metal.observe(kernel, hamiltonian, *args,
                                    max_retries=max_in_process_retries)
            print(json.dumps({"status": "ok", "expectation": r.expectation()}), flush=True)
        elif op == "sample":
            shots = msg.get("shots", 1000)
            r = cudaq_metal.sample(kernel, *args, shots_count=shots,
                                   max_retries=max_in_process_retries)
            print(json.dumps({"status": "ok",
                              "counts": {k: v for k, v in r.items()}}), flush=True)
        elif op == "quit":
            break
        else:
            print(json.dumps({"status": "error", "reason": f"unknown op: {op}"}), flush=True)
    except RuntimeError as e:
        # cudaq_metal gave up: this process is poisoned. Signal parent to restart.
        print(json.dumps({"status": "poisoned", "reason": str(e)}), flush=True)
        sys.exit(2)
'''


class WorkerRestartLimitExceeded(RuntimeError):
    """The subprocess was restarted more than `max_worker_restarts` times."""


class ReliableObserver:
    """Runs cudaq.observe/sample against Metal with unbounded reliability.

    The wrapped kernel/Hamiltonian are built inside a worker subprocess (fresh
    JIT state per worker). When the worker becomes poisoned (its retry-loop
    exhausts), we kill it and spawn a new one transparently. Every call
    eventually returns a real result — no silent |0..0>.

    Args:
        build_module: Python module (importable in the target env) that defines
            a `build_func()` returning ``(kernel, hamiltonian)``.
        build_func: Function name in that module. Default ``"build"``.
        target: cudaq target name. Default ``"metal"``.
        python: Path to the python interpreter for the worker (defaults to the
            interpreter running the parent). Point this at your project's venv
            if you're driving from a different one.
        max_in_process_retries: Retries per call inside the worker before it
            declares itself poisoned. Default 10.
        max_worker_restarts: Global cap on how many times we're willing to
            restart the worker over the observer's lifetime. Default 50; raise
            it for long optimizations, or set to a very large number.
    """

    def __init__(
        self,
        build_module: str,
        build_func: str = "build",
        target: str = "metal",
        python: Optional[str] = None,
        max_in_process_retries: int = 10,
        max_worker_restarts: int = 50,
        extra_pythonpath: Optional[list] = None,
    ) -> None:
        self.build_module = build_module
        self.build_func = build_func
        self.target = target
        self.python = python or sys.executable
        self.max_in_process_retries = max_in_process_retries
        self.max_worker_restarts = max_worker_restarts
        # Propagate parent's sys.path plus extras so the worker can import build_module.
        self.extra_pythonpath = list(extra_pythonpath) if extra_pythonpath else list(sys.path)
        self._proc: Optional[subprocess.Popen] = None
        self._restart_count = 0

    # --- worker lifecycle ---
    def _start(self) -> None:
        # Write the worker source to a temp file so we can pass it to python
        # cleanly (piping via -c makes stdin/stdout awkward).
        env = os.environ.copy()
        # Keep worker output line-buffered so we can readline in the parent.
        env["PYTHONUNBUFFERED"] = "1"
        # Merge extras into PYTHONPATH so the worker can import build_module
        # from wherever the caller has it (default: parent's sys.path).
        existing_pp = env.get("PYTHONPATH", "")
        merged = list(self.extra_pythonpath) + ([existing_pp] if existing_pp else [])
        env["PYTHONPATH"] = os.pathsep.join(p for p in merged if p)
        script_path = os.path.join(os.path.dirname(__file__), "_worker.py")
        if not os.path.exists(script_path):
            with open(script_path, "w") as f:
                f.write(_WORKER_SOURCE)
        self._proc = subprocess.Popen(
            [self.python, script_path, self.build_module, self.build_func,
             self.target, str(self.max_in_process_retries)],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=None,
            text=True, env=env,
        )
        # Wait for ready signal (or death)
        line = self._proc.stdout.readline()
        if not line:
            code = self._proc.poll()
            raise RuntimeError(
                f"[cudaq_metal] worker died during startup (exit={code}); "
                "check your build_module for import errors.")
        info = json.loads(line)
        if info.get("status") != "ready":
            raise RuntimeError(f"[cudaq_metal] unexpected worker startup: {info}")

    def _ensure_worker(self) -> None:
        if self._proc is None or self._proc.poll() is not None:
            if self._restart_count >= self.max_worker_restarts:
                raise WorkerRestartLimitExceeded(
                    f"reached max_worker_restarts={self.max_worker_restarts}. "
                    "Raise it or check for a persistent failure.")
            self._restart_count += 1
            self._start()

    def _kill(self) -> None:
        if self._proc is not None:
            try:
                self._proc.stdin.close()
            except Exception:
                pass
            try:
                self._proc.terminate()
                self._proc.wait(timeout=5)
            except Exception:
                self._proc.kill()
            self._proc = None

    def _rpc(self, msg: dict) -> dict:
        while True:
            self._ensure_worker()
            try:
                self._proc.stdin.write(json.dumps(msg) + "\n")
                self._proc.stdin.flush()
            except (BrokenPipeError, OSError):
                self._kill()
                continue
            line = self._proc.stdout.readline()
            if not line:
                # Worker died mid-call (poisoned or crashed). Kill + restart.
                self._kill()
                continue
            resp = json.loads(line)
            if resp.get("status") == "poisoned":
                self._kill()
                continue
            return resp

    # --- public API ---
    def observe(self, *args) -> float:
        """Reliable cudaq.observe → expectation value."""
        resp = self._rpc({"op": "observe", "args": list(args)})
        if resp.get("status") != "ok":
            raise RuntimeError(f"[cudaq_metal.observe] worker: {resp}")
        return float(resp["expectation"])

    def sample(self, *args, shots: int = 1000) -> dict:
        """cudaq.sample → dict of {bitstring: count}.

        NOT fully reliability-recovering (observe uses a different launcher
        path than sample, and a healthy observe on the same worker doesn't
        guarantee a healthy sample). If you see suspicious all-zero
        distributions from user_qaoa_module-type kernels, verify the worker's
        health with :meth:`health_check` and rerun after any restart.
        For the common VQE/QAOA pattern (many observes, one final sample),
        just re-run the whole driver a couple of times if the sample looks
        wrong — the observe loop's reliability isn't affected.
        """
        resp = self._rpc({"op": "sample", "args": list(args), "shots": shots})
        if resp.get("status") != "ok":
            raise RuntimeError(f"[cudaq_metal.sample] worker: {resp}")
        return resp["counts"]

    def health_check(self, *args) -> bool:
        """Run a probe observe(*args) and return True iff the result is non-zero.

        A legitimately-healthy observe against a zero-diagonal Hamiltonian will
        return a non-zero expectation. A poisoned worker returns EXACTLY 0.0.
        This is a cheap way to verify worker health before critical calls
        (typically before a final sample after convergence).

        Note: the probe uses the SAME hamiltonian your build() returned; args
        are the parameters to pass to observe.
        """
        resp = self._rpc({"op": "observe", "args": list(args)})
        if resp.get("status") != "ok":
            return False
        return abs(float(resp["expectation"])) > 1e-12

    def close(self) -> None:
        if self._proc is not None:
            try:
                self._proc.stdin.write(json.dumps({"op": "quit"}) + "\n")
                self._proc.stdin.flush()
                self._proc.wait(timeout=5)
            except Exception:
                pass
            finally:
                self._kill()

    # context-manager sugar
    def __enter__(self) -> "ReliableObserver":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    @property
    def restart_count(self) -> int:
        """Number of times we've had to restart the worker (poison recovery)."""
        return self._restart_count
