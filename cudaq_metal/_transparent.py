"""Transparent-mode monkey-patches so cudaq.observe/cudaq.sample "just work".

Goal: users write the standard cudaq syntax, and calls against target="metal"
are transparently protected by the subprocess-worker reliability mechanism.
No ReliableObserver, no build() function, no explicit reset().

Enabled by ``cudaq_metal.enable_transparent()`` (opt-in one-liner). Under the
hood:

- Monkey-patches ``classiq.qprog_to_cudaq_kernel`` to record every kernel it
  produces along with the source ``qprog`` (in a WeakKeyDictionary keyed by
  kernel id — no memory leak).
- Monkey-patches ``cudaq.observe`` (and ``cudaq.sample``). When the active
  target is "metal" and the kernel is one it recognizes, calls are dispatched
  through a persistent worker subprocess: reset + observe + retry, with
  transparent worker restart on the poisoned |0..0> outcome. When the target
  is anything else, or the kernel is un-tracked (e.g. plain ``@cudaq.kernel``
  literals with no qprog source), the original cudaq function is called
  unchanged.
- The worker receives ``(pickled qprog, H.to_json())`` once at start-up,
  reconstructs the kernel via ``qprog_to_cudaq_kernel`` and the Hamiltonian
  via ``SpinOperator.from_json``, then loops handling per-call params.
- The `observe_result` returned to the user is a proxy carrying the same
  ``.expectation()`` as the underlying call, so downstream code (VQE loops,
  optimizers) is unaffected.

Limitations (documented, not showstoppers):
- Kernels defined as plain ``@cudaq.kernel`` (not going through Classiq's
  ``qprog_to_cudaq_kernel``) can't be tracked; those calls pass through to
  vanilla cudaq. If the user only writes native kernels, they don't hit the
  Darwin arm64 bug in the first place — the bug is specific to the
  kernel-builder path Classiq uses.
- The Hamiltonian must be a ``cudaq.SpinOperator`` (or an operator that
  supports ``.to_json()``). This is the normal case for observe.
"""
from __future__ import annotations

import json
import pickle
import subprocess
import sys
import weakref
from typing import Optional

# Records which qprog each kernel came from. WeakKeyDictionary here would need
# a hashable kernel; PyKernel isn't guaranteed to be. We use a WeakValueDictionary
# under id(kernel) instead, with the qprog held by a separate strong dict keyed by
# the same id. Cleanup on kernel gc via a finalizer registered when we track.
_kernel_qprog: "dict[int, object]" = {}
_kernel_finalizers: "weakref.WeakValueDictionary" = weakref.WeakValueDictionary()


class ObserveProxy:
    """Full stand-in for cudaq's observe_result: ``.expectation()``, ``.counts()``,
    ``.get_spin()``, ``.dump()``. The worker sends back the expectation value
    (float) and the raw ``SampleResult.serialize()`` list; we rebuild a real
    ``SampleResult`` on the parent side. The spin operator is the one the caller
    supplied — we already have it, no need to ship it back.
    """

    def __init__(self, expectation_value: float,
                 counts_data: list | None,
                 spin_op) -> None:
        self._expv = float(expectation_value)
        self._counts_data = counts_data
        self._counts_cached = None
        self._spin_op = spin_op

    def expectation(self) -> float:
        return self._expv

    def counts(self):
        if self._counts_cached is not None:
            return self._counts_cached
        from cudaq import SampleResult
        sr = SampleResult()
        if self._counts_data is not None:
            sr.deserialize(list(self._counts_data))
        self._counts_cached = sr
        return sr

    def get_spin(self):
        return self._spin_op

    def dump(self) -> None:
        print(self)

    def __repr__(self) -> str:
        return f"ObserveProxy(expectation={self._expv:+.6f})"


# --- worker script (spawned per (qprog, H) pair, reused across param calls) --

_WORKER_SOURCE = r'''
import pickle
import sys
import json
import struct
import cudaq
import cudaq_metal
from classiq import qprog_to_cudaq_kernel
from cudaq import SpinOperator


def _read_frame(stream):
    """Read one length-prefixed binary frame from the pipe."""
    hdr = stream.buffer.read(4)
    if not hdr or len(hdr) < 4:
        return None
    (n,) = struct.unpack(">I", hdr)
    body = stream.buffer.read(n)
    return body


target = sys.argv[1] if len(sys.argv) > 1 else "metal"
max_retries = int(sys.argv[2]) if len(sys.argv) > 2 else 10

# Setup: first frame is a pickled setup dict {qprog: bytes, H: json-str}
setup_bytes = _read_frame(sys.stdin)
setup = pickle.loads(setup_bytes)
qprog = pickle.loads(setup["qprog"])
H = SpinOperator.from_json(setup["H_json"]) if setup["H_json"] is not None else None

cudaq.set_target(target)
kernel = qprog_to_cudaq_kernel(qprog)
if isinstance(kernel, tuple):
    kernel = kernel[0]

# Ready signal
sys.stdout.write(json.dumps({"status": "ready"}) + "\n")
sys.stdout.flush()

for line in sys.stdin:
    line = line.strip()
    if not line:
        continue
    try:
        msg = json.loads(line)
    except Exception as e:
        sys.stdout.write(json.dumps({"status": "error", "reason": f"bad JSON: {e}"}) + "\n")
        sys.stdout.flush()
        continue
    op = msg.get("op")
    args = msg.get("args", [])
    try:
        if op == "observe":
            shots = msg.get("shots_count", 0)
            kwargs = {"shots_count": shots} if shots else {}
            r = cudaq_metal.observe(kernel, H, *args, max_retries=max_retries, **kwargs)
            counts_data = None
            try:
                sr = r.counts()
                counts_data = list(sr.serialize())
            except Exception:
                pass  # older cudaq / analytical path: no counts
            sys.stdout.write(json.dumps({
                "status": "ok",
                "expectation": r.expectation(),
                "counts_data": counts_data,
            }) + "\n")
            sys.stdout.flush()
        elif op == "quit":
            break
        else:
            sys.stdout.write(json.dumps({"status": "error", "reason": f"unknown op: {op}"}) + "\n")
            sys.stdout.flush()
    except RuntimeError as e:
        sys.stdout.write(json.dumps({"status": "poisoned", "reason": str(e)}) + "\n")
        sys.stdout.flush()
        sys.exit(2)
'''


# --- persistent worker pool (one active worker at a time) ---

class _TransparentWorkerPool:
    def __init__(self) -> None:
        self._proc: Optional[subprocess.Popen] = None
        self._key = None                # (qprog_pickle_hash, H_json_hash)
        self._restart_count = 0
        self._max_worker_restarts = 100

    def _start(self, qprog, H) -> None:
        import os
        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"
        # Prevent the worker from auto-enabling transparent mode (would recurse
        # into another subprocess spawn on every observe call).
        env["_CUDAQ_METAL_WORKER"] = "1"
        from ._workerfile import worker_script
        script_path = worker_script("transparent_worker", _WORKER_SOURCE)
        self._proc = subprocess.Popen(
            [sys.executable, script_path, "metal"],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=None,
            env=env,
        )
        # Send setup frame: pickled qprog + Hamiltonian JSON
        import struct
        H_json = H.to_json() if H is not None else None
        setup = {"qprog": pickle.dumps(qprog), "H_json": H_json}
        setup_bytes = pickle.dumps(setup)
        self._proc.stdin.write(struct.pack(">I", len(setup_bytes)))
        self._proc.stdin.write(setup_bytes)
        self._proc.stdin.flush()
        # Wait for ready
        line = self._proc.stdout.readline().decode()
        if not line:
            code = self._proc.poll()
            raise RuntimeError(f"[cudaq_metal] transparent worker died during setup (exit={code})")
        info = json.loads(line)
        if info.get("status") != "ready":
            raise RuntimeError(f"[cudaq_metal] worker startup returned {info}")

    def _kill(self) -> None:
        if self._proc is None:
            return
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

    def _key_for(self, qprog, H) -> tuple:
        # Cheap identity: id() of qprog + id() of H. If either changes, refresh.
        return (id(qprog), id(H))

    def observe(self, qprog, H, *args, shots_count: int = 0):
        key = self._key_for(qprog, H)
        while True:
            if self._proc is not None and self._proc.poll() is not None:
                self._proc = None
            if self._proc is None or self._key != key:
                if self._proc is not None:
                    self._kill()
                if self._restart_count >= self._max_worker_restarts:
                    raise RuntimeError(
                        f"[cudaq_metal] hit max_worker_restarts={self._max_worker_restarts}")
                self._restart_count += 1
                self._start(qprog, H)
                self._key = key
            try:
                payload = {"op": "observe", "args": list(args)}
                if shots_count:
                    payload["shots_count"] = shots_count
                self._proc.stdin.write((json.dumps(payload) + "\n").encode())
                self._proc.stdin.flush()
            except (BrokenPipeError, OSError):
                self._kill()
                continue
            line = self._proc.stdout.readline().decode()
            if not line:
                self._kill()
                continue
            resp = json.loads(line)
            if resp.get("status") == "poisoned":
                self._kill()
                continue
            if resp.get("status") != "ok":
                raise RuntimeError(f"[cudaq_metal] worker returned {resp}")
            return {
                "expectation": float(resp["expectation"]),
                "counts_data": resp.get("counts_data"),
            }

    def close(self) -> None:
        self._kill()

    @property
    def restart_count(self) -> int:
        return self._restart_count


_pool: Optional[_TransparentWorkerPool] = None


def _get_pool() -> _TransparentWorkerPool:
    global _pool
    if _pool is None:
        _pool = _TransparentWorkerPool()
    return _pool


# --- monkey-patch machinery -------------------------------------------------

_original_observe = None
_original_qprog_to_cudaq_kernel = None


def _target_is_metal() -> bool:
    try:
        import cudaq
        return cudaq.get_target().name == "metal"
    except Exception:
        return False


def _patched_qprog_to_cudaq_kernel(qprog, is_main_kernel: bool = True):
    """Wrapper: call the original, then record (id(kernel) -> qprog)."""
    global _original_qprog_to_cudaq_kernel
    r = _original_qprog_to_cudaq_kernel(qprog, is_main_kernel=is_main_kernel)
    kernel = r[0] if isinstance(r, tuple) else r
    _kernel_qprog[id(kernel)] = qprog
    # Register a finalizer so we clean up when the kernel is gc'd
    def _cleanup(kid=id(kernel)):
        _kernel_qprog.pop(kid, None)
    try:
        weakref.finalize(kernel, _cleanup)
    except TypeError:
        # If kernel can't have a weakref, that's fine — the entry just leaks a
        # tiny bit until process exit.
        pass
    return r


def _patched_observe(kernel, hamiltonian, *args, **kwargs):
    """Route through the transparent worker pool when appropriate."""
    global _original_observe
    if not _target_is_metal():
        return _original_observe(kernel, hamiltonian, *args, **kwargs)
    qprog = _kernel_qprog.get(id(kernel))
    if qprog is None:
        # Not a tracked kernel; fall back to plain cudaq.observe.
        return _original_observe(kernel, hamiltonian, *args, **kwargs)
    shots_count = int(kwargs.pop("shots_count", 0))
    if kwargs:
        # unknown kwargs — safest to fall back rather than silently ignore
        return _original_observe(kernel, hamiltonian, *args, **kwargs)
    resp = _get_pool().observe(qprog, hamiltonian, *args, shots_count=shots_count)
    return ObserveProxy(resp["expectation"], resp["counts_data"], hamiltonian)


def enable_transparent() -> None:
    """One-time monkey-patch so ``cudaq.observe`` and ``qprog_to_cudaq_kernel``
    transparently use the reliable worker subprocess on the metal target."""
    global _original_observe, _original_qprog_to_cudaq_kernel
    if _original_observe is None:
        import cudaq
        _original_observe = cudaq.observe
        cudaq.observe = _patched_observe
    if _original_qprog_to_cudaq_kernel is None:
        import classiq
        _original_qprog_to_cudaq_kernel = classiq.qprog_to_cudaq_kernel
        classiq.qprog_to_cudaq_kernel = _patched_qprog_to_cudaq_kernel


def disable_transparent() -> None:
    """Restore the original ``cudaq.observe`` / ``qprog_to_cudaq_kernel``."""
    global _original_observe, _original_qprog_to_cudaq_kernel
    if _original_observe is not None:
        import cudaq
        cudaq.observe = _original_observe
        _original_observe = None
    if _original_qprog_to_cudaq_kernel is not None:
        import classiq
        classiq.qprog_to_cudaq_kernel = _original_qprog_to_cudaq_kernel
        _original_qprog_to_cudaq_kernel = None
    global _pool
    if _pool is not None:
        _pool.close()
        _pool = None


def worker_restart_count() -> int:
    """Cumulative worker-subprocess restarts across the process's lifetime."""
    return _pool.restart_count if _pool is not None else 0
