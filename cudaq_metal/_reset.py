"""Runtime workaround: reset cudaq's deferredKernelException from Python.

The Metal backend + cudaq kernel-builder path has a non-deterministic upstream
bug on Darwin arm64 (see BUG_INVESTIGATION.md): a JIT-frame C++ exception is
non-recoverably deferred onto cudaq's ExecutionContext, and every subsequent
sample() reads |0..0>. The fix (from user code) is to clear that deferred state
between observe/sample calls. Since Python is always outside a JIT frame, this
is safe.

Usage::

    for step in optimization_loop:
        cudaq_metal.reset()          # clear any lingering deferred exception
        val = cudaq.observe(kernel, H, *params).expectation()
"""
from __future__ import annotations

import ctypes
import os
import warnings


_cleaner: "ctypes.CFUNCTYPE | None" = None


def _load() -> "ctypes.CFUNCTYPE | None":
    """Locate libnvqir-metal.dylib and bind `metal_clear_deferred_kernel_exception`."""
    global _cleaner
    if _cleaner is not None:
        return _cleaner
    try:
        import cudaq
    except ImportError:
        return None
    # The dylib is next to cudaq's other backend plugins (site-packages/lib).
    site = os.path.dirname(os.path.dirname(os.path.abspath(cudaq.__file__)))
    dylib_path = os.path.join(site, "lib", "libnvqir-metal.dylib")
    if not os.path.exists(dylib_path):
        return None
    try:
        lib = ctypes.CDLL(dylib_path)
        fn = lib.metal_clear_deferred_kernel_exception
        fn.restype = ctypes.c_int
        fn.argtypes = []
        _cleaner = fn
        return fn
    except (OSError, AttributeError):
        return None


def reset() -> bool:
    """Clear any lingering cudaq deferredKernelException. Returns True if it cleared.

    Call this between observe/sample calls in a variational optimization loop
    on Darwin arm64 to avoid the silent |0..0> bug. No-op if the metal backend
    isn't loaded or nothing was deferred.
    """
    fn = _load()
    if fn is None:
        return False
    return bool(fn())


# --- Retry helper -----------------------------------------------------------
# Given the ~1-in-N success rate per observe() call (see BUG_INVESTIGATION.md),
# the practical workaround is: reset state, try, and re-try until the result
# looks legitimate. This wrapper does that around cudaq.observe and cudaq.sample.

def _looks_like_zero_state(result) -> bool:
    """Detect the silent-|0..0> outcome. Both observe and sample results supported."""
    # observe_result: expectation() close to zero AND the underlying sample map
    # concentrates on the all-zeros string (this is the tell — the untouched
    # state gives exactly 0 for any Hamiltonian without an identity offset,
    # and the sampling would show 100% |0..0>).
    try:
        expv = result.expectation()
        return abs(expv) < 1e-12  # untouched state gives EXACTLY 0
    except AttributeError:
        pass
    # sample_result: check that the only nonzero count is the |0..0> string
    try:
        d = {k: v for k, v in result.items()}
        total = sum(d.values())
        zeros_key = "0" * max(len(k) for k in d)
        return total > 0 and d.get(zeros_key, 0) == total
    except Exception:
        return False


def observe(kernel, hamiltonian, *args, max_retries: int = 20):
    """cudaq.observe wrapped in reset+retry-on-failure. Darwin arm64 workaround."""
    import cudaq
    for attempt in range(max_retries + 1):
        reset()
        r = cudaq.observe(kernel, hamiltonian, *args)
        if not _looks_like_zero_state(r):
            return r
    raise RuntimeError(
        f"cudaq_metal.observe: gave up after {max_retries+1} attempts — every "
        "call returned the untouched |0..0> state. This is the upstream "
        "cudaq deferred-exception bug (see BUG_INVESTIGATION.md).")


def sample(kernel, *args, max_retries: int = 20, **kwargs):
    """cudaq.sample wrapped in reset+retry-on-failure. Darwin arm64 workaround."""
    import cudaq
    for attempt in range(max_retries + 1):
        reset()
        r = cudaq.sample(kernel, *args, **kwargs)
        if not _looks_like_zero_state(r):
            return r
    raise RuntimeError(
        f"cudaq_metal.sample: gave up after {max_retries+1} attempts.")
