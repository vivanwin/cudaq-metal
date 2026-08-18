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
    """Detect the silent-|0..0> outcome in an observe_result.

    A healthy observe against a zero-diagonal Hamiltonian (the standard case
    after identity offset) returns some non-zero expectation. The untouched
    |0..0> state gives EXACTLY 0.0. That's the unambiguous retry signal.

    IMPORTANT: sample_result also has .expectation(), but for a peaked
    distribution it can legitimately be near-zero. Use this on observe results
    only — the wrapper `sample()` deliberately skips this check.
    """
    try:
        return abs(result.expectation()) < 1e-12
    except AttributeError:
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
    """cudaq.sample with a reset() call. Retry-detection is unsafe for sample:
    a peaked healthy distribution can concentrate on |0..0> too, and there is
    no cheap way to distinguish that from the deferred-exception failure. So
    we clear any lingering deferred state and pass through.

    For robust sample-based work, use `ReliableObserver.sample` — its worker
    restarts on observe-driven failures, indirectly protecting sample too.
    """
    import cudaq
    reset()
    return cudaq.sample(kernel, *args, **kwargs)
