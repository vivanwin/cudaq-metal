"""cudaq-metal: a naive Metal (Apple GPU) statevector backend for CUDA-Q.

Usage in any project whose environment has a compatible ``cudaq``::

    import cudaq_metal          # installs the backend into the active cudaq
    import cudaq
    cudaq.set_target("metal")
    # ... define @cudaq.kernel and cudaq.sample / cudaq.get_state ...

Importing this package installs the backend automatically (idempotent). You can
also install/refresh it explicitly with ``cudaq_metal.install(force=True)`` or
the ``cudaq-metal-install`` console command.
"""
from ._install import install, is_installed
from ._reset import reset, observe, sample
from ._reliable import ReliableObserver, WorkerRestartLimitExceeded
from ._transparent import enable_transparent, disable_transparent, worker_restart_count

__all__ = ["install", "is_installed", "reset", "observe", "sample", "ReliableObserver", "WorkerRestartLimitExceeded", "enable_transparent", "disable_transparent", "worker_restart_count"]

# Auto-install on import so `cudaq.set_target("metal")` just works. Kept
# best-effort: a failure here (e.g. cudaq not yet installed) is a warning, not
# an error, so importing this package never hard-crashes a project.
try:
    install()
except Exception as _e:  # pragma: no cover
    import warnings

    warnings.warn(f"cudaq-metal: backend not installed automatically: {_e}", stacklevel=2)


# Auto-enable transparent mode. Rationale: users doing `import cudaq_metal;
# cudaq.set_target("metal")` and forgetting the opt-in got silent |0..0>
# zeros ~60% of the time — a nasty footgun. Auto-enable is scoped: only
# metal-target + Classiq-tracked-kernel calls go through the worker; every
# other call (qpp-cpu, nvidia, native @cudaq.kernel) passes through the
# original cudaq.observe unchanged. Users who want the raw observe_result
# object back or the untouched qprog_to_cudaq_kernel can call
# `cudaq_metal.disable_transparent()`.
import os as _os
# Workers set _CUDAQ_METAL_WORKER=1 when they spawn to skip auto-enable — otherwise
# they'd recursively try to route their observe calls to yet another subprocess.
if not _os.environ.get("_CUDAQ_METAL_WORKER"):
    try:
        enable_transparent()
    except Exception as _e:  # pragma: no cover
        import warnings
        warnings.warn(
            f"cudaq-metal: transparent mode not enabled ({_e!r}); "
            "call cudaq_metal.enable_transparent() manually after cudaq/classiq are imported.",
            stacklevel=2,
        )
