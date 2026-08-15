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

__all__ = ["install", "is_installed"]

# Auto-install on import so `cudaq.set_target("metal")` just works. Kept
# best-effort: a failure here (e.g. cudaq not yet installed) is a warning, not
# an error, so importing this package never hard-crashes a project.
try:
    install()
except Exception as _e:  # pragma: no cover
    import warnings

    warnings.warn(f"cudaq-metal: backend not installed automatically: {_e}", stacklevel=2)
