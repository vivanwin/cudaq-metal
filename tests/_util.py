"""Helpers shared by the test modules (imported after cudaq_metal and cudaq)."""
import ctypes
import os

import cudaq
from cudaq_metal._install import _cudaq_plugin_dirs


def on(target, fn, *args, **kwargs):
    """Run fn(*args, **kwargs) with `target` set, then restore the default."""
    cudaq.set_target(target)
    try:
        return fn(*args, **kwargs)
    finally:
        cudaq.reset_target()


def env_flag(name):
    """The backend's reading of an MSV_DISABLE_* variable (msv_env_flag)."""
    return os.environ.get(name, "").strip().lower() not in ("", "0", "false", "no", "off")


def buffer_stats():
    """(fresh statevector allocations, statevectors reused, scratch buffers
    reused) so far in this process, from the loaded plugin's counters."""
    # Same path cudaq loaded the plugin from, so dlopen returns that image.
    lib = ctypes.CDLL(os.path.join(_cudaq_plugin_dirs()["lib"], "libnvqir-metal.dylib"))
    lib.msv_buffer_stats.restype = None
    vals = [ctypes.c_ulonglong() for _ in range(3)]
    lib.msv_buffer_stats(*(ctypes.byref(v) for v in vals))
    return tuple(v.value for v in vals)
