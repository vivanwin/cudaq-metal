"""Install the prebuilt Metal NVQIR backend into the active CUDA-Q.

The bundled ``libnvqir-metal.dylib`` is ABI-matched to the CUDA-Q build it was
compiled against (see COMPATIBLE_CUDAQ). It is dropped next to CUDA-Q's other
``libnvqir-*.dylib`` plugins (``<site-packages>/lib``) together with the target
config (``<site-packages>/targets/metal.yml``), which is exactly where
``cudaq.set_target("metal")`` looks.
"""
from __future__ import annotations

import filecmp
import importlib.util
import os
import platform
import shutil
import warnings

try:
    from importlib.resources import files as _files  # Python 3.9+
except ImportError:  # pragma: no cover
    _files = None

# CUDA-Q version(s) this prebuilt binary is ABI-compatible with (prefix match on
# the version of the binary distribution, e.g. cuda-quantum-cu13 0.16.0).
COMPATIBLE_CUDAQ = ("0.16.",)


def _data_dir() -> str:
    return str(_files("cudaq_metal") / "data")


def _cudaq_plugin_dirs() -> tuple[str, str]:
    """Return (lib_dir, targets_dir) of the installed cudaq, creating targets/."""
    spec = importlib.util.find_spec("cudaq")
    if spec is None or not spec.origin:
        raise RuntimeError(
            "cudaq is not installed in this environment. "
            "Install it first, e.g. `pip install cudaq==0.16.0`."
        )
    # cudaq/__init__.py -> <site-packages>/cudaq -> <site-packages>
    site = os.path.dirname(os.path.dirname(spec.origin))
    lib_dir = os.path.join(site, "lib")
    targets_dir = os.path.join(site, "targets")
    if not os.path.isdir(lib_dir):
        raise RuntimeError(f"could not find CUDA-Q lib dir at {lib_dir}")
    os.makedirs(targets_dir, exist_ok=True)
    return lib_dir, targets_dir


def _require_platform() -> None:
    if platform.system() != "Darwin" or platform.machine() != "arm64":
        raise RuntimeError(
            "cudaq-metal only supports Apple Silicon macOS (arm64). "
            f"Detected {platform.system()}/{platform.machine()}."
        )


def _warn_if_version_mismatch() -> None:
    # IMPORTANT: read the version via importlib.metadata, NOT by `import cudaq`.
    # Importing cudaq here would make it scan (and cache) its target list before
    # we have copied metal.yml — leaving `set_target("metal")` invalid.
    try:
        from importlib.metadata import PackageNotFoundError, version

        ver = ""
        # The actual binary on macOS is cuda-quantum-cu13; `cudaq` is a
        # meta-package whose version can carry extra suffixes (.post1).
        for dist in ("cuda-quantum-cu13", "cudaq", "cuda-quantum"):
            try:
                ver = version(dist)
                break
            except PackageNotFoundError:
                continue
        if ver and not any(ver.startswith(v) for v in COMPATIBLE_CUDAQ):
            warnings.warn(
                f"cudaq-metal was built for CUDA-Q {COMPATIBLE_CUDAQ}, but this "
                f"environment has cudaq {ver}. The Metal backend may fail to "
                "load; rebuild it against your CUDA-Q if so.",
                stacklevel=2,
            )
    except Exception:
        pass


def install(force: bool = False, quiet: bool = True) -> tuple[str, str]:
    """Copy the Metal backend into the active cudaq. Idempotent.

    Returns the (dylib, yml) destination paths. Set ``force=True`` to overwrite
    an existing install (e.g. after upgrading this package).
    """
    _require_platform()
    # Copy the plugin BEFORE anything might import cudaq, so cudaq sees
    # metal.yml when it first scans its targets directory.
    lib_dir, targets_dir = _cudaq_plugin_dirs()  # find_spec: does not import cudaq
    data = _data_dir()
    src_dylib = os.path.join(data, "libnvqir-metal.dylib")
    src_yml = os.path.join(data, "metal.yml")
    dst_dylib = os.path.join(lib_dir, "libnvqir-metal.dylib")
    dst_yml = os.path.join(targets_dir, "metal.yml")

    for src, dst in ((src_dylib, dst_dylib), (src_yml, dst_yml)):
        # Compare contents, not just size: an upgraded binary can have the
        # same size as the old one. Both files are small, so this is cheap.
        if force or not os.path.exists(dst) or not filecmp.cmp(src, dst, shallow=False):
            shutil.copyfile(src, dst)
            if src.endswith(".dylib"):
                os.chmod(dst, 0o755)
            if not quiet:
                print(f"cudaq-metal: installed {dst}")
    _warn_if_version_mismatch()  # after copying, so cudaq's scan already sees us
    return dst_dylib, dst_yml


def is_installed() -> bool:
    try:
        lib_dir, targets_dir = _cudaq_plugin_dirs()
    except RuntimeError:
        return False
    return os.path.exists(os.path.join(lib_dir, "libnvqir-metal.dylib")) and os.path.exists(
        os.path.join(targets_dir, "metal.yml")
    )


def _main() -> None:
    """Console entry point: `cudaq-metal-install`."""
    dylib, yml = install(force=True, quiet=False)
    print(f"cudaq-metal ready. Use cudaq.set_target('metal').\n  {dylib}\n  {yml}")
