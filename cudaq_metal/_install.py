"""Install the prebuilt Metal NVQIR backend into the active CUDA-Q.

The bundled ``libnvqir-metal.dylib`` is ABI-matched to the CUDA-Q build it was
compiled against (see COMPATIBLE_CUDAQ). It is dropped next to CUDA-Q's other
``libnvqir-*.dylib`` plugins (``<site-packages>/lib``) together with the target
config (``<site-packages>/targets/metal.yml``), which is exactly where
``cudaq.set_target("metal")`` looks.

cudaq reads its targets directory once, when it is first imported, so the
files must be in place before that: ``import cudaq_metal`` before ``cudaq``.
"""
from __future__ import annotations

import contextlib
import filecmp
import importlib.util
import os
import platform
import shutil
import sys
import tempfile
import warnings
from importlib.metadata import PackageNotFoundError, version
from importlib.resources import files

# The binary distribution behind `pip install cudaq` on macOS, and the version
# prefix(es) this prebuilt plugin is ABI-compatible with. Bump together with
# the pin in pyproject.toml whenever the binary is rebuilt for a new CUDA-Q.
CUDAQ_DIST = "cuda-quantum-cu13"
COMPATIBLE_CUDAQ = ("0.16.",)

# (bundled file, destination: "lib" or "targets", permissions)
_FILES = (
    ("libnvqir-metal.dylib", "lib", 0o755),
    ("metal.yml", "targets", 0o644),
)


def _cudaq_plugin_dirs() -> dict[str, str]:
    """Return {"lib": ..., "targets": ...} of the installed cudaq."""
    spec = importlib.util.find_spec("cudaq")  # locates cudaq without importing it
    if spec is None or not spec.origin:
        raise RuntimeError(
            "cudaq is not installed in this environment. "
            "Install it first, e.g. `pip install cudaq==0.16.0`."
        )
    # cudaq/__init__.py -> <site-packages>/cudaq -> <site-packages>
    site = os.path.dirname(os.path.dirname(spec.origin))
    dirs = {"lib": os.path.join(site, "lib"), "targets": os.path.join(site, "targets")}
    if not os.path.isdir(dirs["lib"]):
        raise RuntimeError(f"could not find CUDA-Q lib dir at {dirs['lib']}")
    os.makedirs(dirs["targets"], exist_ok=True)
    return dirs


def _require_platform() -> None:
    if platform.system() != "Darwin" or platform.machine() != "arm64":
        raise RuntimeError(
            "cudaq-metal only supports Apple Silicon macOS (arm64). "
            f"Detected {platform.system()}/{platform.machine()}."
        )


def _copy_atomic(src: str, dst: str, mode: int) -> None:
    """Copy via a temp file + rename. Readers never see a half-written file,
    and a process that already has the old dylib mapped keeps its (now
    unlinked) inode instead of having code pages rewritten underneath it —
    which macOS answers with SIGKILL (invalid code signature)."""
    fd, tmp = tempfile.mkstemp(dir=os.path.dirname(dst), prefix=".cudaq-metal-")
    try:
        with os.fdopen(fd, "wb") as out, open(src, "rb") as inp:
            shutil.copyfileobj(inp, out)
        os.chmod(tmp, mode)
        os.replace(tmp, dst)
    except BaseException:
        with contextlib.suppress(FileNotFoundError):
            os.unlink(tmp)
        raise


def _warn_if_version_mismatch() -> None:
    # Read the version from package metadata: importing cudaq here would make
    # it scan its targets before metal.yml is in place.
    try:
        ver = version(CUDAQ_DIST)
    except PackageNotFoundError:
        return
    if not ver.startswith(COMPATIBLE_CUDAQ):
        warnings.warn(
            f"cudaq-metal was built for CUDA-Q {COMPATIBLE_CUDAQ}, but this "
            f"environment has {CUDAQ_DIST} {ver}. The Metal backend may fail to "
            "load; rebuild it against your CUDA-Q if so.",
            stacklevel=4,
        )


def install(force: bool = False, quiet: bool = True) -> tuple[str, str]:
    """Copy the Metal backend into the active cudaq. Idempotent.

    Returns the (dylib, yml) destination paths. Set ``force=True`` to overwrite
    an existing install even if it is identical.
    """
    _require_platform()
    dirs = _cudaq_plugin_dirs()
    data = files("cudaq_metal") / "data"
    dsts, changed = [], False
    for name, where, mode in _FILES:
        src, dst = str(data / name), os.path.join(dirs[where], name)
        dsts.append(dst)
        # Compare contents, not just size: a rebuilt binary can keep its size.
        if force or not os.path.exists(dst) or not filecmp.cmp(src, dst, shallow=False):
            _copy_atomic(src, dst, mode)
            changed = True
            if not quiet:
                print(f"cudaq-metal: installed {dst}")
    if changed and "cudaq" in sys.modules:
        warnings.warn(
            "cudaq-metal installed or updated the 'metal' target after cudaq was "
            "already imported; cudaq only reads its targets at import, so "
            "cudaq.set_target('metal') will work after restarting Python. "
            "Import cudaq_metal before cudaq to avoid this.",
            stacklevel=3,
        )
    _warn_if_version_mismatch()
    return dsts[0], dsts[1]


def is_installed() -> bool:
    try:
        dirs = _cudaq_plugin_dirs()
    except RuntimeError:
        return False
    return all(os.path.exists(os.path.join(dirs[where], name)) for name, where, _ in _FILES)


def _main() -> None:
    """Console entry point: `cudaq-metal-install`."""
    dylib, yml = install(force=True, quiet=False)
    print(f"cudaq-metal ready. Use cudaq.set_target('metal').\n  {dylib}\n  {yml}")
