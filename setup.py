"""Build hook: make the wheel Apple-Silicon-macOS-only.

All metadata lives in pyproject.toml. This file only exists because the wheel
bundles a native arm64 macOS dylib, so it must not be tagged ``py3-none-any``
(pip would happily install it on Linux / Windows / Intel Macs and it would fail
at import). We tag it ``py3-none-macosx_13_0_arm64`` instead: any Python 3, no
CPython ABI dependency, Apple Silicon macOS 13+ only — the same platform floor
as the cudaq wheels themselves.

Building (and therefore installing from source, e.g. ``pip install git+...``)
on any other platform fails early with a clear message. Set
CUDAQ_METAL_ALLOW_CROSS_BUILD=1 to build the (still mac-only) wheel elsewhere.
"""
import os
import platform
import sys

from setuptools import setup
from setuptools.command.bdist_wheel import bdist_wheel

# Keep in sync with CMAKE_OSX_DEPLOYMENT_TARGET in CMakeLists.txt.
PLAT = "macosx_13_0_arm64"


class MacOnlyWheel(bdist_wheel):
    def finalize_options(self):
        on_apple_silicon = sys.platform == "darwin" and platform.machine() == "arm64"
        if not on_apple_silicon and not os.environ.get("CUDAQ_METAL_ALLOW_CROSS_BUILD"):
            raise SystemExit(
                "cudaq-metal only supports Apple Silicon macOS (arm64); "
                f"this machine is {sys.platform}/{platform.machine()}."
            )
        super().finalize_options()
        self.root_is_pure = False  # ships a native dylib

    def get_tag(self):
        return "py3", "none", PLAT


setup(cmdclass={"bdist_wheel": MacOnlyWheel})
