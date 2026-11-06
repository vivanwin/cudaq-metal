"""Materialize the embedded worker scripts as files the subprocess can run."""
from __future__ import annotations

import hashlib
import os
import tempfile


def worker_script(name: str, source: str) -> str:
    """Return the path of a file containing ``source``, creating it if needed.

    Workers used to be written into the installed package directory, which
    failed on read-only installs, raced between concurrent processes, got swept
    into wheels built from a working tree, and (write-if-missing) kept running a
    stale worker after an upgrade. Now the file lives in a per-user temp cache,
    its name carries a hash of the source (a new version always gets a new
    file), and it is written atomically.
    """
    digest = hashlib.sha256(source.encode("utf-8")).hexdigest()[:16]
    cache = os.path.join(tempfile.gettempdir(), "cudaq_metal")
    os.makedirs(cache, exist_ok=True)
    path = os.path.join(cache, f"{name}-{digest}.py")
    if not os.path.exists(path):
        fd, tmp = tempfile.mkstemp(dir=cache, suffix=".tmp")
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(source)
        os.replace(tmp, path)
    return path
