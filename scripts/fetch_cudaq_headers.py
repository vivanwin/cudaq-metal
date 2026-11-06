#!/usr/bin/env python3
"""Fetch the CUDA-Q C++ headers the Metal plugin compiles against.

Since CUDA-Q 0.16 the macOS pip wheel no longer ships C++ headers or CMake
configs, so instead of downloading the ~630 MB macOS toolkit installer we pull
just the headers we need straight from the cuda-quantum source tag (well under
1 MB). Starting from ``nvqir/CircuitSimulator.h`` we follow ``#include``
directives transitively and download every header that lives in one of CUDA-Q's
include roots (``runtime/``, ``runtime/include/``, ``cudaq/include/``) or in
the nlohmann/json copy vendored in the same repo (``tpls/json/``). Everything
comes from the same tag, so versions always match what the cudaq wheel was
built with. Includes not found in those roots (system / third-party headers)
are skipped; if one of those is actually needed the compile says so loudly.

Usage:
    python3 scripts/fetch_cudaq_headers.py 0.16.0 [OUT_DIR]

OUT_DIR defaults to toolchain/cudaq-headers-<version> (gitignored). Set
GITHUB_TOKEN to avoid the anonymous GitHub API rate limit (one call per run).
"""
from __future__ import annotations

import json
import os
import re
import sys
import time
import urllib.error
import urllib.request

REPO = "NVIDIA/cuda-quantum"
ROOTS = ("nvqir/CircuitSimulator.h",)
SEARCH_PREFIXES = (
    "runtime/",
    "runtime/include/",
    "cudaq/include/",
    "tpls/json/single_include_fwd/",
    "tpls/json/single_include/",
)
INCLUDE_RE = re.compile(r'^\s*#\s*include\s*[<"]([^>"]+)[>"]', re.M)


def _open(url: str, api: bool = False) -> bytes:
    req = urllib.request.Request(url)
    token = os.environ.get("GITHUB_TOKEN")
    if api and token:
        req.add_header("Authorization", f"Bearer {token}")
    attempts = 4
    for attempt in range(attempts):
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                return resp.read()
        except urllib.error.HTTPError as e:
            # 4xx (unknown tag, rate limit, ...) won't fix itself; 5xx/429 may.
            if (e.code < 500 and e.code != 429) or attempt == attempts - 1:
                raise RuntimeError(f"failed to fetch {url}: HTTP {e.code} {e.reason}") from e
        except OSError as e:  # connection reset, timeout, DNS: retry
            if attempt == attempts - 1:
                raise RuntimeError(f"failed to fetch {url}: {e}") from e
        time.sleep(2**attempt)
    raise AssertionError("unreachable")


def _repo_files(version: str) -> set[str]:
    url = f"https://api.github.com/repos/{REPO}/git/trees/{version}?recursive=1"
    tree = json.loads(_open(url, api=True))
    if tree.get("truncated"):
        raise RuntimeError("GitHub returned a truncated tree listing")
    return {e["path"] for e in tree["tree"] if e["type"] == "blob"}


def fetch(version: str, out_dir: str) -> int:
    files = _repo_files(version)

    def resolve(inc: str) -> str | None:
        for prefix in SEARCH_PREFIXES:
            if prefix + inc in files:
                return prefix + inc
        return None

    queue = list(ROOTS)
    seen: set[str] = set()
    fetched = 0
    while queue:
        inc = queue.pop()
        if inc in seen:
            continue
        seen.add(inc)
        src = resolve(inc)
        if src is None:
            continue  # system / third-party header
        dst = os.path.join(out_dir, inc)
        if os.path.exists(dst):
            text = open(dst, encoding="utf-8").read()
        else:
            raw = f"https://raw.githubusercontent.com/{REPO}/{version}/{src}"
            text = _open(raw).decode("utf-8")
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            # Write-then-rename: an interrupted run must not leave a truncated
            # header that later runs would trust as cached.
            with open(dst + ".part", "w", encoding="utf-8") as f:
                f.write(text)
            os.replace(dst + ".part", dst)
            fetched += 1
        here = os.path.dirname(inc)
        for child in INCLUDE_RE.findall(text):
            if here:  # quoted includes may be relative to the including header
                queue.append(os.path.normpath(os.path.join(here, child)))
            queue.append(child)
    return fetched


def main() -> None:
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    version = sys.argv[1]
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    out_dir = (
        sys.argv[2]
        if len(sys.argv) > 2
        else os.path.join(root, "toolchain", f"cudaq-headers-{version}")
    )
    n = fetch(version, out_dir)
    total = sum(len(f) for _, _, f in os.walk(out_dir))
    print(f"cudaq {version} headers: {total} files in {out_dir} ({n} downloaded now)")


if __name__ == "__main__":
    main()
