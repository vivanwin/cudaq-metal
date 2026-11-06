# cudaq-metal — Metal (Apple GPU) backend for NVIDIA CUDA-Q

A naive Apple-Silicon Metal statevector simulator exposed as a CUDA-Q / NVQIR
backend. `import cudaq_metal` and your existing CUDA-Q code runs on the Apple
GPU with `cudaq.set_target("metal")`. No new API, no code restructuring.

> **Unofficial.** This is an independent community project, not affiliated
> with or endorsed by NVIDIA. CUDA-Q is NVIDIA's; this package only plugs a
> Metal simulator into it.

> **🤖 Built entirely with Anthropic's Claude Code.**  
> Every line of C++, Objective-C++, Python, CMake, MSL, tests, benchmarks,
> commits, docs — this file included — was designed and implemented in a
> pair-programming session with Claude. Author-of-record: [@vivanwin](https://github.com/vivanwin);
> co-author on every commit: Claude (Anthropic), see the `Co-Authored-By` trailers.
> Read the [BUG_INVESTIGATION.md](BUG_INVESTIGATION.md) for the deep-dive that
> traced a real upstream bug in CUDA-Q's Darwin arm64 JIT (fixed in CUDA-Q 0.16).

## Install

```bash
pip install cudaq==0.16.0
pip install git+https://github.com/vivanwin/cudaq-metal.git
```

- **Apple Silicon Mac (arm64), macOS 13 or newer** — the same floor as the
  cudaq wheels. The wheel is tagged `macosx_13_0_arm64`, so pip refuses it on
  Linux, Windows and Intel Macs.
- **Python 3.11–3.14.**
- **CUDA-Q 0.16.x.** The bundled `libnvqir-metal.dylib` is ABI-matched to it
  (the package depends on `cuda-quantum-cu13>=0.16,<0.17`, the binary behind
  `pip install cudaq` on macOS). Pin `cudaq==0.16.0`: the newer
  `cudaq==0.16.0.post1` can't be installed on macOS because it requires a
  `cuda-quantum-cu13==0.16.0.post1` that NVIDIA only published for Linux.
- Optional: Classiq (tested with 1.34) if you build kernels with
  `qprog_to_cudaq_kernel`.

## Use it

```python
import cudaq_metal                       # ← registers the target; import it before cudaq
import cudaq
from cudaq import spin

@cudaq.kernel
def ghz(n: int):
    q = cudaq.qvector(n)
    h(q[0])
    for i in range(n - 1):
        x.ctrl(q[i], q[i + 1])

cudaq.set_target("metal")
print(cudaq.get_state(ghz, 3))                                  # statevector
print(cudaq.observe(ghz, spin.z(0) * spin.z(2), 20).expectation())  # 1.0
```

That's it. `import cudaq_metal` only copies the plugin into your cudaq install
so the `metal` target exists; it doesn't patch anything. From there it's plain
CUDA-Q: `cudaq.sample`, `cudaq.observe`, `cudaq.run` and `cudaq.get_state`
return cudaq's own result objects, for native `@cudaq.kernel` functions and
Classiq-built kernels (`qprog_to_cudaq_kernel`) alike — see
[`examples/`](examples/).

Import `cudaq_metal` **before** `cudaq`: cudaq reads its list of targets once,
at import. If cudaq was imported first, the very first install warns you and
`set_target("metal")` works after a Python restart.

## Examples

Two ready-to-run scaling examples in [`examples/`](examples/):

- [`brickwork_native.py`](examples/brickwork_native.py) — pure CUDA-Q,
  hardware-efficient ansatz (RY layer + CX-RZ-CX brickwork). Runs on `metal`
  by default; set `TARGET = "qpp-cpu"` to compare with the CPU simulator.
- [`brickwork_classiq.py`](examples/brickwork_classiq.py) — **identical
  gate sequence** synthesized through Classiq. Runs on Metal via
  `qprog_to_cudaq_kernel`. Bump `N` upward as you like.

Both are ~30 lines including imports; scale up by changing the `N` and
`DEPTH` constants at the top.

## Performance

The benchmark runs the `brickwork_classiq.py` circuit — synthesized by
Classiq and converted with `qprog_to_cudaq_kernel` — on CUDA-Q's CPU
simulator (`qpp-cpu`) and on `metal`, timing one
`cudaq.observe(kernel, Σ Zᵢ)` per point; synthesis and kernel construction
happen before the clock starts. Apple M5 Pro (24 GB), CUDA-Q 0.16.0, mean of
3 runs:

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="benchmarks/results/examples-dark.png">
  <img alt="Time per cudaq.observe() vs qubit count for a Classiq circuit: qpp-cpu doubles per qubit to about 2 s at 17 qubits; metal stays near 1–2 ms up to 17 qubits, about 1,000 times faster there, and reaches 30 qubits in about 18 s." src="benchmarks/results/examples.png">
</picture>

- **qpp-cpu** doubles per qubit, as a CPU statevector does: ~2 s per observe
  at n=17.
- **metal** stays at ~1–2 ms up to n=17, where the fixed GPU dispatch cost
  dominates, and ~10 ms at n=20. From there it doubles per qubit as memory
  bandwidth takes over: ~0.5 s at n=25 (256 MB statevector).
- **At n=17, metal is ~1,000× faster** (2.07 s → 2.1 ms). Metal's
  few-millisecond times vary between runs at these sizes; an earlier run
  measured 550–900×.
- Metal reaches **n=30** (an 8 GiB statevector) in ~18 s per observe;
  extrapolating qpp-cpu's doubling puts it at several hours.
- At 8 qubits there is nothing to win: both take about 1 ms.

The frontend doesn't change these numbers: the same gates written as a
native `@cudaq.kernel` (`brickwork_native.py`) simulate at the same speed.

Reproduce with (needs `classiq.authenticate()` once, plus matplotlib for the
plot):

```bash
python benchmarks/examples_scaling_bench.py
python benchmarks/plot_examples.py
```

## What's in the box

| Path | Purpose |
|------|---------|
| `src/metal_sv.{h,mm}` | Metal statevector engine (C-ABI, Objective-C++). One unified-memory MTLBuffer, fast `apply1q` and `apply2q` kernels + general `applykq` up to 4 targets, lazy command-buffer batching, pipelines compiled once per process. |
| `src/MetalCircuitSimulator.cpp` | The NVQIR backend: `CircuitSimulatorBase<float>` subclass. Gate fusion, multi-core sampling / expectation values / measurement over the unified-memory state, zero-copy `get_state`, state initialization, and a tripwire that warns loudly if cudaq ever silently drops a kernel's gates again (the old Darwin arm64 bug). |
| `scripts/` | Build tooling: `fetch_cudaq_headers.py` (pulls the ~0.7 MB of CUDA-Q headers the plugin needs from the matching source tag), `build_backend.sh`, `refresh_binary.sh` (rebuild + bundle + wheel). |
| `cudaq_metal/` | Python package. Bundles the prebuilt plugin and its CUDA-Q target config (`data/metal.yml`, fp32) and installs both into the active cudaq on import (also available as `cudaq-metal-install`). |
| `tests/` | pytest suite checking `metal` against `qpp-cpu`: every kernel, fusion on/off, sampling, observe, measurement, reset, state init, errors. |
| `benchmarks/` | Scaling benchmark + measured CSV + plot generator, and a fusion A/B benchmark. |
| `examples/` | Small end-to-end examples. |
| `BUG_INVESTIGATION.md` | Root-cause deep-dive on the (now fixed) CUDA-Q Darwin arm64 bug and every workaround we tried. |

## Building from source

Only needed when changing the C++/Metal code or moving to a new CUDA-Q
version. Requires Xcode command-line tools and CMake (Ninja optional):

```bash
pip install cudaq==0.16.0
bash scripts/refresh_binary.sh "$(which python)"
```

That reads the CUDA-Q version from the active environment, fetches the
matching C++ headers from the cuda-quantum source tag (the macOS wheel stopped
shipping them in 0.16, and the full toolkit installer is ~630 MB), builds the
plugin against the wheel's own libraries, copies it into `cudaq_metal/data/`
and writes the wheel to `dist/`. For a new CUDA-Q minor version also bump
`COMPATIBLE_CUDAQ` in `cudaq_metal/_install.py` and the `cuda-quantum-cu13`
pin in `pyproject.toml`.

To develop against the checkout, install it editable — the next
`import cudaq_metal` copies a freshly refreshed dylib into cudaq — and run the
test suite:

```bash
pip install -e ".[test]"
python -m pytest tests/
```

## Qubit ceiling on your machine

Two limits apply; the lower one wins:

1. **31 qubits, always.** The kernels index amplitudes with 32-bit integers,
   so the engine refuses anything above n=31 (a 16 GiB statevector).
2. **`MTLDevice.maxBufferLength`.** Metal caps a single `MTLBuffer` at
   [`maxBufferLength`](https://developer.apple.com/documentation/metal/mtldevice/2966563-maxbufferlength) —
   a *hard* Metal-API limit, [confirmed by Apple engineers on the developer
   forums](https://developer.apple.com/forums/thread/61188). We hold the
   whole statevector as one contiguous buffer, so it bounds the qubit count.
   It scales with device memory.

Empirically on an **Apple M5 Pro (24 GB unified)**:

- `maxBufferLength` = **14.30 GB** (13.32 GiB)
- `recommendedMaxWorkingSetSize` = 19.07 GB (advisory)
- **n=30 → 8.6 GB buffer → OK** (~18 s per observe, verified on-machine)
- **n=31 → 17.2 GB buffer → fails** with
  `a 31-qubit statevector needs 16.00 GiB, but this GPU's maxBufferLength is 13.32 GiB`

So **n=30 is the ceiling on a 24 GB M5 Pro**; n=31 needs a Mac whose
`maxBufferLength` is at least 16 GiB, i.e. one with more unified memory.
Going past 31 qubits needs a redesigned engine (64-bit indexing and a
statevector split across multiple buffers).

The timing curve stays log-linear right up to the wall (the metal line of
the benchmark above):

| n | statevector | observe |
|---|---|---|
| 25 | 256 MB | 0.49 s |
| 27 | 1 GB | 2.0 s |
| 29 | 4 GB | 8.7 s |
| 30 | 8 GB | 18.5 s |

## Limitations, honestly

- **fp32 statevector only** — Metal has no fp64.
- **At most 31 qubits**, and fewer on smaller Macs (see above).
- **Gates: 1–4 qubit targets** (any number of controls). Anything wider raises
  an error (rare in practice — Classiq's synthesizer produces 1- and 2-qubit
  gates for QAOA/VQE).
- **No noise models** — this is an ideal statevector simulator.
- **One CUDA-Q minor version per release.** The plugin is compiled against
  CUDA-Q's C++ internals, so each CUDA-Q minor release needs a rebuild (see
  [Building from source](#building-from-source)).

## The Darwin arm64 bug (fixed in CUDA-Q 0.16)

Up to CUDA-Q 0.15, parametric `observe` loops on Apple Silicon silently
returned the untouched |0..0⟩ state in a large fraction of processes: an
exception stashed inside a JIT frame made cudaq drop every gate. cudaq-metal
0.1 shipped a workaround (a worker subprocess behind a patched
`cudaq.observe`, plus `ReliableObserver`). CUDA-Q 0.16 fixed the underlying
problem: with no workaround, 1,050 observes across 35 fresh processes
(Classiq brickwork, 15-qubit warm-start QAOA, native `@cudaq.kernel` loops)
all matched `qpp-cpu`. So 0.2.0 removed the workaround, which also makes
observe loops roughly 7× faster. The plugin still prints a loud warning if a
kernel's gates are ever silently dropped again. Full history in
[BUG_INVESTIGATION.md](BUG_INVESTIGATION.md).

## License

Apache-2.0 (see [`LICENSE`](LICENSE)).
