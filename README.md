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
import cudaq_metal                       # ← auto-registers the target
import cudaq
from classiq import synthesize, qprog_to_cudaq_kernel, pauli_operator_to_cudaq_spin_op

cudaq.set_target("metal")

qprog  = synthesize(main)                # once, in your code
kernel = qprog_to_cudaq_kernel(qprog)
H      = pauli_operator_to_cudaq_spin_op(...)

# Vanilla CUDA-Q — no new syntax:
for step in range(N):
    val = cudaq.observe(kernel, H, *params).expectation()
```

That's it. `import cudaq_metal` only copies the plugin into your cudaq install
so the `metal` target exists; it doesn't patch anything. From there it's plain
CUDA-Q: `cudaq.sample`, `cudaq.observe` and `cudaq.get_state` return cudaq's
own result objects, for native `@cudaq.kernel` functions and Classiq-built
kernels alike.

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

Full 2×2 matrix — {native @cudaq.kernel, Classiq qprog_to_cudaq_kernel} ×
{qpp-cpu, metal} — on the **same primitive-gate hardware-efficient ansatz**
(RY layer + CX-RZ-CX brickwork, depth=5). Native and Classiq emit identical
gate sequences, so the frontend cancels out and each line measures only its
runtime engine. Only the `cudaq.observe(...)` call is timed; synthesis /
kernel construction happen before `t0`. Metal paths sweep to n=25:

![Examples scaling — engine × target matrix](benchmarks/results/examples.png)

**Key observations:**
- The two **CPU** lines overlap: native ≈ Classiq → qpp-cpu, both doubling
  per qubit as expected for a CPU statevector (n=17 ≈ 2 s).
- The two **metal** lines overlap too: native ≈ Classiq → metal, ~1–10 ms
  per observe through n=20, climbing to ~500–900 ms at n=25 (256 MB statevector — where
  memory bandwidth dominates).
- **At n=17: metal beats qpp-cpu by ~300–400×**  (2 s → 6 ms).
- Metal reaches **n=25** (33 M-amplitude statevector) in ~0.6–0.9 s, where CPU
  is not measurable within this benchmark's budget.

Reproduce with:

```bash
python benchmarks/examples_scaling_bench.py
python benchmarks/plot_examples.py
```

## What's in the box

| Path | Purpose |
|------|---------|
| `src/metal_sv.{h,mm}` | Metal statevector engine (C-ABI, Objective-C++). One unified-memory MTLBuffer, fast `apply1q` and `apply2q` kernels + general `applykq` up to 4 targets, lazy command-buffer batching, strictly-beneficial gate fusion. |
| `src/MetalCircuitSimulator.cpp` | The NVQIR backend: `CircuitSimulatorBase<float>` subclass. Implements sample, `get_state`, single-qubit measurement, and a tripwire that warns loudly if cudaq ever silently drops a kernel's gates again (the old Darwin arm64 bug). |
| `targets/metal.yml` | CUDA-Q target config (fp32). |
| `scripts/` | Build tooling: `fetch_cudaq_headers.py` (pulls the ~0.7 MB of CUDA-Q headers the plugin needs from the matching source tag), `build_backend.sh`, `refresh_binary.sh` (rebuild + bundle + wheel). |
| `cudaq_metal/` | Python package. Bundles the prebuilt plugin and installs it into the active cudaq on import (also available as `cudaq-metal-install`). |
| `benchmarks/` | Reproducible benchmark script + measured CSV + plot generator. |
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

## Qubit ceiling on your machine

Metal caps single `MTLBuffer` allocations at [`MTLDevice.maxBufferLength`](https://developer.apple.com/documentation/metal/mtldevice/2966563-maxbufferlength) —
this is a *hard* Metal-API limit, [confirmed by Apple engineers on the developer
forums](https://developer.apple.com/forums/thread/61188): allocations larger
than `maxBufferLength` throw / return `nil`, and the officially-recommended
workaround is splitting your data across multiple buffers. Since we hold the
whole statevector as one contiguous MTLBuffer, that's your qubit ceiling.

Empirically on an **Apple M5 Pro (24 GB unified)**:

- `maxBufferLength` = **14.30 GB**
- `recommendedMaxWorkingSetSize` = 19.07 GB (advisory)
- **n=30 → 8.6 GB buffer → OK** (~30 s per observe, verified on-machine)
- **n=31 → 17.2 GB buffer → fails** (exceeds `maxBufferLength`)

So **n=30 is the ceiling on a 24 GB M5 Pro** regardless of how much RAM is
free. To go higher you need either a larger Mac (`maxBufferLength` scales with
device memory — a 64 GB M-series is roughly n=32, a 128 GB M4 Max ~n=33) or
a redesigned engine that splits the statevector across multiple buffers.

Timing curve stays log-linear right up to the wall:

| n | statevector | observe |
|---|---|---|
| 25 | 256 MB | 0.6 s |
| 27 | 1 GB | 4 s |
| 29 | 4 GB | 14 s |
| 30 | 8 GB | 30 s |

## Limitations, honestly

- **fp32 statevector only** — Metal has no fp64.
- **Gates: 1–4 qubit targets.** Anything wider throws (rare in practice —
  Classiq's synthesizer produces 1- and 2-qubit gates for QAOA/VQE).
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
