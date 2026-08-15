# Metal statevector backend for CUDA-Q (Apple Silicon)

A naive **Metal (Apple GPU) statevector simulator** exposed as a CUDA-Q / NVQIR
backend, so you can run CUDA-Q kernels on an Apple Silicon GPU with
`--target metal` (C++) or `cudaq.set_target("metal")` (Python). The goal is a
simple, correct backend that is modestly-to-substantially faster than CPU-only
simulation on a Mac — not to match NVIDIA cuQuantum.

Built and tested on: **Apple M5 Pro, 24 GB, macOS 26.5, Command Line Tools only
(no full Xcode), CUDA-Q 0.15.0**.

## What works

- `nvq++ --target metal` (C++) and Python `cudaq` both run kernels on the GPU.
- Correct vs CUDA-Q's `qpp-cpu` for GHZ (10–28 qubits) and random SU(4) circuits
  (SWAP exact; SU(4) on adjacent/non-adjacent qubits matches within sampling noise).
- Single-target gates (H/X/Z/rotations, CX/CCX/CZ) via a fast kernel, plus general
  2–4-qubit-target gates (SWAP/iSWAP/SU(4)) via a general kernel.
- **Classiq → CUDA-Q → Metal**: author circuits in Classiq, `synthesize()`,
  `qprog_to_cudaq_kernel()`, run on `metal`.

## Layout

| Path | What |
|------|------|
| `src/metal_sv.{h,mm}` | Metal statevector engine (C-ABI, Objective-C++). One unified-memory `float2[2^n]` buffer, lazy command-buffer batching, `apply1q` + general `applykq` kernels. |
| `src/MetalCircuitSimulator.cpp` | The NVQIR backend: `nvqir::MetalCircuitSimulator : CircuitSimulatorBase<float>`. |
| `targets/metal.yml` | CUDA-Q target config (fp32). |
| `src/cpu_sv.hpp`, `benchmark/standalone_bench.cpp` | Lean CPU reference sim + a standalone (no-CUDA-Q) Metal-vs-CPU benchmark. |
| `benchmark/*.py` | Python cudaq benchmarks + validation + the Classiq QV pipeline. |
| `scripts/*.sh` | Build / install / benchmark drivers. |

## Design notes

- **fp32 only** — Metal has no fp64, so the backend is `CircuitSimulatorBase<float>`
  and `metal.yml` sets `-D CUDAQ_SIMULATION_SCALAR_FP32`.
- **Little-endian qubit order** matches CUDA-Q's internal ordering, so (unlike the
  `qpp` backend) there is **no index conversion**, and growing the state by a qubit
  is a plain low-half copy rather than a Kronecker product.
- **Lazy batching**: gates accumulate in one serial compute encoder / command buffer
  and only sync on sampling/measurement/state read. This amortizes GPU launch latency
  and is what lets even shallow circuits win.
- Sampling and single-qubit measurement are done host-side on the unified-memory
  buffer (no copies).

## Setup (one time)

Prerequisites: macOS arm64, Command Line Tools, Homebrew.

```bash
brew install cmake ninja
```

1. **CUDA-Q C++ toolkit** — installed to a space-free prefix `~/.cudaq-metal/cudaq`
   from the official `install_cuda_quantum_darwin-0.15.0.arm64`. Two local fixes were
   needed (see `scripts/` and comments):
   - `nvq++` has an upstream ordering bug that drops the SDK `-isysroot` for
     `cudaq-quake`; patched in `~/.cudaq-metal/cudaq/bin/nvq++` (backup `nvq++.orig`).
   - `nvq++` bakes an unquoted `-Wl,-rpath,$PWD`, so CUDA-Q programs are built from a
     **space-free symlink** `~/.cudaq-metal/proj` → this project (the project path has
     spaces).
2. **Python env** — `~/.cudaq-metal/venv` (Python 3.12) with
   `pip install "cudaq==0.15.0" "classiq[cudaq]"`.

## Build the backend

```bash
bash scripts/build_backend.sh      # builds + installs into the C++ toolkit
bash scripts/install_into_venv.sh  # copies the backend into the Python cudaq
```

## Use it in another project (pip — recommended)

The Metal backend ships as a small pip package (the repo root is the package)
that drops the prebuilt plugin into whatever CUDA-Q a project uses:

```bash
pip install cudaq==0.15.0
pip install "git+https://github.com/<you>/<repo>.git"
```

Or from a local checkout: `pip install .`  (or a prebuilt wheel under `dist/`).

```python
import cudaq_metal      # installs the backend into this env's cudaq (idempotent)
import cudaq
cudaq.set_target("metal")
```

The shipped binary (`cudaq_metal/data/libnvqir-metal.dylib`) is ABI-matched to
CUDA-Q 0.15.x. After changing the C++ sources, regenerate + repackage it with
`bash scripts/refresh_binary.sh` and commit `cudaq_metal/data/`.

## Use it (development install)

C++:
```bash
source ~/.cudaq-metal/cudaq/set_env.sh
export CUDAQ_SYSROOT="$(xcrun --show-sdk-path)"
cd ~/.cudaq-metal/proj          # space-free symlink
nvq++ --target metal -O2 benchmark/ghz.cpp -o ghz && ./ghz 20
```

Python:
```python
import cudaq
cudaq.set_target("metal")
# ... define @cudaq.kernel and cudaq.sample(...)
```

Classiq (needs `classiq.authenticate()` once):
```python
from classiq import synthesize, qprog_to_cudaq_kernel
import cudaq
qprog = synthesize(main)
kernel = qprog_to_cudaq_kernel(qprog)
cudaq.set_target("metal")
cudaq.sample(kernel, ...)
```

## Benchmarks (M5 Pro)

GHZ `cudaq::sample`, metal vs qpp-cpu (ms): n=16 → 76 vs 2.2 (35×); n=20 → 1917 vs
4.6 (≈400×). Brickwork SU(4) (depth=n): n=16 → 651 vs 7.3 (89×); metal handles n=24
(~276 SU(4) gates) in 0.45 s.

Caveat: `qpp-cpu` is a general-purpose, not-highly-tuned statevector sim, so the
headline factors are large. Against a *lean* hand-written CPU sim the GPU advantage
is a more modest ~3× (the real memory-bandwidth win). Both are reported.

## Limitations / TODO

- Gates with >4 target qubits unsupported; `getSimulationState()`/`get_state()` and
  state initialization from data not implemented (sampling only).
- No gate fusion (each gate is one dispatch, though all are batched in one buffer).
- CI build+package flow not set up yet.
