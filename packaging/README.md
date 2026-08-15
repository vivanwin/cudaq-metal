# cudaq-metal

A naive **Metal (Apple GPU) statevector backend** for NVIDIA CUDA-Q on Apple
Silicon. Adds a `metal` target to your CUDA-Q Python projects.

```python
import cudaq_metal      # installs the backend into the active cudaq (idempotent)
import cudaq

cudaq.set_target("metal")

@cudaq.kernel
def ghz(n: int):
    q = cudaq.qvector(n)
    h(q[0])
    for i in range(n - 1):
        x.ctrl(q[i], q[i + 1])
    mz(q)

print(cudaq.sample(ghz, 20, shots_count=1000))
```

## Install

```bash
pip install cudaq==0.15.0     # (or 0.15.1) — a compatible CUDA-Q must be present
pip install cudaq-metal        # or: pip install /path/to/this/repo/packaging
```

Then either `import cudaq_metal` (auto-installs the plugin) or run the one-off
`cudaq-metal-install` command.

## Requirements & scope

- **Apple Silicon macOS (arm64)** with a Metal GPU.
- **CUDA-Q 0.15.x** — the shipped binary is ABI-matched to that build. For a
  different CUDA-Q, rebuild the backend from source (see the repo's top-level
  `scripts/build_backend.sh`) and drop the new `libnvqir-metal.dylib` into
  `src/cudaq_metal/data/`.
- fp32 statevector; supports 1- and 2-qubit-target gates (H/X/Z/rotations,
  CX/CZ/CCX, SWAP/iSWAP/SU(4), …). Sampling, `mz`, and `cudaq.get_state()` work.

## Notes

- The plugin is copied next to CUDA-Q's other `libnvqir-*.dylib` files
  (`<site-packages>/lib`) plus `targets/metal.yml`. `cudaq_metal.install(force=True)`
  refreshes it after upgrades.
- This is a naive bandwidth-bound simulator: it beats CUDA-Q's `qpp-cpu` on
  larger / deeper circuits (crossover ~12–15 qubits) and reaches ~28 qubits in
  24 GB. It does not aim to match cuQuantum.
