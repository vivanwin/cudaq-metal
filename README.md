# cudaq-metal — Metal (Apple GPU) backend for NVIDIA CUDA-Q

A naive Apple-Silicon Metal statevector simulator exposed as a CUDA-Q / NVQIR
backend. `import cudaq_metal` and your existing CUDA-Q code runs on the Apple
GPU with `cudaq.set_target("metal")`. No new API, no code restructuring.

> **🤖 Built entirely with Anthropic's Claude Code.**  
> Every line of C++, Objective-C++, Python, CMake, MSL, tests, benchmarks,
> commits, docs — this file included — was designed and implemented in a
> pair-programming session with Claude. Author-of-record: [@vivanwin](https://github.com/vivanwin);
> co-author on every commit: `Claude Opus 4.7 <noreply@anthropic.com>`.
> Read the [BUG_INVESTIGATION.md](BUG_INVESTIGATION.md) for the deep-dive that
> traced a real upstream bug in CUDA-Q's Darwin arm64 JIT.

## Install

```bash
pip install cudaq==0.15.0
pip install git+https://github.com/vivanwin/cudaq-metal.git
```

- Apple Silicon macOS (arm64) with a Metal GPU.
- CUDA-Q 0.15.x (the shipped `libnvqir-metal.dylib` is ABI-matched to that).
- Classiq ≥ 1.2 recommended if you use `qprog_to_cudaq_kernel` (this is where
  the transparent VQE reliability layer kicks in).

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

That's it. Under the hood, `cudaq_metal` monkey-patches `cudaq.observe` to
route metal-target calls through a persistent worker subprocess with restart-
on-poison, working around an upstream Darwin arm64 JIT bug transparently. The
returned object exposes the full `observe_result` API: `.expectation()`,
`.counts()`, `.get_spin()`, `.dump()`, plus `shots_count=…` support.

Non-metal targets (`qpp-cpu`, `nvidia`, etc.) and native `@cudaq.kernel`
objects pass through completely unchanged.

## Benchmark: warm-start QAOA (15-node Max-Cut, 30-iter COBYLA)

Real measurements on an Apple M5 Pro (24 GB), CUDA-Q 0.15.1, Classiq 1.26.
Each point is mean ± std across 3 optimizer runs; every run finished with 100%
correct results.

![Warm-start QAOA scaling — Metal vs qpp-cpu](benchmarks/results/scaling.png)

**Highlights**:

- At n=15 (128k-amplitude statevector): Metal 2.5 s vs qpp-cpu 23 s — **9× faster**.
- At n=17: Metal 6.3 s vs qpp-cpu 111 s — **17× faster**.
- Past n=17, qpp-cpu becomes impractical (>2 min per COBYLA loop). Metal
  reaches n=24 in 70 s.
- Curve stays log-linear: doubling the statevector doubles the observed cost,
  as expected for a memory-bandwidth-bound simulator.

Reproduce:

```bash
python benchmarks/scaling_bench.py --min 13 --max 18 --trials 3
python benchmarks/plot_scaling.py   # re-render the PNG
```

The [benchmarks/](benchmarks/) directory contains the raw CSV, the plot
generator, and the scaling driver.

## What's in the box

| Path | Purpose |
|------|---------|
| `src/metal_sv.{h,mm}` | Metal statevector engine (C-ABI, Objective-C++). One unified-memory MTLBuffer, fast `apply1q` and `apply2q` kernels + general `applykq` up to 4 targets, lazy command-buffer batching, strictly-beneficial gate fusion. |
| `src/MetalCircuitSimulator.cpp` | The NVQIR backend: `CircuitSimulatorBase<float>` subclass. Implements sample, `get_state`, single-qubit measurement, and the Darwin-arm64 deferred-exception guard. |
| `targets/metal.yml` | CUDA-Q target config (fp32). |
| `cudaq_metal/` | Python package. Auto-installs the plugin on import, monkey-patches `cudaq.observe` for transparent worker-subprocess reliability. Exports `ReliableObserver` as an explicit escape hatch. |
| `benchmarks/` | Reproducible benchmark script + measured CSV + plot generator. |
| `examples/` | Small end-to-end examples. |
| `BUG_INVESTIGATION.md` | Root-cause deep-dive on the upstream CUDA-Q Darwin arm64 bug and every workaround we tried. |

## Escape hatches

- **`cudaq_metal.disable_transparent()`** — restores raw `cudaq.observe` and
  `qprog_to_cudaq_kernel`. Use if you need the actual `observe_result` object
  cudaq returns rather than the proxy (rare — everything on the real object
  works on the proxy too).
- **`cudaq_metal.ReliableObserver(qprog, hamiltonian)`** — explicit
  subprocess-worker observe loop. You almost never need this now that
  transparent mode is default; useful if you want to manage a specific
  worker's lifecycle yourself.

## Limitations, honestly

- **fp32 statevector only** — Metal has no fp64.
- **Gates: 1–4 qubit targets.** Anything wider throws (rare in practice —
  Classiq's synthesizer produces 1- and 2-qubit gates for QAOA/VQE).
- **`cudaq.sample` on the metal target isn't recovery-wrapped** because a
  legitimately-peaked distribution is indistinguishable from a poisoned one.
  Observe-based loops are fully reliable; a stray one-shot sample that looks
  wrong just needs a re-run.
- **Only `qprog_to_cudaq_kernel`-created kernels are tracked** by transparent
  mode. Native `@cudaq.kernel` in a parametric loop still hits the upstream
  bug (that's cudaq's baseline behavior, not a regression).

## Upstream

There is a real bug in CUDA-Q's Darwin arm64 JIT (silent |0..0⟩ from parametric
observe calls after a stashed exception). Full characterization in
[BUG_INVESTIGATION.md](BUG_INVESTIGATION.md); ready to file at
[NVIDIA/cuda-quantum](https://github.com/NVIDIA/cuda-quantum) when the time is
right. Once fixed upstream, the transparent-mode workaround becomes vestigial
and can be removed.

## License

Apache-2.0 (see `pyproject.toml`).
