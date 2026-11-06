# Metal backend + CUDA-Q kernel-builder: repeated-evaluation bug

**Status (2026-11-06): fixed upstream in CUDA-Q 0.16.** With every workaround
removed, 1,050 parametric `observe` calls across 35 fresh processes (Classiq
brickwork, 15-qubit warm-start QAOA built through Classiq, native
`@cudaq.kernel` loops) all matched `qpp-cpu`, and the
`checkNotSilentlyDeferred` banner never fired. On 0.15 most processes were
poisoned. CUDA-Q 0.16 reworked the deferral code (`measureQubit` exceptions are
now deferred too and `rethrowDeferredKernelException()` is part of the public
`ExecutionContext` API). cudaq-metal 0.2.0 therefore removed the Python
workarounds (`reset`/`observe`/`sample` wrappers, `ReliableObserver`,
transparent mode) and the ctypes clear helper; only the C++ warning banner
remains, as a tripwire for regressions.

Everything below is the original investigation on CUDA-Q 0.15, kept as a
record.

**Original status (0.15): root cause fully characterized (upstream cudaq bug).
Best-effort mitigations shipped in this repo (`checkNotSilentlyDeferred` warning
banner + `cudaq_metal.reset` / `cudaq_metal.observe` / `cudaq_metal.sample`).
Full reliability required an upstream cudaq fix or a subprocess-per-call
workflow.**

## Root cause (2026-08-18)

On Darwin arm64, cudaq's JIT-compiled kernel-builder code non-deterministically
throws C++ exceptions that cannot unwind through a JIT frame (upstream LLVM /
libunwind issue, `llvm-project#49036`). CUDA-Q's runtime works around unwinding
with a defer/rethrow mechanism (`nvqir/CircuitSimulator.h`, lines 908–931):

- Any exception inside a JIT-compiled kernel is stashed on
  `cudaq::ExecutionContext::deferredKernelException` via
  `deferOrThrowKernelException()`.
- Once stashed, `kernelExecutionDeferred()` returns true.
- `enqueueGate` and `flushGateQueueImpl` become no-ops — every queued gate for
  the current kernel invocation is silently dropped.
- The launcher above the JIT boundary is *supposed* to call
  `cudaq::QPU::rethrowDeferredKernelException()` (declared in
  `include/cudaq/platform/qpu.h`). **On the observe path on Darwin arm64 it
  doesn't fire (or fires and gets swallowed).** So the deferred exception
  stays stashed, gate queues keep being dropped, `sample()` reads the untouched
  `|0…0⟩` state, expectation values collapse to `0.000000` with no error
  visible to the user.

Confirmed by instrumenting `sample()` to log `cudaq::kernelExecutionDeferred()`
at entry and running the actual failing script (`../cudaq qaoa/qaoa_warm_start.py`,
15-node warm-start QAOA, 16-edge Max-Cut Hamiltonian). Ratio of `deferred=1`
samples to total samples matched the ratio of failed calls exactly:

| successful calls | `deferred=1` samples | expected  |
|------------------|----------------------|-----------|
| 0/4              | 64/64                | 4×16 = 64 |
| 1/4              | 48/64                | 3×16 = 48 |
| 4/4              | 0/64                 | 0         |

The exception is thrown from applyGate itself only in the `nT > 4 targets`
path — but that path did **not** fire in any failing repro (verified by
replacing the throw with a log-and-skip). The exception is coming from cudaq's
own code inside the JIT frame; we can only see the effect
(`kernelExecutionDeferred=true`), not the message
(`std::rethrow_exception(deferredKernelException)` itself crashes with the same
unwinder bug).

## Fresh-context observation

Instrumenting `cudaq::getExecutionContext()->deferredKernelException` at
`addQubitsToState` entry (before any gate is applied for a new observe)
revealed:

- Each observe() call gets a **fresh** `ExecutionContext` (different pointer).
- The fresh context arrives with `inKernelLaunch = 0` at addQubitsToState.
- But `deferred = 1` — the fresh context is already poisoned.

That's only possible if `deferredKernelException` gets set during the observe's
own setup or transitively from an earlier context. In either case, from our
side it looks like the poisoned state persists across observes.

## Attempted fixes

We tried every intervention that seemed plausible; the results in brief:

| Attempt                                                     | Result                                                        |
|-------------------------------------------------------------|----------------------------------------------------------------|
| `throw std::runtime_error` from `sample()`                  | Hits the Darwin arm64 unwinder bug itself; SIGSEGV.            |
| `std::rethrow_exception(deferredKernelException)` to inspect it | Same unwinder bug; SIGSEGV.                                     |
| Clear `deferredKernelException` from `deallocateStateImpl` (end-of-kernel) | ~100% SIGSEGV downstream in cudaq's finalization.               |
| Clear at `addQubitsToState` entry only when `inKernelLaunch = 0` | Sometimes works (~10-20%), often SIGSEGV or wrong results.     |
| Clear from a *Python-callable* C function (Python is safely outside a JIT frame) | Doesn't crash. Per-call success rate depends on the process's underlying JIT state (see below). |
| Retry loop around observe/sample                            | Works when the process is "healthy"; hangs forever when the process's JIT state is permanently broken. |
| Prime with a native `@cudaq.kernel` before Classiq observe | Made things worse (2/40 vs baseline ~7-11/40).                 |

Empirically, whether a process's JIT lands in a "healthy" or "poisoned" state
seems set early (probably at first Classiq-kernel invocation) and doesn't
recover in-process. A fresh Python process might be healthy; if it isn't, no
amount of `reset()` / retry / re-set-target rescues it.

## What shipped in cudaq-metal 0.1 (items 2–3 removed in 0.2)

1. **`checkNotSilentlyDeferred` in `MetalCircuitSimulator.cpp`** (still there) — called at the
   top of `sample()` and `getSimulationState()`. Prints a loud, unmissable
   stderr banner on every affected call so a silent `|0..0⟩` result is at
   least visible. Cannot itself throw (would hit the unwinder bug) or clear
   the deferred state (would corrupt cudaq's lifecycle bookkeeping).
2. **`cudaq_metal.reset()`** — Python helper that clears
   `ctx->deferredKernelException` via ctypes. Python is guaranteed to be
   outside any JIT frame. Call between observe/sample calls in a variational
   loop to give each attempt an independent chance.
3. **`cudaq_metal.observe()` and `cudaq_metal.sample()`** — drop-in wrappers
   that call `reset()` and retry-on-`|0..0⟩` up to a configurable number of
   attempts. Useful for "healthy" processes; raises after N failures otherwise.

## Recommended usage on CUDA-Q ≤ 0.15 with cudaq-metal 0.1 (historical)

```python
import cudaq
import cudaq_metal   # registers metal target

cudaq.set_target("metal")

for step in range(30):
    # cudaq_metal.observe = reset + observe + retry
    r = cudaq_metal.observe(kernel, H, *params, max_retries=20)
    expv = r.expectation()
```

If the wrapper raises `RuntimeError: gave up after N attempts`, the process is
in a "poisoned" JIT state. **Restart the Python process** (a fresh JIT tends to
be healthy). For truly bulletproof VQE runs, wrap the whole loop in a
subprocess-per-call pattern: launch a subprocess for each `observe`, then
communicate the result back.

## Reproducing

```bash
# 15-node warm-start QAOA, non-deterministic; run several times.
../cudaq\ qaoa/.venv/bin/python benchmark/repro_realqaoa.py
```

## Upstream ask (no longer needed — fixed in CUDA-Q 0.16)

- File at https://github.com/NVIDIA/cuda-quantum with a minimal repro of
  `benchmark/repro_realqaoa.py`.
- Root cause is that `cudaq::QPU::rethrowDeferredKernelException()`
  (declared in `include/cudaq/platform/qpu.h`) does not fire (or its throw is
  swallowed) on the observe path in the current `DefaultQPU`. The deferred
  exception mechanism exists and works correctly; it just isn't being
  drained.
- Longer-term: figure out what's actually throwing inside the JIT frame during
  the Classiq-derived kernel's execution — from our side that's opaque
  because we cannot inspect `deferredKernelException` without hitting the
  unwinder bug ourselves.
