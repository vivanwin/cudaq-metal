# Metal backend + CUDA-Q kernel-builder: repeated-evaluation bug

**Status:** **root cause found**. Upstream cudaq bug; a partial mitigation
(loud stderr warning on every affected call) is in place in this backend.
The wrong answer itself still slips through — see "Why we can't fully fix it".

## Root cause (nailed 2026-08-18)

On Darwin arm64, cudaq's JIT-compiled kernel-builder code non-deterministically
throws C++ exceptions that cannot unwind through a JIT frame (upstream LLVM /
libunwind issue). CUDA-Q's runtime works around this with a defer/rethrow
mechanism (`nvqir/CircuitSimulator.h`, ~lines 908–931 in 0.15.x):

- Any exception thrown from inside a JIT-compiled kernel is stashed on
  `cudaq::ExecutionContext::deferredKernelException` via
  `deferOrThrowKernelException()`.
- Once stashed, `kernelExecutionDeferred()` returns true.
- `enqueueGate` and `flushGateQueueImpl` become no-ops — the queued gates for
  the current kernel invocation are silently dropped.
- The launcher above the JIT boundary is *supposed* to call
  `cudaq::QPU::rethrowDeferredKernelException()` after the kernel returns.

**The bug:** in the observe/sample finalization path used by
`cudaq.observe(kernel, hamiltonian, *params)` on Darwin arm64, that final
`rethrowDeferredKernelException()` call does not fire (or fires and gets
swallowed). The deferred exception stays stashed, gate queues keep getting
dropped, and every subsequent `sample()` reads the untouched `|0…0⟩` state.
Expectation values collapse to `0.000000` with no error surfaced.

**Non-determinism:** the underlying JIT exception is a spurious unwinder
issue that fires randomly. Some runs of the same script produce 4/4 correct
evaluations; others produce 0/4, 1/4, or 2/4. When an evaluation is affected,
it is *completely* affected (405 gates for one 15-qubit warm-start QAOA
observe() → 0 gates emitted; sample() reads |0…0⟩).

The doc's original observation — "`applyGate` is never called on the failing
evaluations" — is exactly what the deferred-flush short-circuit produces.

## How we confirmed it

Instrumented `MetalCircuitSimulator.cpp` to log `kernelExecutionDeferred()` at
the top of every `sample()` call, then ran the actual failing script
(`../cudaq qaoa/qaoa_warm_start.py`, 15-node warm-start QAOA, 16-edge
Max-Cut Hamiltonian). Ratio of `deferred=1` samples to total samples matched
the ratio of failed calls exactly across dozens of runs:

| Run  | Successful calls | `deferred=1` samples | Expected  |
|------|------------------|----------------------|-----------|
| 1    | 0/4              | 64/64                | 4×16 = 64 |
| 2    | 1/4              | 48/64                | 3×16 = 48 |
| 3    | 2/4              | 32/64                | 2×16 = 32 |
| 4    | 3/4              | 16/64                | 1×16 = 16 |
| 5    | 4/4              | 0/64                 | 0×16 = 0  |

## Why we can't fully fix it from this backend

Three approaches were tried and rejected:

1. **`throw std::runtime_error` from `sample()`** when `kernelExecutionDeferred`
   is true. The throw itself hits the same Darwin arm64 unwinder bug (we're
   still inside the JIT frame during observe's per-term sampling loop), and
   terminates the process.
2. **Clear `deferredKernelException` and continue.** Silently returning a
   |0…0⟩ result confuses cudaq's kernel-lifecycle bookkeeping: it stops
   calling `deallocateStateImpl` between invocations, and `stateDimension`
   grows unboundedly (15 → 30 → 45 qubits) over subsequent calls. Eventually
   segfaults.
3. **`std::rethrow_exception(ctx->deferredKernelException)`** to inspect the
   original exception message — crashes immediately, same unwinder bug.

We can *only* keep the deferred-exception invariant intact by warning and
letting the wrong result through, or by aborting. We went with warning: the
mitigation is a **loud, unmissable stderr banner** printed by
`checkNotSilentlyDeferred()` in `sample()` and `getSimulationState()`. The
user sees the warning per affected call and can restart the process.

## The mitigation (currently in `src/MetalCircuitSimulator.cpp`)

```cpp
static void checkNotSilentlyDeferred(const char *op) {
    if (!kernelExecutionDeferred()) return;
    // Loud stderr banner: "WARNING: this call will return |0..0>, restart the
    // Python process." See MetalCircuitSimulator.cpp for the full message.
}
```

Called at the top of `sample()` and `getSimulationState()`. Produces one
banner per affected call, so an observe() with a 16-term Hamiltonian and 4
evaluations will produce up to 64 banners on a fully-failed run — enough
noise that the user cannot miss it.

## Reproducing

Two variables matter and neither of the doc's original repros pinpointed
them; both are needed to reliably trigger the bug:

1. **cudaq 0.15.1 vs 0.15.0.** The bug reproduces easily on 0.15.1 (commit
   `aca5853a`) and on 0.15.0 we saw it in the original doc. It also reproduces
   on the C++ builder-kernel path *and* through Classiq's
   `qprog_to_cudaq_kernel` bridge. It does *not* reproduce on native
   `@cudaq.kernel` circuits in any test we ran.
2. **A "real" circuit.** Trivial `cudaq.make_kernel()` builders (a single
   `ry(theta, q)`) do not trigger the bug, presumably because their JIT-emitted
   frame doesn't exercise whatever code path throws. The failing script's
   15-qubit warm-start QAOA ansatz with `phase()`-based cost layer reliably
   triggers it.

Fastest way to reproduce today:

```bash
cd "../cudaq qaoa"                    # sibling project
.venv/bin/python -c "..."             # any observe() loop against `metal`
```

See `benchmark/repro_realqaoa.py` in the cudaq-metal repo for a minimal
faithful reproducer. Run the same script 4-5 times: the number of surviving
calls per run varies 0-4 non-deterministically.

## Recommended next steps

1. **File upstream** at https://github.com/NVIDIA/cuda-quantum with a minimal
   repro. Point to `cudaq::QPU::rethrowDeferredKernelException()` (declared in
   `include/cudaq/platform/qpu.h`) — the launcher above the JIT boundary that
   should be calling it does not, or its throw is being swallowed further up
   in the observe path.
2. **In the meantime, on Darwin arm64:** watch stderr for the warning banner
   in any variational loop. Any occurrence means the entire run's results are
   suspect and the process must be restarted. Consider switching to `qpp-cpu`
   for Darwin arm64 variational workflows until fixed upstream.

## Currently uncommitted debug instrumentation

`src/MetalCircuitSimulator.cpp` still has `fprintf`-based tracing of
`ensureEngine`, `addQubitsToState`, `applyGate` (first 5, every 100th),
`sample`, `setToZeroState`, `deallocateStateImpl`. Keep for future work.
Rebuild after any change:

```bash
bash scripts/build_backend.sh
bash scripts/install_into_venv.sh   # or cp libnvqir-metal.dylib into project venv
```
