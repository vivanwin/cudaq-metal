// metal_sv.h — C-ABI for a naive Metal statevector engine.
//
// The engine keeps one complex64 statevector (float2 per amplitude) in a
// unified-memory MTLBuffer and applies gates on the Apple GPU. It is written
// with a plain C ABI so the NVQIR backend (MetalCircuitSimulator.cpp) can use
// it without dragging Objective-C++ into that translation unit.
//
// Gate application is LAZY: each apply call encodes a dispatch into a pending
// command buffer but does not synchronize. Call msv_synchronize() (or
// msv_data(), which implies it) before trusting the state. Batching many gates
// into one command buffer is what makes the GPU competitive for shallow
// circuits, where per-dispatch launch/commit overhead would otherwise dominate.
//
// Errors: functions that can fail return NULL / non-zero and leave a
// human-readable reason in msv_last_error() (per thread).
#ifndef METAL_SV_H
#define METAL_SV_H

#include <stddef.h>

// The kernels index amplitudes with 32-bit integers, so 2^31 amplitudes
// (16 GiB of complex64) is the hard ceiling, independent of GPU memory.
#define MSV_MAX_QUBITS 31
// Largest gate (in target qubits) one dispatch can apply; sizes the applykq
// kernel's register arrays and caps the fused-block size in the backend.
#define MSV_MAX_TARGETS 4

#ifdef __cplusplus
extern "C" {
#endif

typedef struct MSVState MSVState; // opaque handle

// Create a simulator for `nQubits` (1..MSV_MAX_QUBITS), state |0...0>.
// Returns NULL if there is no Metal device or the statevector does not fit.
MSVState *msv_create(int nQubits);
// Destroy a simulator. Its statevector buffer (17+ qubits) is kept, purgeable,
// for a later msv_create or msv_grow_to of the same size, which then skips
// allocating and GPU-mapping fresh memory (the two most recently freed sizes
// are kept); MSV_DISABLE_BUFFER_CACHE=1 frees it instead.
void msv_destroy(MSVState *s);

// Check, without allocating, that an `nQubits` statevector is supported here
// (qubit range and the GPU's maxBufferLength). Returns 0, or -1 with the
// reason in msv_last_error().
int msv_check_qubits(int nQubits);

// Reset the state back to |0...0> (no reallocation).
void msv_reset(MSVState *s);

// Apply a k-target gate (1 <= k <= MSV_MAX_TARGETS), conditioned on ALL
// qubits in controls[] being |1> (`nControls` may be 0). `mat` is the
// 2^k x 2^k matrix, row-major, interleaved complex (2*4^k floats); matrix
// index bit b corresponds to targets[b]. Qubit q is bit q of the amplitude
// index (little-endian), the order CircuitSimulatorBase hands an lsb backend.
// Returns 0, or -1 for invalid arguments (nothing is applied).
int msv_apply(MSVState *s, const float *mat, int k, const int *targets,
              const int *controls, int nControls);

// Flush all pending gates and block until the GPU has finished.
// Returns 0, or -1 if the GPU reported an error (the state is then invalid).
int msv_synchronize(MSVState *s);

// Grow the state to `newNQubits` (>= current), preserving the existing
// amplitudes. New qubits are the high-order bits, initialized to |0>, so this
// is a low-half copy, not a Kronecker product. Returns 0, or -1 on failure
// (the state is left unchanged).
int msv_grow_to(MSVState *s, int newNQubits);

// Synchronize and return the raw amplitude buffer: 2^n interleaved complex64
// values. It lives in unified memory, so the caller may read AND write it
// directly (e.g. to collapse the state on measurement); writes are visible to
// the next gate. Returns NULL if synchronizing failed.
void *msv_data(MSVState *s);

// Expectation values of Pauli strings, computed on the GPU in one pass over
// the state per batch of 8 terms. Term t is given by two qubit masks:
// xmask[t] = qubits carrying X or Y, zmask[t] = qubits carrying Z or Y (so Y
// sits in both). For each term, writes the UNNORMALIZED value
//   out[t] = <psi| P_t |psi>
//          = Re( i^nY * sum_j (-1)^popcount(j & z) * psi[j] * conj(psi[j ^ x]) )
// with nY = popcount(x & z). A term with both masks 0 yields <psi|psi>.
// Per-thread and per-simdgroup sums are fp32; the final reduction is double.
// Returns 0, or -1 for masks outside the state or a GPU failure.
int msv_pauli_expectations(MSVState *s, const unsigned *xmask,
                           const unsigned *zmask, size_t nTerms, double *out);

// Number of qubits the simulator currently holds.
int msv_num_qubits(const MSVState *s);

// 1 if environment variable `name` is set to anything but "", "0", "false",
// "no" or "off" (case-insensitive), else 0. Used for the MSV_DISABLE_* knobs.
int msv_env_flag(const char *name);

// Process-wide counters (for tests and benchmarks): fresh statevector
// allocations, statevectors served from the reuse pool, and expectation-value
// scratch buffers served from it. Any pointer may be NULL.
void msv_buffer_stats(unsigned long long *stateAllocations,
                      unsigned long long *stateReuses,
                      unsigned long long *scratchReuses);

// Reason for the most recent failure on this thread ("" if none).
const char *msv_last_error(void);

#ifdef __cplusplus
}
#endif

#endif // METAL_SV_H
