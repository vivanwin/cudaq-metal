// metal_sv.h — C-ABI for a naive Metal statevector engine.
//
// The engine keeps one complex64 statevector (float2 per amplitude) in a
// unified-memory MTLBuffer and applies gates on the Apple GPU. It is written
// with a plain C ABI so it can be linked from either the standalone benchmark
// or the CUDA-Q / NVQIR backend without dragging Objective-C++ into those
// translation units.
//
// Gate application is LAZY: each apply_* call encodes a dispatch into a
// pending command buffer but does not synchronize. Call msv_synchronize()
// (or any state-reading call, which implies it) before trusting the state or
// timing a result. Batching many gates into one command buffer is what makes
// the GPU competitive for shallow circuits, where per-dispatch launch/commit
// overhead would otherwise dominate.
#ifndef METAL_SV_H
#define METAL_SV_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct MSVState MSVState; // opaque handle

// Create a simulator for `nQubits`, statevector initialized to |0...0>.
// Returns NULL if no Metal device is available or allocation fails.
MSVState *msv_create(int nQubits);
void msv_destroy(MSVState *s);

// Reset the state back to |0...0> (no reallocation).
void msv_reset(MSVState *s);

// Apply a single-target 2x2 gate to qubit `target`, conditioned on ALL
// qubits listed in controls[] being |1>. `nControls` may be 0.
//
// `m` is the 2x2 matrix in row-major, interleaved complex order (8 floats):
//   { Re(u00),Im(u00), Re(u01),Im(u01), Re(u10),Im(u10), Re(u11),Im(u11) }
//
// Qubit convention: qubit q corresponds to bit q of the amplitude index
// (little-endian; qubit 0 is the least-significant bit).
void msv_apply_1q(MSVState *s, const float *m, int target,
                  const int *controls, int nControls);

// Apply a general k-target gate (1 <= k <= 4) with an arbitrary control mask.
// `mat` is the 2^k x 2^k matrix, row-major, interleaved complex (2*4^k floats).
// `targets[]` are the k target qubit indices (any order); matrix index bit b
// corresponds to targets[b] (little-endian within the target block, matching
// the ordering CircuitSimulatorBase hands an lsb backend). Lazy, like apply_1q.
void msv_apply_kq(MSVState *s, const float *mat, int k, const int *targets,
                  int nTargets, const int *controls, int nControls);

// Flush all pending gates and block until the GPU has finished.
void msv_synchronize(MSVState *s);

// Copy the full statevector (2^n complex amplitudes) into host arrays,
// interleaved-complex layout is NOT used here: separate real/imag arrays,
// each of length 2^n. Implies a synchronize. Either pointer may be NULL.
void msv_get_state(MSVState *s, float *outReal, float *outImag);

// Grow the state to `newNQubits` (>= current), preserving the existing
// amplitudes. Newly added qubits are the high-order bits, initialized to |0>
// (little-endian), so this is just a low-half copy — no Kronecker product.
// Flushes any pending gates first.
void msv_grow_to(MSVState *s, int newNQubits);

// Synchronize and return a pointer to the raw amplitude buffer, laid out as
// 2^n interleaved complex64 values (float real, float imag). The buffer lives
// in unified shared memory, so the caller may read AND write it directly
// (e.g. to collapse the state on measurement); changes are visible to the GPU
// on the next gate. Returns NULL on error.
void *msv_data(MSVState *s);

// Number of qubits the simulator currently holds.
int msv_num_qubits(const MSVState *s);

// Human-readable name of the Metal device backing this simulator (or "" ).
const char *msv_device_name(const MSVState *s);

#ifdef __cplusplus
}
#endif

#endif // METAL_SV_H
