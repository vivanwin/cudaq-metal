// metal_sv.mm — naive Metal statevector engine (Objective-C++).
//
// State layout: one MTLBuffer of float2[2^n] in MTLStorageModeShared (unified
// memory, no host<->device copies). Amplitude index is little-endian in the
// qubit label: qubit q occupies bit q of the index.
//
// Four compute kernels:
//   apply1q  — 2x2 matrix on one target qubit, gated by a control mask. Each
//              thread owns one disjoint amplitude pair. Fast path for the
//              common single-target gates (H/X/Z/rotations, CX/CCX/CZ).
//   apply2q  — 4x4 matrix on two targets with O(1) bit-insertion addressing.
//              Handles SWAP/iSWAP/generic 2-qubit (e.g. QV SU(4)).
//   applykq  — general 2^k x 2^k matrix on k <= MSV_MAX_TARGETS targets. Each
//              thread owns one disjoint group of 2^k amplitudes, gathers them,
//              does the matrix-vector product, and scatters back.
//   pauli_expect — read-only: <psi|P|psi> for a batch of 8 Pauli strings per
//              grid row, reduced per simdgroup on the GPU and summed in
//              double on the host (see msv_pauli_expectations).
//
// Execution is lazy: dispatches accumulate in one open command buffer using a
// *serial* compute encoder (Metal orders serial-encoder dispatches with
// implicit barriers, so gate g+1 sees gate g's writes). We only commit + wait
// on msv_synchronize()/msv_data(). This amortizes launch/commit latency, which
// is what otherwise sinks the GPU on shallow circuits.
//
// The device, queue and compiled pipelines are process-wide (built once on
// first use); each MSVState owns only its statevector buffer. Freed buffers
// are kept for reuse (see BufferPool), because a fresh multi-GiB buffer costs
// more than most circuits.

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <dispatch/dispatch.h>
#include <strings.h>
#include <atomic>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <string>
#include <vector>

#include "metal_sv.h"

namespace {

// Must match the MSL structs byte-for-byte (see kMetalSource).
struct C2 { float x, y; }; // complex, 8 bytes
struct Gate1Q {
  C2 u00, u01, u10, u11; // 2x2 matrix, row-major
  unsigned targetMask;   // 1u << target
  unsigned lowMask;      // targetMask - 1
  unsigned controlMask;  // OR of (1u << control)
  unsigned pad;
};
static_assert(sizeof(Gate1Q) == 48, "Gate1Q layout must match MSL");

struct Gate2Q {
  C2 m[16];             // 4x4 row-major, matrix bit b -> targets[b]
  unsigned a, b;        // sorted target positions (a < b), for bit insertion
  unsigned t0mask, t1mask; // 1<<targets[0], 1<<targets[1] (original order)
  unsigned controlMask;
  unsigned pad;
};
static_assert(sizeof(Gate2Q) == 152, "Gate2Q layout must match MSL");

struct GateKQ {
  unsigned n;                         // total qubits
  unsigned k;                         // number of targets
  unsigned targetMask;                // OR of (1<<t) over targets
  unsigned controlMask;               // OR of (1<<c) over controls
  unsigned targets[MSV_MAX_TARGETS];  // matrix bit b -> targets[b]
};
static_assert(sizeof(GateKQ) == 32, "GateKQ layout must match MSL");
// applykq below hard-codes `targets[4]` and `in[16]` (= 2^MSV_MAX_TARGETS).
static_assert(MSV_MAX_TARGETS == 4, "update applykq's arrays in kMetalSource");

struct PauliParams {
  unsigned dim;        // 2^n
  unsigned perThread;  // amplitudes each thread sums
  unsigned slots;      // simdgroups in the x extent = partial sums per term
  unsigned firstBatch; // first 8-term batch of this dispatch
};
static_assert(sizeof(PauliParams) == 16, "PauliParams layout must match MSL");
// pauli_expect hard-codes 8 accumulators per thread.
constexpr size_t kPauliBatch = 8;
// 32-bit amplitude indices in every kernel.
static_assert(MSV_MAX_QUBITS <= 31, "kernels index amplitudes with uint");

const char *kMetalSource = R"MSL(
#include <metal_stdlib>
using namespace metal;

inline float2 cmul(float2 a, float2 b) {
    return float2(a.x*b.x - a.y*b.y, a.x*b.y + a.y*b.x);
}

struct Gate1Q {
    float2 u00; float2 u01; float2 u10; float2 u11;
    uint targetMask;
    uint lowMask;
    uint controlMask;
    uint pad;
};

// One thread per amplitude pair. gid in [0, 2^(n-1)); insert a 0 bit at the
// target position to form b0, set that bit for b1.
kernel void apply1q(device float2*   psi [[buffer(0)]],
                    constant Gate1Q&  g  [[buffer(1)]],
                    uint gid             [[thread_position_in_grid]]) {
    uint low  = gid & g.lowMask;
    uint high = (gid & ~g.lowMask) << 1;
    uint b0   = high | low;
    if ((b0 & g.controlMask) != g.controlMask) return;
    uint b1 = b0 | g.targetMask;
    float2 a0 = psi[b0];
    float2 a1 = psi[b1];
    psi[b0] = cmul(g.u00, a0) + cmul(g.u01, a1);
    psi[b1] = cmul(g.u10, a0) + cmul(g.u11, a1);
}

struct Gate2Q {
    float2 m[16];
    uint a; uint b;
    uint t0mask; uint t1mask;
    uint controlMask;
    uint pad;
};

// Specialized 2-target kernel: O(1) amplitude addressing via direct bit
// insertion (no per-thread loop over qubits). One thread per group of 4.
kernel void apply2q(device float2*   psi [[buffer(0)]],
                    constant Gate2Q&  g  [[buffer(1)]],
                    uint gid             [[thread_position_in_grid]]) {
    uint a = g.a, b = g.b;
    uint low  = gid & ((1u << a) - 1u);
    uint mid  = (gid >> a) & ((1u << (b - 1u - a)) - 1u);
    uint high = gid >> (b - 1u);
    uint base = (high << (b + 1u)) | (mid << (a + 1u)) | low;
    if ((base & g.controlMask) != g.controlMask) return;
    uint i0 = base, i1 = base | g.t0mask, i2 = base | g.t1mask;
    uint i3 = base | g.t0mask | g.t1mask;
    float2 x0 = psi[i0], x1 = psi[i1], x2 = psi[i2], x3 = psi[i3];
    psi[i0] = cmul(g.m[0],x0)+cmul(g.m[1],x1)+cmul(g.m[2],x2)+cmul(g.m[3],x3);
    psi[i1] = cmul(g.m[4],x0)+cmul(g.m[5],x1)+cmul(g.m[6],x2)+cmul(g.m[7],x3);
    psi[i2] = cmul(g.m[8],x0)+cmul(g.m[9],x1)+cmul(g.m[10],x2)+cmul(g.m[11],x3);
    psi[i3] = cmul(g.m[12],x0)+cmul(g.m[13],x1)+cmul(g.m[14],x2)+cmul(g.m[15],x3);
}

struct GateKQ {
    uint n;
    uint k;
    uint targetMask;
    uint controlMask;
    uint targets[4];
};

// One thread per group of 2^k amplitudes. gid in [0, 2^(n-k)). Distribute gid's
// bits into the non-target positions (target bits = 0) to get the group base,
// gather the 2^k amplitudes, apply the dense matrix, scatter back.
kernel void applykq(device float2*        psi [[buffer(0)]],
                    constant GateKQ&      g   [[buffer(1)]],
                    constant float2*      mat [[buffer(2)]],
                    uint gid                  [[thread_position_in_grid]]) {
    uint base = 0u, gpos = 0u;
    for (uint p = 0u; p < g.n; ++p) {
        if (g.targetMask & (1u << p)) continue;
        if (gid & (1u << gpos)) base |= (1u << p);
        ++gpos;
    }
    if ((base & g.controlMask) != g.controlMask) return;

    uint ns = 1u << g.k;
    float2 in[16];
    for (uint c = 0u; c < ns; ++c) {
        uint idx = base;
        for (uint b = 0u; b < g.k; ++b)
            if (c & (1u << b)) idx |= (1u << g.targets[b]);
        in[c] = psi[idx];
    }
    for (uint r = 0u; r < ns; ++r) {
        float2 acc = float2(0.0f, 0.0f);
        for (uint c = 0u; c < ns; ++c)
            acc += cmul(mat[r * ns + c], in[c]);
        uint idx = base;
        for (uint b = 0u; b < g.k; ++b)
            if (r & (1u << b)) idx |= (1u << g.targets[b]);
        psi[idx] = acc;
    }
}

struct PauliParams {
    uint dim;
    uint perThread;
    uint slots;
    uint firstBatch;
};

// Pauli-string expectation values (see msv_pauli_expectations). Grid x covers
// the state, grid y is the 8-term batch. Each simdgroup owns a contiguous
// block of W*perThread amplitudes, lane-interleaved so loads coalesce; each
// thread keeps one fp32 accumulator per term, simd_sum combines the lanes and
// lane 0 writes one partial per (batch, simdgroup, term) for the host to sum
// in double. terms[t] = (x mask, z mask, 1 = take the imaginary part, 0);
// the host applies the remaining sign of i^nY.
kernel void pauli_expect(device const float2*  psi      [[buffer(0)]],
                         device const uint4*   terms    [[buffer(1)]],
                         device float*         partial  [[buffer(2)]],
                         constant PauliParams& p        [[buffer(3)]],
                         uint2 tg       [[threadgroup_position_in_grid]],
                         uint  sgInTg   [[simdgroup_index_in_threadgroup]],
                         uint  sgPerTg  [[simdgroups_per_threadgroup]],
                         uint  lane     [[thread_index_in_simdgroup]],
                         uint  W        [[threads_per_simdgroup]]) {
    const uint sg = tg.x * sgPerTg + sgInTg;
    uint tx[8], tz[8];
    bool tim[8];
    float acc[8];
    for (uint t = 0u; t < 8u; ++t) {
        const uint4 term = terms[(p.firstBatch + tg.y) * 8u + t];
        tx[t] = term.x;
        tz[t] = term.y;
        tim[t] = term.z != 0u;
        acc[t] = 0.0f;
    }
    const uint base = sg * W * p.perThread + lane;
    for (uint j = 0u; j < p.perThread; ++j) {
        const uint idx = base + j * W;
        if (idx >= p.dim) break;
        const float2 a = psi[idx];
        for (uint t = 0u; t < 8u; ++t) {
            // a * conj(b): real part a.b, imaginary part a.y*b.x - a.x*b.y.
            const float2 b = tx[t] != 0u ? psi[idx ^ tx[t]] : a;
            const float v = tim[t] ? (a.y * b.x - a.x * b.y)
                                   : (a.x * b.x + a.y * b.y);
            acc[t] += (popcount(idx & tz[t]) & 1u) ? -v : v;
        }
    }
    for (uint t = 0u; t < 8u; ++t) {
        const float s = simd_sum(acc[t]);
        if (lane == 0u) // sg < p.slots always: slots = every simdgroup
            partial[(tg.y * p.slots + sg) * 8u + t] = s;
    }
}
)MSL";

thread_local std::string gLastError;

void setError(const char *fmt, ...) {
  char buf[512];
  va_list ap;
  va_start(ap, fmt);
  std::vsnprintf(buf, sizeof(buf), fmt, ap);
  va_end(ap);
  gLastError = buf;
}

// Process-wide Metal objects: compiling the MSL source and building pipelines
// costs far more than a small circuit, so it happens once, not per state.
struct MetalContext {
  id<MTLDevice> device = nil;
  id<MTLCommandQueue> queue = nil;
  id<MTLComputePipelineState> pso1q = nil, pso2q = nil, psoKq = nil,
                              psoPauli = nil;
  std::string error; // non-empty if initialization failed
};

const MetalContext &context() {
  // Intentionally leaked: no teardown ordering issues at process exit.
  static const MetalContext *ctx = [] {
    auto *c = new MetalContext();
    @autoreleasepool {
      c->device = MTLCreateSystemDefaultDevice();
      if (!c->device) {
        c->error = "no Metal device available";
        return c;
      }
      c->queue = [c->device newCommandQueue];
      NSError *err = nil;
      id<MTLLibrary> lib = [c->device
          newLibraryWithSource:[NSString stringWithUTF8String:kMetalSource]
                       options:nil
                         error:&err];
      if (!lib) {
        c->error = std::string("Metal shader compile failed: ") +
                   err.localizedDescription.UTF8String;
        return c;
      }
      auto makePso = [&](NSString *fn) -> id<MTLComputePipelineState> {
        NSError *e = nil;
        id<MTLComputePipelineState> p = [c->device
            newComputePipelineStateWithFunction:[lib newFunctionWithName:fn]
                                          error:&e];
        if (!p && c->error.empty())
          c->error = std::string("Metal pipeline ") + fn.UTF8String +
                     " failed: " + e.localizedDescription.UTF8String;
        return p;
      };
      c->pso1q = makePso(@"apply1q");
      c->pso2q = makePso(@"apply2q");
      c->psoKq = makePso(@"applykq");
      c->psoPauli = makePso(@"pauli_expect");
      if (!c->queue && c->error.empty())
        c->error = "could not create a Metal command queue";
    }
    return c;
  }();
  return *ctx;
}

double gib(size_t bytes) { return double(bytes) / double(1ull << 30); }

// ---- Buffer reuse --------------------------------------------------------------
// Every CUDA-Q execution allocates the state, runs, and frees it. For a fresh
// buffer the OS must fault in and zero every page and the GPU must map it on
// first use, which costs far more than the zeroing itself: ~230 ms for 28
// qubits, against ~25 ms to reset and use a buffer that is already mapped. So
// freed buffers go to small process-wide pools and are handed out again:
//   * statevectors of 17+ qubits: the two most recently freed, each reused
//     only for a state of exactly its size (two, so that a loop alternating
//     two sizes, e.g. a kernel with and without an ancilla, still hits);
//   * expectation-value scratch: the most recently freed, reused for any
//     request it covers.
// While idle a buffer is marked purgeable (volatile): under memory pressure
// macOS may reclaim it instead of swapping, and a reclaimed buffer is simply
// not reused. Before a fresh statevector is allocated, idle ones are released
// (oldest first) until idle + new fits in the GPU's recommended working set;
// if the allocation fails anyway, every idle buffer is released and it is
// retried once. MSV_DISABLE_BUFFER_CACHE=1 frees every buffer immediately.
constexpr size_t kMinCachedBytes = size_t(1) << 20; // a 17-qubit statevector
constexpr size_t kMaxIdleStates = 2;

bool bufferCacheEnabled() {
  return !msv_env_flag("MSV_DISABLE_BUFFER_CACHE");
}

// Idle MTLBuffers kept for reuse; thread-safe. Buffers released by a call are
// released outside the lock.
class BufferPool {
public:
  BufferPool(size_t maxIdle, bool exactSize)
      : maxIdle_(maxIdle), exactSize_(exactSize) {}

  // The most recent idle buffer exactly `bytes` long (exact-size pool) or at
  // least that long, or nil.
  id<MTLBuffer> take(size_t bytes) {
    id<MTLBuffer> buf = nil;
    {
      std::lock_guard<std::mutex> lock(mutex_);
      for (size_t i = idle_.size(); i-- > 0;) {
        const size_t len = idle_[i].length;
        if (exactSize_ ? len == bytes : len >= bytes) {
          buf = idle_[i];
          idle_.erase(idle_.begin() + i);
          break;
        }
      }
    }
    // Setting NonVolatile returns the previous state; Empty means the OS
    // reclaimed the memory while the buffer sat idle (released here).
    if (buf && [buf setPurgeableState:MTLPurgeableStateNonVolatile] ==
                   MTLPurgeableStateEmpty)
      return nil;
    if (buf)
      ++hits_;
    return buf;
  }

  // Keep `buf` (idle: nothing uses it and its GPU work has finished) as the
  // most recent entry, releasing the oldest beyond the pool's size.
  void give(id<MTLBuffer> buf) {
    [buf setPurgeableState:MTLPurgeableStateVolatile];
    std::vector<id<MTLBuffer>> released;
    std::lock_guard<std::mutex> lock(mutex_);
    idle_.push_back(buf);
    while (idle_.size() > maxIdle_) {
      released.push_back(idle_.front());
      idle_.erase(idle_.begin());
    }
    // (`lock` is destroyed before `released`: declared after it.)
  }

  // Release idle buffers, oldest first, until at most `keepBytes` remain.
  void trim(size_t keepBytes) {
    std::vector<id<MTLBuffer>> released;
    std::lock_guard<std::mutex> lock(mutex_);
    size_t total = 0;
    for (id<MTLBuffer> b : idle_)
      total += b.length;
    while (!idle_.empty() && total > keepBytes) {
      total -= idle_.front().length;
      released.push_back(idle_.front());
      idle_.erase(idle_.begin());
    }
  }

  void clear() { trim(0); }
  unsigned long long hits() const { return hits_; }

private:
  const size_t maxIdle_;
  const bool exactSize_;
  std::mutex mutex_;
  std::vector<id<MTLBuffer>> idle_; // oldest first
  std::atomic<unsigned long long> hits_{0};
};

// Leaked, like the context: no teardown ordering issues at process exit.
BufferPool &statePool() {
  static BufferPool *pool = new BufferPool(kMaxIdleStates, /*exactSize=*/true);
  return *pool;
}
BufferPool &scratchPool() {
  static BufferPool *pool = new BufferPool(1, /*exactSize=*/false);
  return *pool;
}
std::atomic<unsigned long long> gStateAllocations{0};

void releaseIdleBuffers() {
  statePool().clear();
  scratchPool().clear();
}

// A fresh shared buffer, retried once after releasing every idle buffer: the
// memory they hold may be exactly what is missing.
id<MTLBuffer> newSharedBuffer(const MetalContext &ctx, size_t bytes) {
  id<MTLBuffer> buf = [ctx.device newBufferWithLength:bytes
                                              options:MTLResourceStorageModeShared];
  if (!buf) {
    releaseIdleBuffers();
    buf = [ctx.device newBufferWithLength:bytes
                                  options:MTLResourceStorageModeShared];
  }
  return buf;
}

// Whether a 2^n statevector is allowed on this device; sets gLastError if not.
bool checkQubits(const MetalContext &ctx, int n) {
  if (n < 1 || n > MSV_MAX_QUBITS) {
    setError("%d qubits is outside the supported range 1..%d", n,
             MSV_MAX_QUBITS);
    return false;
  }
  const size_t bytes = sizeof(C2) << n;
  const size_t limit = ctx.device.maxBufferLength;
  if (bytes > limit) {
    setError("a %d-qubit statevector needs %.2f GiB, but this GPU's "
             "maxBufferLength is %.2f GiB",
             n, gib(bytes), gib(limit));
    return false;
  }
  return true;
}

// A buffer for a 2^n statevector, or nil with gLastError set. Its contents
// are undefined (it may be a reused one): callers zero it (zeroState) or
// overwrite all of it (msv_grow_to).
id<MTLBuffer> allocState(const MetalContext &ctx, int n) {
  if (!checkQubits(ctx, n))
    return nil;
  const size_t bytes = sizeof(C2) << n;
  if (!bufferCacheEnabled()) {
    releaseIdleBuffers(); // left over from before it was disabled
  } else if (bytes >= kMinCachedBytes) {
    if (id<MTLBuffer> buf = statePool().take(bytes))
      return buf;
    const size_t budget = ctx.device.recommendedMaxWorkingSetSize;
    statePool().trim(budget > bytes ? budget - bytes : 0);
  }
  ++gStateAllocations;
  id<MTLBuffer> buf = newSharedBuffer(ctx, bytes);
  if (!buf)
    setError("could not allocate %.2f GiB for a %d-qubit statevector",
             gib(bytes), n);
  return buf;
}

// Scratch of at least `bytes` (a reused one may be larger), or nil.
id<MTLBuffer> takeScratch(const MetalContext &ctx, size_t bytes) {
  if (bufferCacheEnabled())
    if (id<MTLBuffer> buf = scratchPool().take(bytes))
      return buf;
  return newSharedBuffer(ctx, bytes);
}

void giveScratch(id<MTLBuffer> buf) {
  if (bufferCacheEnabled())
    scratchPool().give(buf);
}

NSUInteger threadsPerGroup(id<MTLComputePipelineState> pso, NSUInteger grid) {
  NSUInteger t = pso.maxTotalThreadsPerThreadgroup;
  if (t > grid) t = grid;
  return t == 0 ? 1 : t;
}

} // namespace

struct MSVState {
  const MetalContext *ctx = nullptr;
  int nQubits = 0;
  size_t dim = 0; // 2^n
  id<MTLBuffer> psi = nil;

  // Lazy execution state (nil when nothing is pending).
  id<MTLCommandBuffer> pending = nil;
  id<MTLComputeCommandEncoder> encoder = nil;

  id<MTLComputeCommandEncoder> encoderFor(id<MTLComputePipelineState> pso) {
    if (encoder == nil) {
      pending = [ctx->queue commandBuffer];
      // Default computeCommandEncoder is serial: dispatches run in order with
      // implicit memory barriers between them — exactly gate-to-gate ordering.
      encoder = [pending computeCommandEncoder];
      [encoder setBuffer:psi offset:0 atIndex:0];
    }
    [encoder setComputePipelineState:pso];
    return encoder;
  }

  void dispatch(id<MTLComputePipelineState> pso, NSUInteger grid) {
    [encoder dispatchThreads:MTLSizeMake(grid, 1, 1)
        threadsPerThreadgroup:MTLSizeMake(threadsPerGroup(pso, grid), 1, 1)];
  }

  // Commit pending work and wait. Returns false if the GPU reported an error,
  // in which case the statevector contents are undefined.
  bool flush() {
    if (encoder == nil) return true;
    [encoder endEncoding];
    [pending commit];
    [pending waitUntilCompleted];
    const bool ok = pending.status != MTLCommandBufferStatusError;
    if (!ok)
      setError("GPU command buffer failed (%s); the simulator state is invalid",
               pending.error ? pending.error.localizedDescription.UTF8String
                             : "unknown error");
    encoder = nil;
    pending = nil;
    return ok;
  }

  void zeroState() {
    flush(); // the state is being overwritten, so a failed batch is moot
    C2 *p = (C2 *)psi.contents;
    std::memset(p, 0, dim * sizeof(C2));
    p[0].x = 1.0f; // |0...0>
  }

  // Validate a gate's qubits; returns their OR-ed bit mask, or -1 on error.
  long long maskOf(const int *qubits, int count, const char *what) const {
    unsigned mask = 0;
    for (int i = 0; i < count; ++i) {
      const int q = qubits[i];
      if (q < 0 || q >= nQubits) {
        setError("%s qubit %d is out of range for %d qubits", what, q, nQubits);
        return -1;
      }
      mask |= 1u << q;
    }
    return mask;
  }
};

extern "C" {

MSVState *msv_create(int nQubits) {
  @autoreleasepool {
    const MetalContext &ctx = context();
    if (!ctx.error.empty()) {
      setError("%s", ctx.error.c_str());
      return nullptr;
    }
    id<MTLBuffer> buf = allocState(ctx, nQubits);
    if (!buf) return nullptr;
    MSVState *s = new MSVState();
    s->ctx = &ctx;
    s->nQubits = nQubits;
    s->dim = (size_t)1 << nQubits;
    s->psi = buf;
    s->zeroState();
    return s;
  }
}

void msv_destroy(MSVState *s) {
  if (!s) return;
  @autoreleasepool {
    // flush() waits for the GPU, so the buffer is idle before it is kept for
    // reuse; after a GPU error it is not reused.
    if (s->flush() && s->psi.length >= kMinCachedBytes && bufferCacheEnabled())
      statePool().give(s->psi);
    delete s; // ARC releases the buffer unless the pool kept it
  }
}

void msv_reset(MSVState *s) {
  if (!s) return;
  @autoreleasepool { s->zeroState(); }
}

int msv_apply(MSVState *s, const float *m, int k, const int *targets,
              const int *controls, int nControls) {
  if (!s) return -1;
  if (k < 1 || k > MSV_MAX_TARGETS) {
    setError("%d-target gates are not supported (max %d)", k, MSV_MAX_TARGETS);
    return -1;
  }
  const long long tmask = s->maskOf(targets, k, "target");
  const long long cmask = s->maskOf(controls, nControls, "control");
  if (tmask < 0 || cmask < 0) return -1;
  if (__builtin_popcountll(tmask) != k || (tmask & cmask)) {
    setError("gate qubits must be distinct");
    return -1;
  }
  @autoreleasepool {
    const MetalContext &ctx = *s->ctx;
    if (k == 1) {
      Gate1Q g;
      g.u00 = {m[0], m[1]};
      g.u01 = {m[2], m[3]};
      g.u10 = {m[4], m[5]};
      g.u11 = {m[6], m[7]};
      g.targetMask = (unsigned)tmask;
      g.lowMask = g.targetMask - 1u;
      g.controlMask = (unsigned)cmask;
      g.pad = 0;
      [s->encoderFor(ctx.pso1q) setBytes:&g length:sizeof(g) atIndex:1];
      s->dispatch(ctx.pso1q, s->dim >> 1); // 2^(n-1) pairs
    } else if (k == 2) {
      Gate2Q g;
      for (int i = 0; i < 16; ++i) g.m[i] = {m[2 * i], m[2 * i + 1]};
      const int t0 = targets[0], t1 = targets[1];
      g.t0mask = 1u << t0;
      g.t1mask = 1u << t1;
      g.a = (unsigned)(t0 < t1 ? t0 : t1);
      g.b = (unsigned)(t0 < t1 ? t1 : t0);
      g.controlMask = (unsigned)cmask;
      g.pad = 0;
      [s->encoderFor(ctx.pso2q) setBytes:&g length:sizeof(g) atIndex:1];
      s->dispatch(ctx.pso2q, s->dim >> 2); // 2^(n-2) groups
    } else {
      GateKQ g{};
      g.n = (unsigned)s->nQubits;
      g.k = (unsigned)k;
      for (int i = 0; i < k; ++i) g.targets[i] = (unsigned)targets[i];
      g.targetMask = (unsigned)tmask;
      g.controlMask = (unsigned)cmask;
      const NSUInteger ns = (NSUInteger)1 << k; // 2^k
      id<MTLComputeCommandEncoder> enc = s->encoderFor(ctx.psoKq);
      [enc setBytes:&g length:sizeof(g) atIndex:1];
      [enc setBytes:m length:ns * ns * sizeof(C2) atIndex:2];
      s->dispatch(ctx.psoKq, s->dim >> k); // 2^(n-k) groups
    }
  }
  return 0;
}

int msv_synchronize(MSVState *s) {
  if (!s) return -1;
  @autoreleasepool { return s->flush() ? 0 : -1; }
}

int msv_grow_to(MSVState *s, int newNQubits) {
  if (!s || newNQubits < s->nQubits) {
    setError("cannot shrink the statevector");
    return -1;
  }
  if (newNQubits == s->nQubits) return 0;
  @autoreleasepool {
    if (!s->flush()) return -1;
    id<MTLBuffer> nb = allocState(*s->ctx, newNQubits);
    if (!nb) return -1;
    // Old amplitudes map 1:1 into the low half (new high bits = 0 = |0>).
    char *dst = (char *)nb.contents;
    const size_t oldBytes = s->dim * sizeof(C2);
    std::memcpy(dst, s->psi.contents, oldBytes);
    std::memset(dst + oldBytes, 0, nb.length - oldBytes);
    s->psi = nb;
    s->dim = (size_t)1 << newNQubits;
    s->nQubits = newNQubits;
  }
  return 0;
}

void *msv_data(MSVState *s) {
  if (!s) return nullptr;
  @autoreleasepool { return s->flush() ? s->psi.contents : nullptr; }
}

int msv_pauli_expectations(MSVState *s, const unsigned *xmask,
                           const unsigned *zmask, size_t nTerms, double *out) {
  if (!s) return -1;
  if (nTerms == 0) return 0;
  const unsigned outside = ~(unsigned)(s->dim - 1); // dim <= 2^31
  for (size_t t = 0; t < nTerms; ++t)
    if ((xmask[t] | zmask[t]) & outside) {
      setError("Pauli term %zu acts on a qubit outside the %d-qubit state", t,
               s->nQubits);
      return -1;
    }
  @autoreleasepool {
    if (!s->flush()) return -1; // pending gates first
    const MetalContext &ctx = *s->ctx;
    id<MTLComputePipelineState> pso = ctx.psoPauli;

    // Terms, padded to whole batches with (0, 0) = <psi|psi> (ignored).
    const size_t nBatches = (nTerms + kPauliBatch - 1) / kPauliBatch;
    std::vector<unsigned> packed(nBatches * kPauliBatch * 4, 0u);
    for (size_t t = 0; t < nTerms; ++t) {
      packed[4 * t] = xmask[t];
      packed[4 * t + 1] = zmask[t];
      packed[4 * t + 2] = __builtin_popcount(xmask[t] & zmask[t]) & 1u;
    }

    // Geometry: enough threads to fill the GPU, each summing up to 64
    // amplitudes so the fp32 partials stay short.
    const NSUInteger W = pso.threadExecutionWidth;
    NSUInteger tg = pso.maxTotalThreadsPerThreadgroup < 256
                        ? pso.maxTotalThreadsPerThreadgroup : 256;
    tg -= tg % W;
    size_t perThread = s->dim >> 15;
    perThread = perThread < 1 ? 1 : (perThread > 64 ? 64 : perThread);
    const size_t threads = (s->dim + perThread - 1) / perThread;
    const size_t numTG = (threads + tg - 1) / tg;
    const size_t slots = numTG * (tg / W);

    // One partial per (batch, simdgroup, term); bound the buffer so huge
    // states with many terms run in several dispatches instead.
    // MSV_PAULI_SCRATCH_BYTES lowers the bound so tests reach that path.
    const size_t bytesPerBatch = slots * kPauliBatch * sizeof(float);
    size_t maxPartialBytes = size_t(64) << 20;
    if (const char *e = std::getenv("MSV_PAULI_SCRATCH_BYTES"))
      if (const long long v = std::atoll(e); v > 0)
        maxPartialBytes = (size_t)v;
    size_t chunk = maxPartialBytes / bytesPerBatch;
    chunk = chunk < 1 ? 1 : (chunk > nBatches ? nBatches : chunk);

    // One scratch buffer: the packed terms, then (256-byte aligned) the
    // partials. Reused across calls (see BufferPool); every byte read is
    // written first, so a reused buffer's old contents do not matter.
    const size_t termBytes = packed.size() * sizeof(unsigned);
    const size_t partialOffset = (termBytes + 255) & ~size_t(255);
    const size_t scratchBytes = partialOffset + chunk * bytesPerBatch;
    id<MTLBuffer> scratch = takeScratch(ctx, scratchBytes);
    if (!scratch) {
      setError("could not allocate %.2f GiB of expectation-value scratch",
               gib(scratchBytes));
      return -1;
    }
    std::memcpy(scratch.contents, packed.data(), termBytes);

    for (size_t first = 0; first < nBatches; first += chunk) {
      const size_t nb = nBatches - first < chunk ? nBatches - first : chunk;
      const PauliParams prm{(unsigned)s->dim, (unsigned)perThread,
                            (unsigned)slots, (unsigned)first};
      id<MTLComputeCommandEncoder> enc = s->encoderFor(pso);
      [enc setBuffer:scratch offset:0 atIndex:1];
      [enc setBuffer:scratch offset:partialOffset atIndex:2];
      [enc setBytes:&prm length:sizeof(prm) atIndex:3];
      [enc dispatchThreadgroups:MTLSizeMake(numTG, nb, 1)
          threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
      if (!s->flush()) return -1;

      // Sum each term's partials in double, then apply Re(i^nY * S): the
      // kernel produced Re S (nY even) or Im S (nY odd).
      const float *P =
          (const float *)((const char *)scratch.contents + partialOffset);
      dispatch_apply(nb * kPauliBatch, DISPATCH_APPLY_AUTO, ^(size_t i) {
        const size_t b = i / kPauliBatch, t = i % kPauliBatch;
        const size_t term = (first + b) * kPauliBatch + t;
        if (term >= nTerms) return;
        const float *col = P + b * slots * kPauliBatch + t;
        double sum = 0.0;
        for (size_t g = 0; g < slots; ++g)
          sum += col[g * kPauliBatch];
        const unsigned nY = __builtin_popcount(xmask[term] & zmask[term]) & 3u;
        out[term] = (nY == 1u || nY == 2u) ? -sum : sum;
      });
    }
    giveScratch(scratch);
  }
  return 0;
}

int msv_check_qubits(int nQubits) {
  @autoreleasepool {
    const MetalContext &ctx = context();
    if (!ctx.error.empty()) {
      setError("%s", ctx.error.c_str());
      return -1;
    }
    return checkQubits(ctx, nQubits) ? 0 : -1;
  }
}

int msv_env_flag(const char *name) {
  const char *v = std::getenv(name);
  if (!v)
    return 0;
  for (const char *off : {"", "0", "false", "no", "off"})
    if (strcasecmp(v, off) == 0)
      return 0;
  return 1;
}

void msv_buffer_stats(unsigned long long *stateAllocations,
                      unsigned long long *stateReuses,
                      unsigned long long *scratchReuses) {
  if (stateAllocations) *stateAllocations = gStateAllocations;
  if (stateReuses) *stateReuses = statePool().hits();
  if (scratchReuses) *scratchReuses = scratchPool().hits();
}

int msv_num_qubits(const MSVState *s) { return s ? s->nQubits : 0; }

const char *msv_last_error(void) { return gLastError.c_str(); }

} // extern "C"
