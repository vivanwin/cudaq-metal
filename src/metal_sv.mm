// metal_sv.mm — naive Metal statevector engine (Objective-C++).
//
// State layout: one MTLBuffer of float2[2^n] in MTLStorageModeShared (unified
// memory, no host<->device copies). Amplitude index is little-endian in the
// qubit label: qubit q occupies bit q of the index.
//
// Three compute kernels:
//   apply1q  — 2x2 matrix on one target qubit, gated by a control mask. Each
//              thread owns one disjoint amplitude pair. Fast path for the
//              common single-target gates (H/X/Z/rotations, CX/CCX/CZ).
//   apply2q  — 4x4 matrix on two targets with O(1) bit-insertion addressing.
//              Handles SWAP/iSWAP/generic 2-qubit (e.g. QV SU(4)).
//   applykq  — general 2^k x 2^k matrix on k <= MSV_MAX_TARGETS targets. Each
//              thread owns one disjoint group of 2^k amplitudes, gathers them,
//              does the matrix-vector product, and scatters back.
//
// Execution is lazy: dispatches accumulate in one open command buffer using a
// *serial* compute encoder (Metal orders serial-encoder dispatches with
// implicit barriers, so gate g+1 sees gate g's writes). We only commit + wait
// on msv_synchronize()/msv_data(). This amortizes launch/commit latency, which
// is what otherwise sinks the GPU on shallow circuits.
//
// The device, queue and compiled pipelines are process-wide (built once on
// first use); each MSVState owns only its statevector buffer.

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <cstdarg>
#include <cstdio>
#include <cstring>
#include <string>

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
  id<MTLComputePipelineState> pso1q = nil, pso2q = nil, psoKq = nil;
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
      if (!c->queue && c->error.empty())
        c->error = "could not create a Metal command queue";
    }
    return c;
  }();
  return *ctx;
}

double gib(size_t bytes) { return double(bytes) / double(1ull << 30); }

// Allocate a zero-filled 2^n statevector, or nil with gLastError set.
id<MTLBuffer> allocState(const MetalContext &ctx, int n) {
  if (n < 1 || n > MSV_MAX_QUBITS) {
    setError("%d qubits is outside the supported range 1..%d", n,
             MSV_MAX_QUBITS);
    return nil;
  }
  const size_t bytes = sizeof(C2) << n;
  const size_t limit = ctx.device.maxBufferLength;
  if (bytes > limit) {
    setError("a %d-qubit statevector needs %.2f GiB, but this GPU's "
             "maxBufferLength is %.2f GiB",
             n, gib(bytes), gib(limit));
    return nil;
  }
  id<MTLBuffer> buf = [ctx.device newBufferWithLength:bytes
                                              options:MTLResourceStorageModeShared];
  if (!buf)
    setError("could not allocate %.2f GiB for a %d-qubit statevector",
             gib(bytes), n);
  return buf;
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
    s->flush();
    delete s; // ARC releases the buffer
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

int msv_num_qubits(const MSVState *s) { return s ? s->nQubits : 0; }

const char *msv_last_error(void) { return gLastError.c_str(); }

} // extern "C"
