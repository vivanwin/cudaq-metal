// metal_sv.mm — naive Metal statevector engine (Objective-C++).
//
// State layout: one MTLBuffer of float2[2^n] in MTLStorageModeShared (unified
// memory, no host<->device copies). Amplitude index is little-endian in the
// qubit label: qubit q occupies bit q of the index.
//
// Two compute kernels:
//   apply1q  — 2x2 matrix on one target qubit, gated by a control mask. Each
//              thread owns one disjoint amplitude pair. Fast path for the
//              common single-target gates (H/X/Z/rotations, CX/CCX/CZ).
//   applykq  — general 2^k x 2^k matrix on k target qubits (k<=4), gated by a
//              control mask. Each thread owns one disjoint group of 2^k
//              amplitudes, gathers them, does the matrix-vector product, and
//              scatters back. Handles SWAP/iSWAP/generic 2-qubit (e.g. QV SU(4)).
//
// Execution is lazy: dispatches accumulate in one open command buffer using a
// *serial* compute encoder (Metal orders serial-encoder dispatches with
// implicit barriers, so gate g+1 sees gate g's writes). We only commit + wait
// on msv_synchronize()/msv_get_state()/msv_data(). This amortizes launch/commit
// latency, which is what otherwise sinks the GPU on shallow circuits.

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <cstring>
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

constexpr int kMaxKTargets = 4; // applykq supports up to 4 target qubits
struct GateKQ {
  unsigned n;                       // total qubits
  unsigned k;                       // number of targets
  unsigned targetMask;              // OR of (1<<t) over targets
  unsigned controlMask;             // OR of (1<<c) over controls
  unsigned targets[kMaxKTargets];   // original order: matrix bit b -> targets[b]
};
static_assert(sizeof(GateKQ) == 32, "GateKQ layout must match MSL");

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

} // namespace

struct MSVState {
  int nQubits = 0;
  size_t dim = 0; // 2^n
  id<MTLDevice> device = nil;
  id<MTLCommandQueue> queue = nil;
  id<MTLComputePipelineState> pso1q = nil;
  id<MTLComputePipelineState> pso2q = nil;
  id<MTLComputePipelineState> psoKq = nil;
  id<MTLBuffer> psi = nil;
  std::string deviceName;

  // Lazy execution state (nil when nothing is pending).
  id<MTLCommandBuffer> pending = nil;
  id<MTLComputeCommandEncoder> encoder = nil;

  void openEncoderIfNeeded() {
    if (encoder != nil) return;
    pending = [queue commandBuffer];
    // Default computeCommandEncoder is serial: dispatches run in order with
    // implicit memory barriers between them — exactly gate-to-gate ordering.
    encoder = [pending computeCommandEncoder];
    [encoder setBuffer:psi offset:0 atIndex:0];
  }

  void flush() {
    if (encoder == nil) return;
    [encoder endEncoding];
    [pending commit];
    [pending waitUntilCompleted];
    encoder = nil;
    pending = nil;
  }

  void zeroState() {
    flush();
    C2 *p = (C2 *)psi.contents;
    std::memset(p, 0, dim * sizeof(C2));
    p[0].x = 1.0f; // |0...0>
  }
};

extern "C" {

MSVState *msv_create(int nQubits) {
  @autoreleasepool {
    if (nQubits < 1 || nQubits > 31) return nullptr;
    id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
    if (!dev) return nullptr;

    MSVState *s = new MSVState();
    s->nQubits = nQubits;
    s->dim = (size_t)1 << nQubits;
    s->device = dev;
    s->deviceName = std::string([[dev name] UTF8String]);
    s->queue = [dev newCommandQueue];

    NSError *err = nil;
    id<MTLLibrary> lib =
        [dev newLibraryWithSource:[NSString stringWithUTF8String:kMetalSource]
                          options:nil
                            error:&err];
    if (!lib) {
      NSLog(@"metal_sv: shader compile failed: %@", err);
      delete s;
      return nullptr;
    }
    auto makePso = [&](NSString *fn) -> id<MTLComputePipelineState> {
      id<MTLFunction> f = [lib newFunctionWithName:fn];
      NSError *e = nil;
      id<MTLComputePipelineState> p =
          [dev newComputePipelineStateWithFunction:f error:&e];
      if (!p) NSLog(@"metal_sv: pipeline %@ failed: %@", fn, e);
      return p;
    };
    s->pso1q = makePso(@"apply1q");
    s->pso2q = makePso(@"apply2q");
    s->psoKq = makePso(@"applykq");
    if (!s->pso1q || !s->pso2q || !s->psoKq) {
      delete s;
      return nullptr;
    }

    s->psi = [dev newBufferWithLength:s->dim * sizeof(C2)
                              options:MTLResourceStorageModeShared];
    if (!s->psi) {
      NSLog(@"metal_sv: buffer alloc failed for %d qubits", nQubits);
      delete s;
      return nullptr;
    }
    s->zeroState();
    return s;
  }
}

void msv_destroy(MSVState *s) {
  if (!s) return;
  @autoreleasepool {
    s->flush();
    s->psi = nil;
    s->pso1q = nil;
    s->pso2q = nil;
    s->psoKq = nil;
    s->queue = nil;
    s->device = nil;
    delete s;
  }
}

void msv_reset(MSVState *s) {
  if (!s) return;
  @autoreleasepool { s->zeroState(); }
}

static NSUInteger tptgFor(id<MTLComputePipelineState> pso, NSUInteger grid) {
  NSUInteger t = pso.maxTotalThreadsPerThreadgroup;
  if (t > grid) t = grid;
  if (t == 0) t = 1;
  return t;
}

void msv_apply_1q(MSVState *s, const float *m, int target, const int *controls,
                  int nControls) {
  if (!s) return;
  @autoreleasepool {
    Gate1Q g;
    g.u00 = {m[0], m[1]};
    g.u01 = {m[2], m[3]};
    g.u10 = {m[4], m[5]};
    g.u11 = {m[6], m[7]};
    g.targetMask = 1u << target;
    g.lowMask = g.targetMask - 1u;
    unsigned cmask = 0;
    for (int i = 0; i < nControls; ++i) cmask |= (1u << controls[i]);
    g.controlMask = cmask;
    g.pad = 0;

    s->openEncoderIfNeeded();
    [s->encoder setComputePipelineState:s->pso1q];
    [s->encoder setBytes:&g length:sizeof(g) atIndex:1];

    NSUInteger grid = (NSUInteger)(s->dim >> 1); // 2^(n-1) pairs
    [s->encoder dispatchThreads:MTLSizeMake(grid, 1, 1)
          threadsPerThreadgroup:MTLSizeMake(tptgFor(s->pso1q, grid), 1, 1)];
  }
}

void msv_apply_kq(MSVState *s, const float *mat, int k, const int *targets,
                  int nTargets, const int *controls, int nControls) {
  if (!s) return;
  if (k != nTargets || k < 1 || k > kMaxKTargets) {
    NSLog(@"metal_sv: msv_apply_kq unsupported k=%d (max %d)", k, kMaxKTargets);
    return;
  }
  if (k == 1) { // route to the fast path
    msv_apply_1q(s, mat, targets[0], controls, nControls);
    return;
  }
  if (k == 2) { // specialized fast 2-qubit kernel
    @autoreleasepool {
      Gate2Q g;
      for (int i = 0; i < 16; ++i) g.m[i] = {mat[2 * i], mat[2 * i + 1]};
      const int t0 = targets[0], t1 = targets[1];
      g.t0mask = 1u << t0;
      g.t1mask = 1u << t1;
      g.a = (unsigned)(t0 < t1 ? t0 : t1);
      g.b = (unsigned)(t0 < t1 ? t1 : t0);
      unsigned cmask = 0;
      for (int i = 0; i < nControls; ++i) cmask |= (1u << controls[i]);
      g.controlMask = cmask;
      g.pad = 0;

      s->openEncoderIfNeeded();
      [s->encoder setComputePipelineState:s->pso2q];
      [s->encoder setBytes:&g length:sizeof(g) atIndex:1];
      NSUInteger grid = (NSUInteger)(s->dim >> 2); // 2^(n-2) groups
      [s->encoder dispatchThreads:MTLSizeMake(grid, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(tptgFor(s->pso2q, grid), 1, 1)];
    }
    return;
  }
  @autoreleasepool {
    GateKQ g{};
    g.n = (unsigned)s->nQubits;
    g.k = (unsigned)k;
    unsigned tmask = 0;
    for (int i = 0; i < k; ++i) {
      g.targets[i] = (unsigned)targets[i];
      tmask |= (1u << targets[i]);
    }
    g.targetMask = tmask;
    unsigned cmask = 0;
    for (int i = 0; i < nControls; ++i) cmask |= (1u << controls[i]);
    g.controlMask = cmask;

    const NSUInteger ns = (NSUInteger)1 << k; // 2^k
    const NSUInteger matBytes = ns * ns * sizeof(C2);

    s->openEncoderIfNeeded();
    [s->encoder setComputePipelineState:s->psoKq];
    [s->encoder setBytes:&g length:sizeof(g) atIndex:1];
    [s->encoder setBytes:mat length:matBytes atIndex:2];

    NSUInteger grid = (NSUInteger)(s->dim >> k); // 2^(n-k) groups
    [s->encoder dispatchThreads:MTLSizeMake(grid, 1, 1)
          threadsPerThreadgroup:MTLSizeMake(tptgFor(s->psoKq, grid), 1, 1)];
  }
}

void msv_synchronize(MSVState *s) {
  if (!s) return;
  @autoreleasepool { s->flush(); }
}

void msv_get_state(MSVState *s, float *outReal, float *outImag) {
  if (!s) return;
  @autoreleasepool {
    s->flush();
    const C2 *p = (const C2 *)s->psi.contents;
    for (size_t i = 0; i < s->dim; ++i) {
      if (outReal) outReal[i] = p[i].x;
      if (outImag) outImag[i] = p[i].y;
    }
  }
}

void msv_grow_to(MSVState *s, int newNQubits) {
  if (!s || newNQubits <= s->nQubits) return;
  @autoreleasepool {
    s->flush();
    size_t newDim = (size_t)1 << newNQubits;
    id<MTLBuffer> nb = [s->device newBufferWithLength:newDim * sizeof(C2)
                                              options:MTLResourceStorageModeShared];
    C2 *dst = (C2 *)nb.contents;
    std::memset(dst, 0, newDim * sizeof(C2));
    // Old amplitudes map 1:1 into the low half (new high bits = 0 = |0>).
    std::memcpy(dst, s->psi.contents, s->dim * sizeof(C2));
    s->psi = nb;
    s->dim = newDim;
    s->nQubits = newNQubits;
  }
}

void *msv_data(MSVState *s) {
  if (!s) return nullptr;
  @autoreleasepool {
    s->flush();
    return s->psi.contents;
  }
}

int msv_num_qubits(const MSVState *s) { return s ? s->nQubits : 0; }

const char *msv_device_name(const MSVState *s) {
  return s ? s->deviceName.c_str() : "";
}

} // extern "C"
