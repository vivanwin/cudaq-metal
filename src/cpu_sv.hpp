// cpu_sv.hpp — naive CPU statevector reference, same algorithm & qubit
// convention as the Metal engine (little-endian amplitude index).
//
// Provides single-threaded and multi-threaded (GCD dispatch_apply, scales to
// all cores) single-target-with-controls gate application. This is a *fair*
// naive baseline for validating the Metal engine and for a first performance
// signal; the authoritative CPU baseline is CUDA-Q's qpp-cpu backend, compared
// separately once the NVQIR backend is wired in.
#pragma once
#include <complex>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <vector>
#include <dispatch/dispatch.h>

struct CpuSV {
  int nQubits;
  std::size_t dim;
  std::vector<std::complex<float>> psi;

  explicit CpuSV(int n) : nQubits(n), dim(std::size_t(1) << n), psi(dim) {
    reset();
  }
  void reset() {
    std::memset(psi.data(), 0, dim * sizeof(std::complex<float>));
    psi[0] = {1.0f, 0.0f};
  }

  // m: interleaved-complex 2x2 row-major {re00,im00,re01,im01,re10,im10,re11,im11}
  template <bool Parallel>
  void apply1q(const float *m, int target, const std::uint32_t controlMask) {
    const std::complex<float> u00(m[0], m[1]), u01(m[2], m[3]);
    const std::complex<float> u10(m[4], m[5]), u11(m[6], m[7]);
    const std::uint32_t targetMask = 1u << target;
    const std::uint32_t lowMask = targetMask - 1u;
    const std::size_t pairs = dim >> 1;
    std::complex<float> *p = psi.data();

    auto body = [=](std::size_t gid) {
      std::uint32_t low = std::uint32_t(gid) & lowMask;
      std::uint32_t high = (std::uint32_t(gid) & ~lowMask) << 1;
      std::size_t b0 = high | low;
      if ((b0 & controlMask) != controlMask) return;
      std::size_t b1 = b0 | targetMask;
      std::complex<float> a0 = p[b0], a1 = p[b1];
      p[b0] = u00 * a0 + u01 * a1;
      p[b1] = u10 * a0 + u11 * a1;
    };

    if (Parallel) {
      // Chunk so each GCD iteration does real work (avoid per-pair overhead).
      const std::size_t chunk = 1u << 12;
      const std::size_t nChunks = (pairs + chunk - 1) / chunk;
      dispatch_apply(nChunks, DISPATCH_APPLY_AUTO, ^(std::size_t c) {
        std::size_t begin = c * chunk;
        std::size_t end = std::min(begin + chunk, pairs);
        for (std::size_t gid = begin; gid < end; ++gid) body(gid);
      });
    } else {
      for (std::size_t gid = 0; gid < pairs; ++gid) body(gid);
    }
  }
};
