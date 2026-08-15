// standalone_bench.cpp — validate the Metal engine against the CPU reference
// and get a first timing signal, independent of CUDA-Q.
//
// Builds GHZ(n) = H(0), CX(0,1), CX(1,2), ..., CX(n-2,n-1) and runs it on:
//   - CpuSV single-threaded
//   - CpuSV multi-threaded (GCD, all cores)
//   - Metal engine (lazy-batched, one command buffer)
// Then checks the Metal statevector matches the CPU one and reports timings.
//
// Usage: standalone_bench [minQ] [maxQ] [repeats]
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include "../src/cpu_sv.hpp"
#include "../src/metal_sv.h"

using clk = std::chrono::steady_clock;
static double ms_since(clk::time_point t0) {
  return std::chrono::duration<double, std::milli>(clk::now() - t0).count();
}

// 1/sqrt(2)
static const float R2 = 0.70710678118654752440f;
// Interleaved-complex 2x2 matrices.
static const float H[8] = {R2, 0, R2, 0, R2, 0, -R2, 0};
static const float X[8] = {0, 0, 1, 0, 1, 0, 0, 0};

struct Gate {
  const float *mat;
  int target;
  std::vector<int> controls;
};

static std::vector<Gate> ghz(int n) {
  std::vector<Gate> g;
  g.push_back({H, 0, {}});
  for (int i = 0; i < n - 1; ++i) g.push_back({X, i + 1, {i}});
  return g;
}

template <bool Parallel>
static double run_cpu(int n, const std::vector<Gate> &gates, int repeats,
                      CpuSV &sv) {
  double best = 1e18;
  for (int r = 0; r < repeats; ++r) {
    sv.reset();
    auto t0 = clk::now();
    for (const auto &g : gates) {
      std::uint32_t cmask = 0;
      for (int c : g.controls) cmask |= (1u << c);
      sv.apply1q<Parallel>(g.mat, g.target, cmask);
    }
    best = std::min(best, ms_since(t0));
  }
  return best;
}

static double run_metal(MSVState *s, const std::vector<Gate> &gates,
                        int repeats) {
  double best = 1e18;
  for (int r = 0; r < repeats; ++r) {
    msv_reset(s);
    auto t0 = clk::now();
    for (const auto &g : gates)
      msv_apply_1q(s, g.mat, g.target, g.controls.data(),
                   (int)g.controls.size());
    msv_synchronize(s);
    best = std::min(best, ms_since(t0));
  }
  return best;
}

int main(int argc, char **argv) {
  int minQ = argc > 1 ? std::atoi(argv[1]) : 10;
  int maxQ = argc > 2 ? std::atoi(argv[2]) : 20;
  int repeats = argc > 3 ? std::atoi(argv[3]) : 5;

  printf("%-4s | %12s | %12s | %12s | %9s | %10s\n", "n", "CPU-1T(ms)",
         "CPU-MT(ms)", "Metal(ms)", "MT/Metal", "maxdiff");
  printf("-----+--------------+--------------+--------------+-----------+------------\n");

  for (int n = minQ; n <= maxQ; ++n) {
    auto gates = ghz(n);

    CpuSV sv(n);
    double t1 = run_cpu<false>(n, gates, repeats, sv);
    std::vector<std::complex<float>> cpuState = sv.psi; // reference state
    double tmt = run_cpu<true>(n, gates, repeats, sv);

    MSVState *s = msv_create(n);
    if (!s) { fprintf(stderr, "metal create failed for n=%d\n", n); return 1; }
    double tm = run_metal(s, gates, repeats);

    // Correctness: compare Metal state to CPU reference.
    std::size_t dim = std::size_t(1) << n;
    std::vector<float> re(dim), im(dim);
    msv_get_state(s, re.data(), im.data());
    double maxdiff = 0.0;
    for (std::size_t i = 0; i < dim; ++i) {
      double dr = re[i] - cpuState[i].real();
      double di = im[i] - cpuState[i].imag();
      maxdiff = std::max(maxdiff, std::sqrt(dr * dr + di * di));
    }
    const char *dev = msv_device_name(s);
    if (n == minQ) printf("# Metal device: %s\n", dev);
    msv_destroy(s);

    printf("%-4d | %12.4f | %12.4f | %12.4f | %8.2fx | %10.2e\n", n, t1, tmt,
           tm, tmt / tm, maxdiff);
  }
  return 0;
}
