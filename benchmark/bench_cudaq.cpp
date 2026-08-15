// bench_cudaq.cpp — benchmark cudaq::sample on a GHZ circuit for the target
// this binary was compiled with (metal or qpp-cpu). Compile the SAME source
// once per target and run both to compare; timing is internal so it excludes
// process startup / JIT of the first call (a warmup sample is discarded).
//
// Usage: bench_cudaq [minQ] [maxQ] [shots] [repeats]
#include <cudaq.h>

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <string>

struct ghz {
  void operator()(int n) __qpu__ {
    cudaq::qvector q(n);
    h(q[0]);
    for (int i = 0; i < n - 1; ++i)
      x<cudaq::ctrl>(q[i], q[i + 1]);
    mz(q);
  }
};

using clk = std::chrono::steady_clock;

int main(int argc, char **argv) {
  int minQ = argc > 1 ? std::atoi(argv[1]) : 10;
  int maxQ = argc > 2 ? std::atoi(argv[2]) : 20;
  int shots = argc > 3 ? std::atoi(argv[3]) : 1000;
  int repeats = argc > 4 ? std::atoi(argv[4]) : 5;

  printf("# shots=%d repeats=%d (best-of time)\n", shots, repeats);
  printf("%-4s | %12s | %-28s\n", "n", "sample(ms)", "GHZ check");
  printf("-----+--------------+------------------------------\n");

  for (int n = minQ; n <= maxQ; ++n) {
    // Warmup (JIT + first allocation) — discarded.
    auto warm = cudaq::sample(shots, ghz{}, n);

    double best = 1e18;
    cudaq::sample_result last;
    for (int r = 0; r < repeats; ++r) {
      auto t0 = clk::now();
      last = cudaq::sample(shots, ghz{}, n);
      double ms = std::chrono::duration<double, std::milli>(clk::now() - t0).count();
      if (ms < best) best = ms;
    }

    // GHZ correctness: only the all-zeros and all-ones strings should appear.
    std::string z(n, '0'), o(n, '1');
    std::size_t cz = last.count(z), co = last.count(o);
    std::size_t total = 0;
    for (auto &kv : last.to_map()) total += kv.second;
    bool ok = (cz + co == total) && cz > 0 && co > 0;
    char check[64];
    std::snprintf(check, sizeof(check), "%s  |0..0>=%zu |1..1>=%zu",
                  ok ? "OK " : "BAD", cz, co);

    printf("%-4d | %12.4f | %-28s\n", n, best, check);
    fflush(stdout);
  }
  return 0;
}
