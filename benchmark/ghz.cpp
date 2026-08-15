// ghz.cpp — minimal CUDA-Q GHZ program used for the toolchain smoke test and
// (later) the qpp-cpu vs metal comparison. Sample the n-qubit GHZ state.
#include <cudaq.h>
#include <cstdio>
#include <cstdlib>

struct ghz {
  void operator()(int n) __qpu__ {
    cudaq::qvector q(n);
    h(q[0]);
    for (int i = 0; i < n - 1; ++i)
      x<cudaq::ctrl>(q[i], q[i + 1]);
    mz(q);
  }
};

int main(int argc, char **argv) {
  int n = argc > 1 ? std::atoi(argv[1]) : 5;
  int shots = argc > 2 ? std::atoi(argv[2]) : 1000;
  auto counts = cudaq::sample(shots, ghz{}, n);
  printf("GHZ(%d), %d shots:\n", n, shots);
  counts.dump();
  return 0;
}
