/*******************************************************************************
 * Metal statevector backend for CUDA-Q / NVQIR.
 *
 * Subclasses nvqir::CircuitSimulatorBase<float> (Metal has no fp64) and routes
 * gate application to a naive Apple-GPU statevector engine (see metal_sv.h).
 *
 * Qubit convention: CUDA-Q's internal qubit indices are little-endian (q0 =
 * least-significant bit of the amplitude index), exactly the convention the
 * Metal engine uses, so — unlike the QPP backend — no index conversion is
 * needed and growing the state by a qubit is a plain low-half copy.
 *
 * Sampling, expectation values and single-qubit measurement run on the host,
 * on all cores, directly over the unified-memory amplitude buffer; because the
 * buffer is shared, collapsing the state in place is immediately visible to
 * subsequent GPU gates.
 *
 * Errors raised while a JIT-compiled kernel frame is on the stack go through
 * deferOrThrowKernelException: on macOS arm64 a C++ throw cannot unwind
 * through such a frame, so CUDA-Q stores the error and re-raises it once the
 * kernel returns.
 ******************************************************************************/

#include "nvqir/CircuitSimulator.h"

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <complex>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <numeric>
#include <ostream>
#include <random>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#include <dispatch/dispatch.h>

#include "metal_sv.h"

namespace nvqir {
namespace {

using Complex = std::complex<float>;

// std::complex<float> is guaranteed layout-compatible with float[2], which is
// the interleaved-complex layout the engine takes.
const float *asFloats(const Complex *c) {
  return reinterpret_cast<const float *>(c);
}

// Run fn(i) for i in [0, n) on all cores.
template <typename Fn>
void parallelFor(std::size_t n, const Fn &fn) {
  if (n == 1) {
    fn(0);
    return;
  }
  const Fn *f = &fn;
  dispatch_apply(n, DISPATCH_APPLY_AUTO, ^(std::size_t i) { (*f)(i); });
}

// Host passes over the statevector are split into fixed-size chunks so they
// parallelize and so per-chunk partial results stay small.
constexpr std::size_t kChunk = std::size_t(1) << 16;

std::size_t numChunks(std::size_t n) { return (n + kChunk - 1) / kChunk; }

// Run fn(chunkIndex, begin, end) over [0, n) in chunks, on all cores.
template <typename Fn>
void forEachChunk(std::size_t n, const Fn &fn) {
  parallelFor(numChunks(n), [&](std::size_t c) {
    const std::size_t b = c * kChunk;
    fn(c, b, std::min(b + kChunk, n));
  });
}

} // namespace

/// @brief SimulationState for the Metal backend (fp32, canonical qubit q =
/// bit q ordering, the same ordering qpp presents). Either owns a host vector
/// (state built from user data) or adopts the simulator's engine (state
/// extracted with get_state), so extraction never copies 2^n amplitudes.
class MetalSimState : public cudaq::SimulationState {
  std::vector<Complex> owned;
  MSVState *engine = nullptr;
  Complex *amps = nullptr;
  std::size_t dim = 0;

public:
  explicit MetalSimState(std::vector<Complex> data)
      : owned(std::move(data)), amps(owned.data()), dim(owned.size()) {}
  /// Takes ownership of `adopted`; `data` is its synchronized amplitude view.
  MetalSimState(MSVState *adopted, Complex *data)
      : engine(adopted), amps(data),
        dim(std::size_t(1) << msv_num_qubits(adopted)) {}
  MetalSimState(const MetalSimState &) = delete;
  MetalSimState &operator=(const MetalSimState &) = delete;
  ~MetalSimState() override { destroyState(); }

  std::size_t getNumQubits() const override {
    return dim ? std::countr_zero(dim) : 0;
  }

  precision getPrecision() const override { return precision::fp32; }

  Tensor getTensor(std::size_t tensorIdx = 0) const override {
    if (tensorIdx != 0)
      throw std::runtime_error("[metal-state] invalid tensor requested.");
    return Tensor{amps, std::vector<std::size_t>{dim}, precision::fp32};
  }
  std::vector<Tensor> getTensors() const override { return {getTensor()}; }
  std::size_t getNumTensors() const override { return 1; }

  std::complex<double>
  operator()(std::size_t tensorIdx,
             const std::vector<std::size_t> &indices) override {
    if (tensorIdx != 0 || indices.size() != 1 || indices[0] >= dim)
      throw std::runtime_error("[metal-state] invalid element extraction.");
    return amps[indices[0]];
  }

  std::complex<double> getAmplitude(const std::vector<int> &basisState) override {
    if (getNumQubits() != basisState.size())
      throw std::runtime_error(
          "[metal-state] getAmplitude: wrong number of bits.");
    std::size_t idx = 0;
    for (std::size_t i = 0; i < basisState.size(); ++i)
      if (basisState[i]) idx |= (std::size_t(1) << i); // qubit i -> bit i
    return amps[idx];
  }

  std::complex<double> overlap(const cudaq::SimulationState &other) override {
    if (other.getNumTensors() != 1 ||
        other.getTensor().extents != getTensor().extents)
      throw std::runtime_error("[metal-state] overlap: dimension mismatch.");
    std::complex<double> acc{0.0, 0.0};
    if (other.getPrecision() == precision::fp32) {
      auto *o = static_cast<const Complex *>(other.getTensor().data);
      for (std::size_t i = 0; i < dim; ++i)
        acc += std::complex<double>(amps[i]) *
               std::conj(std::complex<double>(o[i]));
    } else {
      auto *o = static_cast<const std::complex<double> *>(other.getTensor().data);
      for (std::size_t i = 0; i < dim; ++i)
        acc += std::complex<double>(amps[i]) * std::conj(o[i]);
    }
    return std::abs(acc);
  }

  std::unique_ptr<cudaq::SimulationState>
  createFromSizeAndPtr(std::size_t size, void *ptr, std::size_t) override {
    if (!std::has_single_bit(size))
      throw std::runtime_error(
          "[metal-state] state data length must be a power of two.");
    auto *p = static_cast<const Complex *>(ptr);
    return std::make_unique<MetalSimState>(std::vector<Complex>(p, p + size));
  }

  void dump(std::ostream &os) const override {
    for (std::size_t i = 0; i < dim; ++i)
      os << amps[i] << "\n";
  }

  void destroyState() override {
    if (engine)
      msv_destroy(engine);
    engine = nullptr;
    owned = {};
    amps = nullptr;
    dim = 0;
  }
};

class MetalCircuitSimulator : public nvqir::CircuitSimulatorBase<float> {
  MSVState *engine = nullptr;
  std::mt19937_64 rng{std::random_device{}()};

  int numQubitsFromDim() const {
    return stateDimension ? std::countr_zero(stateDimension) : 0;
  }
  std::size_t engineDim() const {
    return std::size_t(1) << msv_num_qubits(engine);
  }

  static void fail(const std::string &msg) {
    deferOrThrowKernelException("[metal] " + msg);
  }

  // Make the engine hold exactly `n` qubits, keeping existing amplitudes when
  // growing. Returns false (after reporting the error) on failure.
  bool ensureEngine(int n) {
    if (engine && msv_num_qubits(engine) > n) {
      msv_destroy(engine);
      engine = nullptr;
    }
    const bool ok = engine ? msv_grow_to(engine, n) == 0
                           : (engine = msv_create(n)) != nullptr;
    if (!ok)
      fail(msv_last_error());
    return ok;
  }

  // Synchronize and return the host view of the amplitudes, or nullptr (after
  // reporting the error) if there is no state or the GPU failed.
  Complex *amplitudes() {
    if (!engine)
      return nullptr;
    if (void *p = msv_data(engine))
      return static_cast<Complex *>(p);
    fail(msv_last_error());
    return nullptr;
  }

  void applyMatrix(const Complex *mat, const std::vector<int> &targets,
                   const std::vector<int> &controls) {
    if (msv_apply(engine, asFloats(mat), static_cast<int>(targets.size()),
                  targets.data(), controls.data(),
                  static_cast<int>(controls.size())) != 0)
      throw std::runtime_error(std::string("[metal] ") + msv_last_error());
  }

  // Initialize the qubits just allocated from `data` (2^k amplitudes): the
  // result is |data> ⊗ |current>, the new qubits being the high-order bits.
  void appendState(const Complex *data, std::size_t size) {
    const int oldN = msv_num_qubits(engine), newN = numQubitsFromDim();
    if (newN <= oldN || size != (std::size_t(1) << (newN - oldN))) {
      fail("initial state has " + std::to_string(size) +
           " amplitudes, expected 2^" + std::to_string(newN - oldN));
      return;
    }
    if (!ensureEngine(newN))
      return;
    Complex *amp = amplitudes();
    if (!amp)
      return;
    if (oldN == 0) {
      std::memcpy(amp, data, size * sizeof(Complex));
      return;
    }
    // Block j of the new state is data[j] * old. Block 0 holds the old state,
    // so it is rewritten last.
    const std::size_t oldDim = std::size_t(1) << oldN;
    for (std::size_t j = size; j-- > 0;) {
      const Complex c = data[j];
      Complex *dst = amp + j * oldDim;
      forEachChunk(oldDim, [&](std::size_t, std::size_t b, std::size_t e) {
        for (std::size_t i = b; i < e; ++i)
          dst[i] = c * amp[i];
      });
    }
  }

  // ---- Gate fusion ------------------------------------------------------
  // Consecutive gates are greedily combined into a single dense unitary when
  // one gate's qubits are a subset of the other's (e.g. 1-qubit rotations
  // absorbed into a 2-qubit block), then applied in ONE GPU pass instead of
  // one pass per gate. A bandwidth-bound statevector sim gains directly from
  // fewer full-statevector passes. Fusion never grows a block past the larger
  // operand, which would multiply the per-thread dense-matmul cost. Controls
  // are baked into the dense block, so fused blocks carry no control mask.
  struct DenseOp {
    std::vector<int> qubits;  // sorted ascending
    std::vector<Complex> mat; // 2^m x 2^m, row-major; bit b -> qubits[b]
  };
  int groupSize_ = 0;             // gates in the pending group (0 = none)
  std::uint64_t groupMask_ = 0;   // qubits the pending group acts on
  DenseOp group_;                 // the fused block, valid iff groupSize_ >= 2
  // A 1-gate group stays in sparse (control-mask) form: applying a lone CX as
  // a dense 2-qubit block would only cost more. Buffers are reused, so
  // holding a gate does not allocate once they have grown.
  std::vector<Complex> loneMat_;
  std::vector<int> loneTargets_, loneControls_;
  std::vector<int> scratchTargets_, scratchControls_;
  // Set MSV_DISABLE_FUSION=1 to bypass fusion (for A/B benchmarking).
  const bool fuseEnabled_ = (std::getenv("MSV_DISABLE_FUSION") == nullptr);

  static int posIn(const std::vector<int> &v, int q) {
    return int(std::lower_bound(v.begin(), v.end(), q) - v.begin());
  }

  // Embed `op` (on op.qubits) into the larger sorted qubit set U; identity on
  // U-qubits not in op. Returns a 2^|U| x 2^|U| matrix.
  static std::vector<Complex> embed(const DenseOp &op, const std::vector<int> &U) {
    const int m = int(U.size()), mo = int(op.qubits.size());
    const std::size_t dim = std::size_t(1) << m, dimo = std::size_t(1) << mo;
    std::vector<int> pos(mo);
    for (int b = 0; b < mo; ++b) pos[b] = posIn(U, op.qubits[b]);
    std::size_t opMask = 0;
    for (int b = 0; b < mo; ++b) opMask |= (std::size_t(1) << pos[b]);
    std::vector<Complex> M(dim * dim, Complex(0, 0));
    for (std::size_t r = 0; r < dim; ++r) {
      std::size_t rSub = 0;
      for (int b = 0; b < mo; ++b)
        if (r & (std::size_t(1) << pos[b])) rSub |= (std::size_t(1) << b);
      const std::size_t rRest = r & ~opMask;
      for (std::size_t cSub = 0; cSub < dimo; ++cSub) {
        std::size_t c = rRest;
        for (int b = 0; b < mo; ++b)
          if (cSub & (std::size_t(1) << b)) c |= (std::size_t(1) << pos[b]);
        M[r * dim + c] = op.mat[rSub * dimo + cSub];
      }
    }
    return M;
  }

  // C = A * B (all 2^m x 2^m, row-major).
  static std::vector<Complex> matmul(const std::vector<Complex> &A,
                                     const std::vector<Complex> &B, std::size_t dim) {
    std::vector<Complex> C(dim * dim, Complex(0, 0));
    for (std::size_t i = 0; i < dim; ++i)
      for (std::size_t k = 0; k < dim; ++k) {
        Complex a = A[i * dim + k];
        if (a == Complex(0, 0)) continue;
        for (std::size_t j = 0; j < dim; ++j)
          C[i * dim + j] += a * B[k * dim + j];
      }
    return C;
  }

  // Dense form of a (possibly controlled) gate on sorted(controls + targets):
  // identity unless all controls are 1, where `mat` acts on the targets.
  static DenseOp denseFrom(const Complex *mat, const std::vector<int> &tgt,
                           const std::vector<int> &ctl) {
    std::vector<int> S(ctl);
    S.insert(S.end(), tgt.begin(), tgt.end());
    std::sort(S.begin(), S.end());
    const int m = int(S.size()), nt = int(tgt.size());
    const std::size_t dim = std::size_t(1) << m, dt = std::size_t(1) << nt;
    std::vector<int> tpos(nt);
    for (int b = 0; b < nt; ++b) tpos[b] = posIn(S, tgt[b]);
    std::size_t cmask = 0;
    for (int c : ctl) cmask |= (std::size_t(1) << posIn(S, c));
    std::vector<Complex> M(dim * dim, Complex(0, 0));
    for (std::size_t r = 0; r < dim; ++r) {
      if ((r & cmask) != cmask) { M[r * dim + r] = Complex(1, 0); continue; }
      std::size_t rTgt = 0, rRest = r;
      for (int b = 0; b < nt; ++b) {
        if (r & (std::size_t(1) << tpos[b])) rTgt |= (std::size_t(1) << b);
        rRest &= ~(std::size_t(1) << tpos[b]);
      }
      for (std::size_t cTgt = 0; cTgt < dt; ++cTgt) {
        std::size_t c = rRest;
        for (int b = 0; b < nt; ++b)
          if (cTgt & (std::size_t(1) << b)) c |= (std::size_t(1) << tpos[b]);
        M[r * dim + c] = mat[rTgt * dt + cTgt];
      }
    }
    return DenseOp{std::move(S), std::move(M)};
  }

  // Apply a gate straight to the engine (control-mask path), no fusion.
  void applyTask(const GateApplicationTask &task) {
    scratchTargets_.assign(task.targets.begin(), task.targets.end());
    scratchControls_.assign(task.controls.begin(), task.controls.end());
    applyMatrix(task.matrix.data(), scratchTargets_, scratchControls_);
  }

  // Fold `task` (acting on qubit set `u` together with the group) into the
  // pending group.
  void fuse(const GateApplicationTask &task, std::uint64_t u) {
    std::vector<int> U;
    for (int q = 0; (u >> q) != 0; ++q)
      if ((u >> q) & 1)
        U.push_back(q);
    if (groupSize_ == 1)
      group_ = denseFrom(loneMat_.data(), loneTargets_, loneControls_);
    scratchTargets_.assign(task.targets.begin(), task.targets.end());
    scratchControls_.assign(task.controls.begin(), task.controls.end());
    const DenseOp op =
        denseFrom(task.matrix.data(), scratchTargets_, scratchControls_);
    const auto g = embed(group_, U), o = embed(op, U);
    group_.mat = matmul(o, g, std::size_t(1) << U.size()); // task after group
    group_.qubits = std::move(U);
    groupMask_ = u;
    ++groupSize_;
  }

  void flushGroup() {
    const int size = groupSize_;
    groupSize_ = 0; // reset first: a failed apply must not be retried
    if (size == 1)
      applyMatrix(loneMat_.data(), loneTargets_, loneControls_);
    else if (size > 1)
      applyMatrix(group_.mat.data(), group_.qubits, {});
  }

  void discardGroup() { groupSize_ = 0; }

  // Tripwire for a silently-dropped kernel. While an error raised inside a
  // JIT kernel frame is stashed on the ExecutionContext, `enqueueGate` and
  // `flushGateQueueImpl` are no-ops, so the kernel's gates are never applied.
  // CUDA-Q 0.16 re-raises the stashed error when the kernel returns; on
  // CUDA-Q <= 0.15 on Darwin arm64 it often did not, and sampling silently
  // returned the untouched |0..0> state (see BUG_INVESTIGATION.md). We warn so
  // a regression is loud. It only warns: it does not throw, inspect or clear
  // the stashed exception — all three crashed in the 0.15 failure mode.
  static void checkNotSilentlyDeferred(const char *op) {
    if (!kernelExecutionDeferred())
      return;
    fprintf(stderr,
            "[metal] WARNING: %s() is reading a state whose gates were skipped "
            "because an error was raised inside this kernel, so this result is "
            "meaningless.\n"
            "  CUDA-Q normally re-raises that error as soon as the kernel "
            "returns. If no error follows, you have hit the silent-|0..0> bug "
            "of CUDA-Q <= 0.15 on Darwin arm64; please report it at "
            "https://github.com/vivanwin/cudaq-metal/issues\n",
            op);
  }

protected:
  void addQubitToState() override { addQubitsToState(1); }

  /// @brief Allocate/grow the state. `stateDimension` has already been updated
  /// by the base class to reflect the new total qubit count.
  void addQubitsToState(std::size_t qubitCount,
                        const void *stateDataIn = nullptr) override {
    if (qubitCount == 0)
      return;
    if (stateDataIn)
      appendState(static_cast<const Complex *>(stateDataIn),
                  std::size_t(1) << qubitCount);
    else
      ensureEngine(numQubitsFromDim());
  }

  /// @brief Allocate qubits initialized from an existing state (e.g. one
  /// returned by cudaq.get_state or built with cudaq.State.from_data).
  void addQubitsToState(const cudaq::SimulationState &state) override {
    if (!state.isArrayLike() || state.getNumTensors() != 1 ||
        state.getTensor().get_rank() != 1) {
      fail("qubits can only be initialized from a state vector");
      return;
    }
    const auto t = state.getTensor();
    const std::size_t size = t.get_num_elements();
    if (t.fp_precision == cudaq::SimulationState::precision::fp32) {
      appendState(static_cast<const Complex *>(t.data), size);
    } else {
      auto *d = static_cast<const std::complex<double> *>(t.data);
      const std::vector<Complex> converted(d, d + size);
      appendState(converted.data(), size);
    }
  }

  /// @brief Free the device state.
  void deallocateStateImpl() override {
    discardGroup();
    if (engine) {
      msv_destroy(engine);
      engine = nullptr;
    }
  }

  void applyGate(const GateApplicationTask &task) override {
    const int k = static_cast<int>(task.targets.size());
    if (k < 1 || k > MSV_MAX_TARGETS)
      throw std::runtime_error(
          "[metal] gates with " + std::to_string(k) +
          " targets are not supported (max " + std::to_string(MSV_MAX_TARGETS) +
          ") for op '" + task.operationName + "'.");
    std::uint64_t mask = 0;
    for (auto q : task.targets) mask |= std::uint64_t(1) << q;
    for (auto q : task.controls) mask |= std::uint64_t(1) << q;
    // Gates too wide for a dense block (or every gate, with fusion off) go
    // straight to the engine with a control mask.
    if (!fuseEnabled_ || std::popcount(mask) > MSV_MAX_TARGETS) {
      flushGroup();
      applyTask(task);
      return;
    }
    if (groupSize_ > 0) {
      const std::uint64_t u = groupMask_ | mask;
      if (std::popcount(u) <= std::max(std::popcount(groupMask_), std::popcount(mask))) {
        fuse(task, u);
        return;
      }
      flushGroup();
    }
    loneMat_.assign(task.matrix.begin(), task.matrix.end());
    loneTargets_.assign(task.targets.begin(), task.targets.end());
    loneControls_.assign(task.controls.begin(), task.controls.end());
    groupMask_ = mask;
    groupSize_ = 1;
  }

  // Called by the base at the end of every flushGateQueue(): apply any pending
  // fused block, then make sure the GPU has finished.
  void synchronize() override {
    try {
      flushGroup();
    } catch (const std::exception &e) {
      deferOrThrowKernelException(e.what());
      return;
    }
    if (engine && msv_synchronize(engine) != 0)
      fail(msv_last_error());
  }

  /// @brief Reset the current state back to |0...0>.
  void setToZeroState() override {
    discardGroup();
    const int n = numQubitsFromDim();
    if (n == 0)
      deallocateStateImpl();
    else if (ensureEngine(n))
      msv_reset(engine);
  }

  /// @brief Measure a single qubit, collapsing the state.
  bool measureQubit(const std::size_t index) override {
    flushGateQueue(); // apply any pending fused block first
    Complex *amp = amplitudes();
    if (!amp)
      return false;
    const std::size_t dim = engineDim(), bit = std::size_t(1) << index;

    std::vector<std::array<double, 2>> part(numChunks(dim));
    forEachChunk(dim, [&](std::size_t c, std::size_t b, std::size_t e) {
      double p[2] = {0.0, 0.0};
      for (std::size_t i = b; i < e; ++i)
        p[(i & bit) != 0] += std::norm(amp[i]);
      part[c] = {p[0], p[1]};
    });
    double p0 = 0.0, p1 = 0.0;
    for (const auto &p : part) {
      p0 += p[0];
      p1 += p[1];
    }
    if (!(p0 + p1 > 0.0))
      throw std::runtime_error("[metal] cannot measure a zero-norm state.");

    // Draw against the actual total instead of assuming |psi| = 1: with fp32
    // drift, 1 - p1 is not p0 and could select an impossible outcome.
    std::uniform_real_distribution<double> U(0.0, p0 + p1);
    const bool one = p0 == 0.0 || (p1 > 0.0 && U(rng) < p1);
    const float scale = static_cast<float>(1.0 / std::sqrt(one ? p1 : p0));
    forEachChunk(dim, [&](std::size_t, std::size_t b, std::size_t e) {
      for (std::size_t i = b; i < e; ++i)
        amp[i] = (((i & bit) != 0) == one) ? amp[i] * scale : Complex(0, 0);
    });
    return one;
  }

  /// @brief Reset a qubit to |0> (measure, then flip if it was |1>).
  void resetQubit(const std::size_t index) override {
    if (!measureQubit(index)) // flushes pending gates first
      return;
    static const Complex X[4] = {{0, 0}, {1, 0}, {1, 0}, {0, 0}};
    applyMatrix(X, {static_cast<int>(index)}, {});
  }

  QubitOrdering getQubitOrdering() const override {
    return QubitOrdering::lsb; // matches the engine; no index conversion needed
  }

public:
  MetalCircuitSimulator() { summaryData.name = name(); }
  ~MetalCircuitSimulator() override {
    if (engine)
      msv_destroy(engine);
  }

  void setRandomSeed(std::size_t seed) override { rng.seed(seed); }

  bool isStateVectorSimulator() const override { return true; }

  /// @brief Sample the given qubits. shots<1 => return <Z...Z> expectation.
  cudaq::ExecutionResult sample(const std::vector<std::size_t> &qubits,
                                const int shots,
                                bool includeSequentialData = true) override {
    checkNotSilentlyDeferred("sample");
    flushGateQueue(); // apply any pending fused block first
    const Complex *amp = amplitudes();
    if (!amp)
      return {};
    const std::size_t dim = engineDim(), nc = numChunks(dim);

    // Expectation-value path (used by observe when no shots), normalized by
    // the actual norm so fp32 drift does not bias it.
    if (shots < 1) {
      std::size_t mask = 0;
      for (auto q : qubits)
        mask |= (std::size_t(1) << q);
      std::vector<std::array<double, 2>> part(nc);
      forEachChunk(dim, [&](std::size_t c, std::size_t b, std::size_t e) {
        double norm = 0.0, z = 0.0;
        for (std::size_t i = b; i < e; ++i) {
          const double p = std::norm(amp[i]);
          norm += p;
          z += (std::popcount(i & mask) & 1) ? -p : p;
        }
        part[c] = {norm, z};
      });
      double norm = 0.0, z = 0.0;
      for (const auto &p : part) {
        norm += p[0];
        z += p[1];
      }
      return cudaq::ExecutionResult{{}, norm > 0.0 ? z / norm : 0.0};
    }

    // Inverse-CDF sampling without a 2^n CDF: per-chunk probability totals,
    // sorted uniform draws, then a parallel scan of only the chunks that
    // received shots. Extra memory is O(shots + 2^n / kChunk).
    std::vector<double> cum(nc);
    forEachChunk(dim, [&](std::size_t c, std::size_t b, std::size_t e) {
      double s = 0.0;
      for (std::size_t i = b; i < e; ++i)
        s += std::norm(amp[i]);
      cum[c] = s;
    });
    std::partial_sum(cum.begin(), cum.end(), cum.begin());
    const double total = cum.back();
    if (!(total > 0.0)) {
      fail("cannot sample a zero-norm state.");
      return {};
    }
    std::uniform_real_distribution<double> U(0.0, total);
    std::vector<double> u(shots);
    for (auto &x : u)
      x = U(rng);
    std::sort(u.begin(), u.end());

    struct Hit { std::size_t chunk, first, last; }; // shots [first, last)
    std::vector<Hit> hits;
    for (std::size_t s = 0; s < u.size();) {
      const std::size_t c = std::min<std::size_t>(
          std::upper_bound(cum.begin(), cum.end(), u[s]) - cum.begin(), nc - 1);
      std::size_t e = s + 1;
      while (e < u.size() && u[e] < cum[c])
        ++e;
      hits.push_back({c, s, e});
      s = e;
    }
    std::vector<std::size_t> idx(shots);
    parallelFor(hits.size(), [&](std::size_t h) {
      const Hit &hit = hits[h];
      const std::size_t b = hit.chunk * kChunk, e = std::min(b + kChunk, dim);
      double running = hit.chunk ? cum[hit.chunk - 1] : 0.0;
      std::size_t s = hit.first, last = b;
      for (std::size_t i = b; i < e && s < hit.last; ++i) {
        const double p = std::norm(amp[i]);
        if (p == 0.0)
          continue;
        running += p;
        last = i;
        while (s < hit.last && u[s] < running)
          idx[s++] = i;
      }
      while (s < hit.last) // rounding at the chunk's upper edge
        idx[s++] = last;
    });

    // idx is sorted, so equal outcomes are adjacent: count them in runs.
    std::unordered_map<std::string, std::size_t> local;
    std::string bits(qubits.size(), '0');
    for (std::size_t s = 0; s < idx.size();) {
      std::size_t e = s + 1;
      while (e < idx.size() && idx[e] == idx[s])
        ++e;
      for (std::size_t j = 0; j < qubits.size(); ++j)
        bits[j] = ((idx[s] >> qubits[j]) & 1) ? '1' : '0';
      local[bits] += e - s;
      s = e;
    }

    cudaq::ExecutionResult counts;
    double expVal = 0.0;
    for (auto &[bitstring, count] : local) {
      if (includeSequentialData)
        counts.appendResult(bitstring, count);
      else
        counts.counts[bitstring] += count;
      double p = static_cast<double>(count) / shots;
      expVal += cudaq::sample_result::has_even_parity(bitstring) ? p : -p;
    }
    counts.expectationValue = expVal;
    return counts;
  }

  /// @brief Expose the current statevector for cudaq::get_state() etc.
  std::unique_ptr<cudaq::SimulationState> getSimulationState() override {
    checkNotSilentlyDeferred("get_state");
    flushGateQueue();
    Complex *amp = amplitudes();
    if (!amp)
      return std::make_unique<MetalSimState>(std::vector<Complex>{});
    // Zero-copy hand-off: the base deallocates the simulator state right after
    // this call (finalizeExecutionContext), so the returned state adopts the
    // engine instead of copying 2^n amplitudes and doubling peak memory.
    auto state = std::make_unique<MetalSimState>(engine, amp);
    engine = nullptr;
    return state;
  }

  std::unique_ptr<cudaq::SimulationState>
  createStateFromData(const cudaq::state_data &data) override {
    return MetalSimState(std::vector<Complex>{}).createFromData(data);
  }

  std::string name() const override { return "metal"; }
  NVQIR_SIMULATOR_CLONE_IMPL(MetalCircuitSimulator)
};

} // namespace nvqir

/// Register this simulator with NVQIR under the name "metal".
NVQIR_REGISTER_SIMULATOR(nvqir::MetalCircuitSimulator, metal)
