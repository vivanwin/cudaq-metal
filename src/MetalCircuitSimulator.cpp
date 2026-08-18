/*******************************************************************************
 * Metal statevector backend for CUDA-Q / NVQIR.
 *
 * Subclasses nvqir::CircuitSimulatorBase<float> (Metal has no fp64) and routes
 * gate application to a naive Apple-GPU statevector engine (see metal_sv.h).
 *
 * Qubit convention note: CUDA-Q's internal qubit indices are little-endian
 * (q0 = least-significant bit of the amplitude index), which is exactly the
 * convention the Metal engine uses. So — unlike the QPP backend, which is
 * big-endian internally and must convert every index — this backend needs NO
 * index conversion, and growing the state by a qubit is a plain low-half copy.
 *
 * Sampling and single-qubit measurement are performed on the host by reading
 * the unified-memory amplitude buffer directly; because the buffer is shared,
 * collapsing the state in place is immediately visible to subsequent GPU gates.
 ******************************************************************************/

#include "nvqir/CircuitSimulator.h"

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <complex>
#include <optional>
#include <ostream>
#include <random>
#include <set>
#include <string>
#include <vector>

#include "metal_sv.h"

namespace nvqir {

/// @brief SimulationState for the Metal backend. Owns an fp32 copy of the
/// statevector. The array uses CUDA-Q's canonical ordering (qubit q = bit q of
/// the amplitude index), the same ordering qpp presents, so no reindexing is
/// needed for get_state to agree with qpp-cpu.
struct MetalSimState : public cudaq::SimulationState {
  std::vector<std::complex<float>> state;

  MetalSimState() = default;
  explicit MetalSimState(std::vector<std::complex<float>> s)
      : state(std::move(s)) {}

  std::size_t getNumQubits() const override {
    return static_cast<std::size_t>(std::log2(static_cast<double>(state.size())));
  }

  precision getPrecision() const override { return precision::fp32; }

  Tensor getTensor(std::size_t tensorIdx = 0) const override {
    if (tensorIdx != 0)
      throw std::runtime_error("[metal-state] invalid tensor requested.");
    return Tensor{const_cast<void *>(reinterpret_cast<const void *>(state.data())),
                  std::vector<std::size_t>{state.size()}, precision::fp32};
  }
  std::vector<Tensor> getTensors() const override { return {getTensor()}; }
  std::size_t getNumTensors() const override { return 1; }

  std::complex<double>
  operator()(std::size_t tensorIdx,
             const std::vector<std::size_t> &indices) override {
    if (tensorIdx != 0 || indices.size() != 1)
      throw std::runtime_error("[metal-state] invalid element extraction.");
    return state[indices[0]];
  }

  std::complex<double> getAmplitude(const std::vector<int> &basisState) override {
    if (getNumQubits() != basisState.size())
      throw std::runtime_error(
          "[metal-state] getAmplitude: wrong number of bits.");
    std::size_t idx = 0;
    for (std::size_t i = 0; i < basisState.size(); ++i)
      if (basisState[i]) idx |= (std::size_t(1) << i); // qubit i -> bit i
    return std::complex<double>(state[idx]);
  }

  std::complex<double> overlap(const cudaq::SimulationState &other) override {
    if (other.getNumTensors() != 1 ||
        other.getTensor().extents != getTensor().extents)
      throw std::runtime_error("[metal-state] overlap: dimension mismatch.");
    std::complex<double> acc{0.0, 0.0};
    if (other.getPrecision() == precision::fp32) {
      auto *o = reinterpret_cast<std::complex<float> *>(other.getTensor().data);
      for (std::size_t i = 0; i < state.size(); ++i)
        acc += std::complex<double>(state[i]) *
               std::conj(std::complex<double>(o[i]));
    } else {
      auto *o = reinterpret_cast<std::complex<double> *>(other.getTensor().data);
      for (std::size_t i = 0; i < state.size(); ++i)
        acc += std::complex<double>(state[i]) * std::conj(o[i]);
    }
    return std::abs(acc);
  }

  std::unique_ptr<cudaq::SimulationState>
  createFromSizeAndPtr(std::size_t size, void *ptr, std::size_t) override {
    auto *p = reinterpret_cast<std::complex<float> *>(ptr);
    return std::make_unique<MetalSimState>(
        std::vector<std::complex<float>>(p, p + size));
  }

  void dump(std::ostream &os) const override {
    for (auto &a : state) os << a << "\n";
  }

  void destroyState() override {
    state.clear();
    state.shrink_to_fit();
  }
};

class MetalCircuitSimulator : public nvqir::CircuitSimulatorBase<float> {
  using ScalarType = float;
  using Complex = std::complex<float>;

  MSVState *engine = nullptr;
  std::mt19937_64 rng{std::random_device{}()};

  int numQubitsFromDim() const {
    return stateDimension ? std::countr_zero(stateDimension) : 0;
  }

  // Ensure the engine exists and holds exactly `n` qubits, preserving existing
  // amplitudes when growing.
  void ensureEngine(int n) {
    if (n <= 0)
      return;
    if (!engine) {
      engine = msv_create(n);
      if (!engine)
        throw std::runtime_error("[metal] failed to create Metal simulator "
                                 "(no Metal device?)");
      return;
    }
    int cur = msv_num_qubits(engine);
    if (n > cur) {
      msv_grow_to(engine, n);
    } else if (n < cur) {
      msv_destroy(engine);
      engine = msv_create(n);
    }
  }

  Complex *amplitudes() {
    return reinterpret_cast<Complex *>(msv_data(engine));
  }

  // ---- Gate fusion ------------------------------------------------------
  // Consecutive gates are greedily combined into a single dense unitary on the
  // union of their qubits (up to kFuseMax qubits), then applied in ONE GPU pass
  // instead of one pass per gate. On a bandwidth-bound statevector sim this is
  // the dominant win for deep circuits: it cuts the number of full-statevector
  // read/write passes. Controls are baked into the dense block, so fused blocks
  // carry no control mask.
  static constexpr int kFuseMax = 4; // matches the applykq kernel's max targets

  struct DenseOp {
    std::vector<int> qubits;              // sorted ascending
    std::vector<Complex> mat;             // 2^m x 2^m, row-major; bit b -> qubits[b]
  };
  DenseOp group_;
  bool haveGroup_ = false;
  int groupCount_ = 0;                          // how many gates are in group_
  std::optional<GateApplicationTask> groupSingle_; // the sole gate iff count==1
  // Set MSV_DISABLE_FUSION=1 to bypass fusion (for A/B benchmarking).
  bool fuseEnabled_ = (std::getenv("MSV_DISABLE_FUSION") == nullptr);

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

  // Turn a (possibly controlled) gate task into a dense op on sorted(controls
  // + targets): identity unless all controls are 1, where task.matrix acts on
  // the targets.
  DenseOp denseFromTask(const GateApplicationTask &task) {
    std::vector<int> tgt(task.targets.begin(), task.targets.end());
    std::vector<int> S(task.controls.begin(), task.controls.end());
    S.insert(S.end(), tgt.begin(), tgt.end());
    std::sort(S.begin(), S.end());
    const int m = int(S.size()), nt = int(tgt.size());
    const std::size_t dim = std::size_t(1) << m, dt = std::size_t(1) << nt;
    std::vector<int> tpos(nt);
    for (int b = 0; b < nt; ++b) tpos[b] = posIn(S, tgt[b]);
    std::size_t cmask = 0;
    for (int c : task.controls) cmask |= (std::size_t(1) << posIn(S, c));
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
        M[r * dim + c] = task.matrix[rTgt * dt + cTgt];
      }
    }
    return DenseOp{std::move(S), std::move(M)};
  }

  void flushGroup() {
    if (!haveGroup_) return;
    // A group of a single gate is applied via its original (possibly sparse,
    // control-masked) form — converting a lone CX/rotation into a dense block
    // would be pure overhead. Only genuinely fused (>=2 gate) blocks go dense.
    if (groupCount_ == 1 && groupSingle_) {
      applyDirect(*groupSingle_);
    } else {
      const int m = int(group_.qubits.size());
      std::vector<float> mi(group_.mat.size() * 2);
      for (std::size_t i = 0; i < group_.mat.size(); ++i) {
        mi[2 * i] = group_.mat[i].real();
        mi[2 * i + 1] = group_.mat[i].imag();
      }
      if (m == 1)
        msv_apply_1q(engine, mi.data(), group_.qubits[0], nullptr, 0);
      else
        msv_apply_kq(engine, mi.data(), m, group_.qubits.data(), m, nullptr, 0);
    }
    haveGroup_ = false;
    groupCount_ = 0;
    groupSingle_.reset();
    group_.qubits.clear();
    group_.mat.clear();
  }

  void startGroup(const GateApplicationTask &task) {
    group_ = denseFromTask(task);
    haveGroup_ = true;
    groupCount_ = 1;
    groupSingle_.emplace(task);
  }

  void fuseIn(const GateApplicationTask &task) {
    if (!haveGroup_) { startGroup(task); return; }
    DenseOp op = denseFromTask(task);
    std::vector<int> U;
    std::set_union(group_.qubits.begin(), group_.qubits.end(), op.qubits.begin(),
                   op.qubits.end(), std::back_inserter(U));
    // Only fuse when the union is no bigger than the larger operand, i.e. one
    // gate's qubits are a subset of the other's. This absorbs cheap gates
    // (e.g. 1-qubit rotations into a 2-qubit block) for free, without growing
    // the block size — which would multiply the per-thread dense-matmul cost
    // and can be a net loss for already-dense gates.
    const int maxOperand = std::max(group_.qubits.size(), op.qubits.size());
    if (int(U.size()) > maxOperand) {
      flushGroup();
      startGroup(task);
      return;
    }
    const std::size_t dim = std::size_t(1) << U.size();
    auto Fg = embed(group_, U);
    auto Og = embed(op, U);
    group_ = DenseOp{std::move(U), matmul(Og, Fg, dim)}; // op applied after group
    haveGroup_ = true;
    ++groupCount_;
    groupSingle_.reset(); // no longer a lone gate
  }

protected:
  /// @brief Grow the state by one qubit (new qubit is the high bit, |0>).
  void addQubitToState() override { addQubitsToState(1); }

  /// @brief Allocate/grow the state. `stateDimension` has already been updated
  /// by the base class to reflect the new total qubit count.
  void addQubitsToState(std::size_t qubitCount,
                        const void *stateDataIn = nullptr) override {

    if (qubitCount == 0)
      return;
    const int newN = numQubitsFromDim();
    const bool firstAlloc = (engine == nullptr);
    ensureEngine(newN);
    if (stateDataIn) {
      if (!firstAlloc)
        throw std::runtime_error("[metal] initializing appended qubits from "
                                 "explicit state data is not supported.");
      std::memcpy(amplitudes(), stateDataIn, stateDimension * sizeof(Complex));
    }
  }

  void discardGroup() {
    haveGroup_ = false;
    groupCount_ = 0;
    groupSingle_.reset();
    group_.qubits.clear();
    group_.mat.clear();
  }

  /// @brief Free the device state.
  void deallocateStateImpl() override {
    discardGroup();
    if (engine) {
      msv_destroy(engine);
      engine = nullptr;
    }
  }

  /// @brief Apply a gate. Only single-target gates (with any number of
  /// controls) are supported so far — enough for H/X/Z/rotations and
  /// (multi-)controlled-X/Z, i.e. GHZ and most textbook circuits.
  int gateCount_ = 0;

  // Defensive check: cudaq's `deferOrThrowKernelException` stashes JIT-frame
  // exceptions on the ExecutionContext because a C++ exception cannot unwind
  // through a JIT frame on Darwin arm64. In that state, `enqueueGate` and
  // `flushGateQueueImpl` become no-ops, and sampling then returns an untouched
  // |0..0> state — a silently wrong answer that has bitten every multi-
  // evaluation workflow on this backend (see BUG_INVESTIGATION.md).
  //
  // We can't just `throw` from here — if we're still inside a JIT launch frame
  // (ctx->inKernelLaunch=true), the throw itself would hit the same Darwin
  // arm64 unwinder bug and terminate the process. So:
  //   - print a loud, unmissable warning to stderr on every affected call
  //   - clear the deferred exception so the next kernel invocation gets a
  //     fresh chance (rather than being permanently poisoned)
  //   - throw only when it is safe to do so (outside the JIT launch frame)
  static void checkNotSilentlyDeferred(const char *op) {
    if (!kernelExecutionDeferred())
      return;
    auto *ctx = cudaq::getExecutionContext();
    const bool safeToThrow = !ctx || !ctx->inKernelLaunch;
    // Always warn loudly — this is the last chance to tell the caller the
    // result is untrustworthy.
    fprintf(stderr,
            "\n\n==============================================================================\n"
            "  [metal] WARNING: %s() called with a deferred kernel exception.\n"
            "  cudaq discarded this kernel's gate queue (no gates were applied)\n"
            "  due to an upstream Darwin arm64 JIT-frame exception. The state is\n"
            "  the untouched |0..0>, so any result computed from it is WRONG.\n"
            "  This process's cudaq state is now inconsistent; RESTART the Python\n"
            "  process and re-run. This is a known upstream cudaq bug; see\n"
            "  BUG_INVESTIGATION.md in the cudaq-metal repo for details.\n"
            "==============================================================================\n\n",
            op);
    // NB: we deliberately do NOT clear ctx->deferredKernelException here.
    // Clearing seemed a good idea (poison-vs-fresh) but in practice caused a
    // downstream crash — apparently cudaq relies on that flag remaining set
    // until *it* clears it, and unsetting from underneath the runtime races
    // against something in the observe/sample finalization path.
    (void)ctx;
    // We deliberately DO NOT throw here. Throwing from sample()/get_state()
    // while inKernelLaunch=true hits the same Darwin arm64 unwinder bug the
    // deferred-exception mechanism was invented to work around, and terminates
    // the process. Instead we fall through and let the (wrong) |0..0> state be
    // sampled. The warning above is our only signal to the user; they should
    // notice, kill the process, and restart. The alternative (silent zeros
    // with no warning) is what shipped before this defensive check was added
    // and was the whole reason BUG_INVESTIGATION.md exists.
    (void)safeToThrow;
  }
  void applyGate(const GateApplicationTask &task) override {
    const int nT = static_cast<int>(task.targets.size());
    ++gateCount_;
    if (gateCount_ <= 5 || gateCount_ % 100 == 0) {
      std::string tgts;
      for (auto t : task.targets) tgts += std::to_string(t) + ",";
      std::string ctrls;
      for (auto c : task.controls) ctrls += std::to_string(c) + ",";
    }
    if (nT < 1 || nT > 4)
      throw std::runtime_error(
          "[metal] gates with " + std::to_string(nT) +
          " targets are not supported (max 4) for op '" + task.operationName +
          "'.");
    // If controls+targets span more than a fused block, it can't become a
    // dense <=kFuseMax block: flush the pending group and apply it directly
    // with a control mask.
    if (!fuseEnabled_) { applyDirect(task); return; }
    std::set<int> span(task.targets.begin(), task.targets.end());
    span.insert(task.controls.begin(), task.controls.end());
    if (static_cast<int>(span.size()) > kFuseMax) {
      flushGroup();
      applyDirect(task);
      return;
    }
    fuseIn(task);
  }

  // Apply a gate straight to the engine (control-mask path), no fusion.
  void applyDirect(const GateApplicationTask &task) {
    const int nT = static_cast<int>(task.targets.size());
    std::vector<float> m(task.matrix.size() * 2);
    for (std::size_t i = 0; i < task.matrix.size(); ++i) {
      m[2 * i] = task.matrix[i].real();
      m[2 * i + 1] = task.matrix[i].imag();
    }
    std::vector<int> ctrls(task.controls.begin(), task.controls.end());
    if (nT == 1) {
      msv_apply_1q(engine, m.data(), static_cast<int>(task.targets[0]),
                   ctrls.data(), static_cast<int>(ctrls.size()));
    } else {
      std::vector<int> tgts(task.targets.begin(), task.targets.end());
      msv_apply_kq(engine, m.data(), nT, tgts.data(), nT, ctrls.data(),
                   static_cast<int>(ctrls.size()));
    }
  }

  // Called by the base at the end of every flushGateQueue(): apply any pending
  // fused block, then make sure the GPU has finished.
  void synchronize() override {
    flushGroup();
    if (engine)
      msv_synchronize(engine);
  }

  /// @brief Reset the current state back to |0...0>.
  void setToZeroState() override {
    discardGroup();
    const int n = numQubitsFromDim();
    if (n == 0) {
      deallocateStateImpl();
      return;
    }
    ensureEngine(n);
    msv_reset(engine);
  }

  /// @brief Measure a single qubit, collapsing the state.
  bool measureQubit(const std::size_t index) override {
    flushGateQueue(); // apply any pending fused block first
    Complex *amp = amplitudes();
    const std::size_t dim = std::size_t(1) << msv_num_qubits(engine);
    const std::size_t bit = std::size_t(1) << index;

    double p1 = 0.0;
    for (std::size_t i = 0; i < dim; ++i)
      if (i & bit)
        p1 += std::norm(amp[i]);

    std::uniform_real_distribution<double> U(0.0, 1.0);
    const bool outcome = U(rng) < p1;
    const double pOut = outcome ? p1 : (1.0 - p1);
    const float inv = pOut > 0.0 ? static_cast<float>(1.0 / std::sqrt(pOut)) : 0.0f;
    for (std::size_t i = 0; i < dim; ++i) {
      if (static_cast<bool>(i & bit) == outcome)
        amp[i] *= inv;
      else
        amp[i] = Complex(0.0f, 0.0f);
    }
    return outcome;
  }

  /// @brief Reset a qubit to |0> (measure, then flip if it was |1>).
  void resetQubit(const std::size_t index) override {
    flushGateQueue();
    if (measureQubit(index)) {
      static const float X[8] = {0, 0, 1, 0, 1, 0, 0, 0};
      msv_apply_1q(engine, X, static_cast<int>(index), nullptr, 0);
      msv_synchronize(engine);
    }
  }

  QubitOrdering getQubitOrdering() const override {
    return QubitOrdering::lsb; // matches the engine; no index conversion needed
  }

public:
  MetalCircuitSimulator() { summaryData.name = name(); }
  virtual ~MetalCircuitSimulator() {
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

    Complex *amp = amplitudes();
    const std::size_t dim = std::size_t(1) << msv_num_qubits(engine);
    {
      double totalNorm = 0.0;
      int nonzeroCount = 0;
      for (std::size_t i = 0; i < dim; ++i) {
        double n = std::norm(amp[i]);
        totalNorm += n;
        if (n > 1e-12) ++nonzeroCount;
      }
    }

    // Expectation-value path (used by observe when no shots).
    if (shots < 1) {
      std::size_t mask = 0;
      for (auto q : qubits)
        mask |= (std::size_t(1) << q);
      double ev = 0.0;
      for (std::size_t i = 0; i < dim; ++i) {
        double p = std::norm(amp[i]);
        ev += (std::popcount(i & mask) & 1) ? -p : p;
      }
      return cudaq::ExecutionResult{{}, ev};
    }

    // Build a cumulative distribution over basis states and inverse-CDF sample.
    std::vector<double> cdf(dim);
    double running = 0.0;
    for (std::size_t i = 0; i < dim; ++i) {
      running += std::norm(amp[i]);
      cdf[i] = running;
    }
    std::uniform_real_distribution<double> U(0.0, running);

    const std::size_t m = qubits.size();
    std::unordered_map<std::string, std::size_t> local;
    std::string bits(m, '0');
    for (int s = 0; s < shots; ++s) {
      double u = U(rng);
      std::size_t idx =
          std::lower_bound(cdf.begin(), cdf.end(), u) - cdf.begin();
      if (idx >= dim)
        idx = dim - 1;
      for (std::size_t j = 0; j < m; ++j)
        bits[j] = ((idx >> qubits[j]) & 1) ? '1' : '0';
      ++local[bits];
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
  /// Returns an fp32 copy of the amplitudes in canonical (qubit q = bit q) order.
  std::unique_ptr<cudaq::SimulationState> getSimulationState() override {
    checkNotSilentlyDeferred("get_state");
    flushGateQueue();
    if (!engine)
      return std::make_unique<MetalSimState>();
    const Complex *amp = amplitudes(); // syncs the GPU
    const std::size_t dim = std::size_t(1) << msv_num_qubits(engine);
    return std::make_unique<MetalSimState>(
        std::vector<Complex>(amp, amp + dim));
  }

  std::unique_ptr<cudaq::SimulationState>
  createStateFromData(const cudaq::state_data &data) override {
    return std::make_unique<MetalSimState>()->createFromData(data);
  }

  std::string name() const override { return "metal"; }
  NVQIR_SIMULATOR_CLONE_IMPL(MetalCircuitSimulator)
};

} // namespace nvqir

/// Exported C helper: clear cudaq's deferredKernelException from Python.
/// Python is always outside a JIT launch frame, so this is safe.
/// Returns 1 if it cleared something, 0 otherwise.
extern "C" int metal_clear_deferred_kernel_exception() {
  auto *ctx = cudaq::getExecutionContext();
  if (ctx && ctx->deferredKernelException) {
    ctx->deferredKernelException = nullptr;
    return 1;
  }
  return 0;
}

/// Register this simulator with NVQIR under the name "metal".
NVQIR_REGISTER_SIMULATOR(nvqir::MetalCircuitSimulator, metal)
