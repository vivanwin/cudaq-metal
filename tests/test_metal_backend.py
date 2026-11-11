"""Correctness tests: the `metal` target must agree with CUDA-Q's `qpp-cpu`.

Run on an Apple Silicon Mac with cudaq installed:

    pip install -e ".[test]"
    python -m pytest tests/

Covers every engine kernel (apply1q, apply2q, applykq with k=3/4, controlled
multi-target gates), gate fusion on and off, sampling, observe, mid-circuit
measurement, reset, state initialization, the multi-chunk host paths (n >= 17),
lazy qubit allocation, buffer reuse (on and off, checked with the plugin's
own counters) and error reporting.
"""
import os
import subprocess
import sys

import numpy as np
import pytest

import cudaq_metal  # noqa: F401  (registers the target; must precede cudaq)
import cudaq
from cudaq import spin

from _util import buffer_stats, env_flag, on

_rng = np.random.default_rng(1234)


def random_unitary(dim):
    z = _rng.standard_normal((dim, dim)) + 1j * _rng.standard_normal((dim, dim))
    q, r = np.linalg.qr(z)
    d = np.diagonal(r)
    return q * (d / np.abs(d))


cudaq.register_operation("su4", random_unitary(4))
cudaq.register_operation("u3q", random_unitary(8))
cudaq.register_operation("u4q", random_unitary(16))
cudaq.register_operation("u5q", random_unitary(32))


@cudaq.kernel
def ghz(n: int):
    q = cudaq.qvector(n)
    h(q[0])
    for i in range(n - 1):
        x.ctrl(q[i], q[i + 1])


@cudaq.kernel
def ghz_measured(n: int):
    q = cudaq.qvector(n)
    h(q[0])
    for i in range(n - 1):
        x.ctrl(q[i], q[i + 1])
    mz(q)


@cudaq.kernel
def phases():
    q = cudaq.qvector(3)
    h(q[0])
    h(q[1])
    h(q[2])
    rx(0.7, q[0])
    ry(1.3, q[1])
    rz(0.9, q[2])
    t(q[0])
    s(q[1])
    x.ctrl(q[0], q[2])
    rz(2.1, q[1])


@cudaq.kernel
def su4_brickwork(n: int, layers: int):
    q = cudaq.qvector(n)
    for i in range(n):
        h(q[i])
    for layer in range(layers):
        j = layer % 2
        while j + 1 < n:
            su4(q[j], q[j + 1])
            j += 2
    su4(q[0], q[n - 1])  # non-adjacent targets


@cudaq.kernel
def wide_gates():
    q = cudaq.qvector(6)
    h(q[0])
    h(q[1])
    h(q[3])
    x.ctrl(q[0], q[1], q[2])  # CCX ...
    rx(0.3, q[0])  # ... with absorbed rotations -> fused 3-qubit block
    ry(0.2, q[2])
    swap.ctrl(q[3], q[4], q[5])  # controlled 2-target gate
    u3q(q[0], q[2], q[4])  # applykq, k=3
    u4q(q[1], q[2], q[4], q[5])  # applykq, k=4
    u4q.ctrl(q[0], q[1], q[2], q[3], q[5])  # 4 targets + control (unfusable)


@cudaq.kernel
def hw_efficient(n: int, layers: int):
    q = cudaq.qvector(n)
    for _ in range(layers):
        for i in range(n):
            ry(0.4 + 0.1 * i, q[i])
        for i in range(n - 1):
            x.ctrl(q[i], q[i + 1])
            rz(0.5, q[i + 1])
            x.ctrl(q[i], q[i + 1])


@cudaq.kernel
def reset_after_h(n: int):
    q = cudaq.qvector(n)
    for i in range(n):
        h(q[i])
    for i in range(n):
        reset(q[i])
    x(q[0])


@cudaq.kernel
def copy_bit() -> bool:
    q = cudaq.qvector(2)
    h(q[0])
    b = mz(q[0])
    if b:
        x(q[1])
    return b == mz(q[1])


@cudaq.kernel
def from_state(st: cudaq.State):
    q = cudaq.qvector(st)
    h(q[0])


@cudaq.kernel
def append_state(st: cudaq.State):
    a = cudaq.qvector(2)
    x(a[0])
    b = cudaq.qvector(st)
    h(a[1])
    h(b[0])


@cudaq.kernel
def append_after_flush(st: cudaq.State):
    # exp_pauli flushes, so the engine exists (2 qubits) when the state block
    # is appended: the grow-then-tensor path, not the lazy one.
    a = cudaq.qvector(2)
    h(a[0])
    exp_pauli(0.3, a, "ZZ")
    b = cudaq.qvector(st)
    x.ctrl(a[0], b[0])


@cudaq.kernel
def from_list(v: list[complex]):
    q = cudaq.qvector(v)
    h(q[0])


@cudaq.kernel
def five_target_gate():
    q = cudaq.qvector(5)
    u5q(q[0], q[1], q[2], q[3], q[4])
    mz(q)


@cudaq.kernel
def register_of(n: int):
    q = cudaq.qvector(n)
    mz(q)


@cudaq.kernel
def literal_register():
    # A constant-size register reaches the backend one qubit at a time; the
    # engine is built lazily, once, at 18 qubits.
    q = cudaq.qvector(18)
    for i in range(18):
        ry(0.1 + 0.05 * i, q[i])
    for i in range(17):
        x.ctrl(q[i], q[i + 1])


@cudaq.kernel
def grow_after_flush():
    # exp_pauli flushes the gate queue, so the engine is built at 2 qubits
    # before the ancilla arrives and has to grow afterwards.
    q = cudaq.qvector(2)
    h(q[0])
    ry(0.4, q[1])
    exp_pauli(0.3, q, "ZZ")
    a = cudaq.qubit()
    x.ctrl(q[0], a)
    ry(0.2, a)


@cudaq.kernel
def no_gates(n: int):
    q = cudaq.qvector(n)


def builder_brickwork(n):
    """Builder kernel (what Classiq's qprog_to_cudaq_kernel returns)."""
    k = cudaq.make_kernel()
    q = k.qalloc(n)
    for i in range(n):
        k.ry(0.3 + 0.02 * i, q[i])
    for j in range(0, n - 1, 2):
        k.cx(q[j], q[j + 1])
        k.rz(0.7, q[j + 1])
        k.cx(q[j], q[j + 1])
    return k


def state(target, kernel, *args):
    return np.array(on(target, cudaq.get_state, kernel, *args), copy=True).astype(np.complex128)


def assert_same_state(a, b, tol=2e-5):
    assert a.shape == b.shape
    overlap = np.vdot(a, b)
    fidelity = abs(overlap) ** 2 / (np.vdot(a, a).real * np.vdot(b, b).real)
    phase = overlap / abs(overlap) if abs(overlap) > 1e-12 else 1.0
    assert fidelity > 1 - 1e-5
    assert np.max(np.abs(a - b * np.conj(phase))) < tol


STATE_CASES = {
    "ghz5": (ghz, 5),
    "phases": (phases,),
    "su4_brickwork6": (su4_brickwork, 6, 3),
    "wide_gates": (wide_gates,),
    "hw_efficient8": (hw_efficient, 8, 3),
    "su4_brickwork18": (su4_brickwork, 18, 4),  # > 1 host chunk
}


@pytest.mark.parametrize("case", STATE_CASES)
def test_state_matches_qpp(case):
    kernel, *args = STATE_CASES[case]
    assert_same_state(state("qpp-cpu", kernel, *args), state("metal", kernel, *args))


@pytest.mark.skipif(env_flag("MSV_DISABLE_FUSION"), reason="this is the fusion-off run")
def test_state_matches_qpp_with_fusion_disabled():
    # MSV_DISABLE_FUSION is read when the simulator is created, so it needs a
    # fresh process.
    env = dict(os.environ, MSV_DISABLE_FUSION="1")
    r = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", __file__,
         "-k", "test_state_matches_qpp and not fusion"],
        env=env, capture_output=True, text=True,
    )
    assert r.returncode == 0, r.stdout + r.stderr


@pytest.mark.parametrize("n", [4, 18])
def test_sample_distribution(n):
    cudaq.set_random_seed(7)
    shots = 20000
    counts = on("metal", cudaq.sample, ghz_measured, n, shots_count=shots)
    assert set(counts) == {"0" * n, "1" * n}
    assert abs(counts.count("0" * n) / shots - 0.5) < 0.02


def test_sample_matches_exact_probabilities():
    cudaq.set_random_seed(11)
    shots = 40000
    exact = np.abs(state("qpp-cpu", su4_brickwork, 4, 3)) ** 2
    counts = on("metal", cudaq.sample, su4_brickwork, 4, 3, shots_count=shots)
    # Bitstring character j is qubit j; qubit j is bit j of the index.
    sampled = np.zeros_like(exact)
    for bits in counts:
        sampled[int(bits[::-1], 2)] = counts.count(bits) / shots
    assert 0.5 * np.abs(sampled - exact).sum() < 0.02


@pytest.mark.parametrize("n", [6, 18])
def test_observe_matches_qpp(n):
    H = spin.z(0) * spin.z(n - 1) + 0.5 * spin.x(1) * spin.x(2) - 0.3 * spin.y(0) + 0.2 * spin.z(3)
    expected = on("qpp-cpu", cudaq.observe, hw_efficient, H, n, 2).expectation()
    assert on("metal", cudaq.observe, hw_efficient, H, n, 2).expectation() == pytest.approx(expected, abs=1e-4)


@pytest.mark.parametrize("n", [3, 18])
def test_reset(n):
    s = state("metal", reset_after_h, n)
    expected = np.zeros(2**n)
    expected[1] = 1.0  # x(q[0]) after resetting everything
    assert_same_state(expected.astype(np.complex128), s)


def test_mid_circuit_measurement_and_conditional():
    cudaq.set_random_seed(3)
    assert set(on("metal", cudaq.run, copy_bit, shots_count=200)) == {True}


def test_initialize_from_state():
    v = np.array([0.6, 0.8j, 0.0, 0.0], dtype=np.complex64)
    for kernel in (from_state, append_state, append_after_flush):
        expected = state("qpp-cpu", kernel, cudaq.State.from_data(v.astype(np.complex128)))
        cudaq.set_target("metal")
        try:
            got = np.array(cudaq.get_state(kernel, cudaq.State.from_data(v)), copy=True)
        finally:
            cudaq.reset_target()
        assert_same_state(expected, got.astype(np.complex128))


def test_initialize_from_extracted_state():
    s = on("metal", cudaq.get_state, ghz, 3)
    assert_same_state(state("qpp-cpu", from_state, on("qpp-cpu", cudaq.get_state, ghz, 3)),
                      state("metal", from_state, s))


def test_initialize_from_list():
    v = [0.6 + 0j, 0.8j, 0j, 0j]
    assert_same_state(state("qpp-cpu", from_list, v), state("metal", from_list, v))


def test_unsupported_gate_raises():
    with pytest.raises(RuntimeError, match="max 4"):
        on("metal", cudaq.sample, five_target_gate)


def test_too_many_qubits_raises():
    with pytest.raises(RuntimeError, match="outside the supported range"):
        on("metal", cudaq.sample, register_of, 32)
    # The simulator must still work afterwards.
    assert set(on("metal", cudaq.sample, ghz_measured, 3, shots_count=100)) <= {"000", "111"}


def test_too_many_qubits_raises_without_gates():
    # The engine is built lazily, but an impossible size is still reported at
    # the allocation, even if nothing ever touches the state.
    with pytest.raises(RuntimeError, match="outside the supported range"):
        on("metal", cudaq.observe, no_gates, spin.z(0), 32)


# ---- lazy qubit allocation and statevector buffer reuse ---------------------
# MSV_DISABLE_BUFFER_CACHE is read on every allocation and free, so the
# buffer_cache fixture toggles it in-process.

def _set_buffer_cache(monkeypatch, mode):
    """mode "on" (unset), "off" (=1) or "zero" (=0, which must keep it on);
    returns whether reuse is on."""
    value = {"on": None, "off": "1", "zero": "0"}[mode]
    if value is None:
        monkeypatch.delenv("MSV_DISABLE_BUFFER_CACHE", raising=False)
    else:
        monkeypatch.setenv("MSV_DISABLE_BUFFER_CACHE", value)
    return mode != "off"


@pytest.fixture(params=["on", "off"])
def buffer_cache(request, monkeypatch):
    """Run with buffer reuse on and off; the value is whether it is on."""
    return _set_buffer_cache(monkeypatch, request.param)


@pytest.mark.parametrize("case", ["literal_register", "grow_after_flush", "builder"])
def test_lazy_allocation_matches_qpp(case, buffer_cache):
    kernel, args = {"literal_register": (literal_register, ()),
                    "grow_after_flush": (grow_after_flush, ()),
                    "builder": (builder_brickwork(18), ())}[case]
    assert_same_state(state("qpp-cpu", kernel, *args), state("metal", kernel, *args))


def test_gate_free_kernel():
    expected = np.zeros(2**18, dtype=np.complex128)
    expected[0] = 1.0
    assert_same_state(expected, state("metal", no_gates, 18))
    H = spin.z(0) + 2.0 * spin.z(17) + 0.5 * spin.x(3)
    assert on("metal", cudaq.observe, no_gates, H, 18).expectation() == pytest.approx(3.0)


def test_reused_buffer_starts_clean(buffer_cache):
    # Same-size states back to back reuse one buffer (18 qubits is above the
    # cache threshold): every run must start from |0...0>, whatever the
    # previous one left behind, through each way a state ends (observe,
    # sample, an extracted state that is then dropped).
    n = 18
    H = spin.z(0) * spin.z(n - 1) + 0.5 * spin.x(1) * spin.x(2)
    want_hw = state("qpp-cpu", hw_efficient, n, 2)
    want_obs = on("qpp-cpu", cudaq.observe, hw_efficient, H, n, 2).expectation()
    for _ in range(2):
        assert_same_state(want_hw, state("metal", hw_efficient, n, 2))
        assert on("metal", cudaq.observe, hw_efficient, H, n, 2).expectation() == pytest.approx(want_obs, abs=1e-4)
        counts = on("metal", cudaq.sample, ghz_measured, n, shots_count=200)
        assert set(counts) <= {"0" * n, "1" * n}
        assert_same_state(state("qpp-cpu", ghz, n), state("metal", ghz, n))
    # A different size in between, then the original size again.
    assert_same_state(state("qpp-cpu", ghz, n + 1), state("metal", ghz, n + 1))
    assert_same_state(want_hw, state("metal", hw_efficient, n, 2))


def test_extracted_state_survives_reuse(buffer_cache):
    # get_state hands the engine's buffer to the returned state; running more
    # same-size kernels while it is alive must not touch it.
    n = 18
    cudaq.set_target("metal")
    try:
        kept = cudaq.get_state(ghz, n)
        cudaq.get_state(hw_efficient, n, 2)
        cudaq.observe(hw_efficient, spin.z(0), n, 2)
        got = np.array(kept, copy=True).astype(np.complex128)
    finally:
        cudaq.reset_target()
    assert_same_state(state("qpp-cpu", ghz, n), got)


def _observe_sizes(sizes):
    """Shot-free observe of an 18+-qubit kernel per size (GPU expectation
    path, so it uses scratch); returns the change in buffer_stats()."""
    before = buffer_stats()
    for n in sizes:
        on("metal", cudaq.observe, hw_efficient, spin.z(0), n, 1)
    return tuple(b - a for a, b in zip(before, buffer_stats()))


@pytest.mark.parametrize("mode", ["on", "off", "zero"])
def test_buffers_are_reused(mode, monkeypatch):
    enabled = _set_buffer_cache(monkeypatch, mode)
    _observe_sizes([18])  # leaves an idle 18-qubit buffer and scratch
    allocs, reuses, scratch = _observe_sizes([18] * 3)
    if enabled:
        assert (allocs, reuses) == (0, 3) and scratch == 3
    else:
        assert (allocs, reuses, scratch) == (3, 0, 0)


def test_two_alternating_sizes_both_reuse(buffer_cache):
    # The pool keeps the two most recent sizes, so a loop that alternates
    # (a kernel with and without an ancilla, say) never reallocates.
    _observe_sizes([18, 19])
    allocs, reuses, _ = _observe_sizes([18, 19] * 2)
    assert (allocs, reuses) == ((0, 4) if buffer_cache else (4, 0))
