"""Batched observe: the Metal backend evaluates a whole Hamiltonian in one
pass (canHandleObserve), on the GPU or on the host depending on size.

Every check runs on both devices (MSV_OBSERVE_DEVICE=gpu|host). References:
  * numpy on the SAME Metal state, applying each Pauli as a 2x2 matrix on its
    tensor axis (no bit tricks shared with the backend). The host path must
    match it to ~1e-9 (both are double over the same fp32 amplitudes); the GPU
    path to fp32 accumulation error.
  * CUDA-Q's qpp-cpu target, for totals and per-term results (fp64 state).
"""
import functools
import json
import operator
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

import cudaq_metal  # noqa: F401  (registers the target; must precede cudaq)
import cudaq
from cudaq import spin

from _util import on

REPO = Path(__file__).resolve().parents[1]
TOL = {"host": 1e-9, "gpu": 2e-6}  # per-term, vs numpy on the Metal state


@cudaq.kernel
def scrambler(n: int, layers: int, angles: list[float]):
    q = cudaq.qvector(n)
    for l in range(layers):
        for i in range(n):
            rx(angles[3 * (l * n + i)], q[i])
            ry(angles[3 * (l * n + i) + 1], q[i])
            rz(angles[3 * (l * n + i) + 2], q[i])
        for i in range(n - 1):
            x.ctrl(q[i], q[i + 1])
        if n > 2:
            z.ctrl(q[n - 1], q[0])


@cudaq.kernel
def bell_plus(n: int):
    q = cudaq.qvector(n)
    h(q[0])
    x.ctrl(q[0], q[1])
    ry(0.3, q[2])


@pytest.fixture(params=["host", "gpu"])
def device(request, monkeypatch):
    monkeypatch.setenv("MSV_OBSERVE_DEVICE", request.param)
    return request.param


# ---- independent numpy reference -------------------------------------------
PAULI = {
    "X": np.array([[0, 1], [1, 0]], dtype=complex),
    "Y": np.array([[0, -1j], [1j, 0]], dtype=complex),
    "Z": np.array([[1, 0], [0, -1]], dtype=complex),
}


def apply_word(psi, n, word):
    """Apply a Pauli word ((qubit, 'X'|'Y'|'Z'), ...); qubit q = bit q."""
    t = psi.reshape([2] * n)  # C order: axis 0 is the highest bit
    for q, p in word:
        ax = n - 1 - q
        t = np.moveaxis(np.tensordot(PAULI[p], t, axes=([1], [ax])), 0, ax)
    return t.reshape(-1)


def np_expect(psi, n, word):
    return np.vdot(psi, apply_word(psi, n, word)).real / np.vdot(psi, psi).real


# ---- problem generation --------------------------------------------------------
def angles_for(n, layers, seed):
    return list(np.random.default_rng(seed).uniform(-np.pi, np.pi, 3 * n * layers))


def random_words(n, count, seed):
    rng = np.random.default_rng(seed + 1)
    words = set()
    while len(words) < count:
        k = n if rng.random() < 0.1 else int(rng.integers(1, min(n, 6) + 1))
        qs = rng.choice(n, size=k, replace=False)
        words.add(tuple(sorted((int(q), "XYZ"[int(rng.integers(3))]) for q in qs)))
    return sorted(words)


def word_op(word):
    return functools.reduce(operator.mul, [getattr(spin, p.lower())(q) for q, p in word])


def hamiltonian(words, coefs, identity=0.0):
    H = identity * spin.i(0)
    for w, c in zip(words, coefs):
        H += c * word_op(w)
    return H


QPP_MAX_QUBITS = 17  # an fp64 qpp reference beyond this costs tens of seconds


@functools.lru_cache(maxsize=None)
def problem(n, nterms, layers=3, seed=0):
    """(args, words, complex coefs, identity coef, fp64 qpp state or None,
    fp32 metal state)."""
    args = (n, layers, angles_for(n, layers, seed))
    words = random_words(n, nterms, seed)
    rng = np.random.default_rng(seed + 2)
    coefs = list(rng.normal(size=nterms) + 1j * rng.normal(size=nterms))
    ref = None
    if n <= QPP_MAX_QUBITS:
        ref = np.array(on("qpp-cpu", cudaq.get_state, scrambler, *args), copy=True)
        ref = ref.astype(np.complex128)
    met = np.array(on("metal", cudaq.get_state, scrambler, *args), copy=True)
    return args, words, coefs, 0.75, ref, met.astype(np.complex128)


# ---- tests ------------------------------------------------------------------------
CASES = [  # (n, terms): GPU perThread 1 (n<=15), 4 (17), 32 (20), 64 (n>=21);
    (1, 3), (2, 10), (3, 20), (5, 40), (8, 64),  # 1..13 batches of 8 terms;
    (12, 101), (17, 30), (20, 24), (21, 16),  # host multi-chunk at n>=17
]


@pytest.mark.parametrize("n,nterms", CASES)
def test_terms_match_numpy_and_qpp(device, n, nterms):
    args, words, coefs, c0, ref, met = problem(n, nterms)
    H = hamiltonian(words, coefs, c0)
    r = on("metal", cudaq.observe, scrambler, H, *args)

    exact_met = [np_expect(met, n, w) for w in words]
    for w, e in zip(words, exact_met):
        assert r.expectation(word_op(w)) == pytest.approx(e, abs=TOL[device]), w
    total_met = c0 + sum(c.real * e for c, e in zip(coefs, exact_met))
    scale = 1 + sum(abs(c.real) for c in coefs)
    assert r.expectation() == pytest.approx(total_met, abs=TOL[device] * scale)

    # And against an fp64 simulation (fp32 state error dominates here).
    if ref is not None:
        total_ref = c0 + sum(c.real * np_expect(ref, n, w) for c, w in zip(coefs, words))
        assert r.expectation() == pytest.approx(total_ref, abs=5e-5 * scale)


def test_every_y_phase(device):
    # nY mod 4 = 0..3 selects Re/Im and the sign; mix in X and Z.
    n = 6
    args, _, _, _, _, met = problem(n, 1, seed=5)
    words = [((0, "Y"),), ((0, "Y"), (1, "Y")), ((0, "Y"), (1, "Y"), (2, "Y")),
             ((0, "Y"), (1, "Y"), (2, "Y"), (3, "Y")),
             tuple((q, "Y") for q in range(5)), tuple((q, "Y") for q in range(6)),
             ((0, "X"), (1, "Y")), ((0, "Y"), (1, "Z")), ((2, "Y"), (5, "X")),
             ((0, "Y"), (1, "X"), (2, "Z"), (3, "Y"), (4, "Y"), (5, "X")),
             ((1, "X"), (4, "X")), ((3, "Z"), (5, "Z"))]
    H = hamiltonian(words, [1.0] * len(words))
    r = on("metal", cudaq.observe, scrambler, H, *args)
    for w in words:
        e = np_expect(met, n, w)
        assert r.expectation(word_op(w)) == pytest.approx(e, abs=TOL[device]), w


def test_per_term_results_match_qpp(device):
    H = (2.0 + 0.5 * spin.z(0) * spin.z(1) + 0.25 * spin.x(0) * spin.x(1)
         - 0.7 * spin.y(2) + (0.3 + 0.2j) * spin.z(2) + 0.4 * spin.x(2) * spin.y(1))
    ref = on("qpp-cpu", cudaq.observe, bell_plus, H, 3)
    got = on("metal", cudaq.observe, bell_plus, H, 3)
    assert got.expectation() == pytest.approx(ref.expectation(), abs=1e-6)
    for t in H:  # includes the identity term (0.0 on both: no per-term entry)
        assert got.expectation(t) == pytest.approx(ref.expectation(t), abs=1e-6), str(t)
    assert set(str(got.counts()).split()) == set(str(ref.counts()).split())


def test_identity_only_and_complex_coefficients(device):
    assert on("metal", cudaq.observe, bell_plus, 2.5 * spin.i(0), 3).expectation() == pytest.approx(2.5)
    # Only the real part of a coefficient contributes (as on every backend).
    H = 1.5 + (0.3 + 0.7j) * spin.z(0) * spin.z(1)
    assert on("metal", cudaq.observe, bell_plus, H, 3).expectation() == pytest.approx(1.8, abs=1e-6)


def test_qubits_outside_the_state_are_zero(device):
    def ev(H):
        return on("metal", cudaq.observe, bell_plus, H, 3).expectation()
    assert ev(spin.z(5)) == pytest.approx(1.0)  # Z|0> = |0>
    assert ev(spin.x(5)) == pytest.approx(0.0)  # <0|X|0> = 0
    assert ev(spin.y(7)) == pytest.approx(0.0)
    assert ev(spin.x(0) * spin.x(1) * spin.z(7)) == pytest.approx(ev(spin.x(0) * spin.x(1)))
    for H in (spin.z(5), spin.z(0) * spin.z(5), spin.x(0) * spin.x(1) * spin.z(7)):
        ref = on("qpp-cpu", cudaq.observe, bell_plus, H, 3).expectation()
        assert ev(H) == pytest.approx(ref, abs=1e-6)


def test_list_and_async(device):
    terms = [[((0, "Z"),)], [((0, "X"), (1, "X"))], [((0, "Y"), (1, "Y")), ((2, "Z"),)]]
    Hs = [hamiltonian(ws, [1.0, 0.5][:len(ws)]) for ws in terms]
    singles = [on("metal", cudaq.observe, bell_plus, H, 3) for H in Hs]
    listed = on("metal", cudaq.observe, bell_plus, Hs, 3)
    for ws, s, r in zip(terms, singles, listed):
        for w in ws:
            assert r.expectation(word_op(w)) == pytest.approx(
                s.expectation(word_op(w)), abs=1e-12)
    # CUDA-Q 0.16's list form drops term coefficients from the totals on every
    # backend (qpp-cpu too), so compare totals with qpp's list results.
    ref = on("qpp-cpu", cudaq.observe, bell_plus, Hs, 3)
    assert [r.expectation() for r in listed] == pytest.approx(
        [r.expectation() for r in ref], abs=1e-6)
    fut = on("metal", cudaq.observe_async, bell_plus, Hs[2], 3)
    assert fut.get().expectation() == pytest.approx(singles[2].expectation(), abs=1e-12)


def test_builder_kernel(device):
    # Classiq's qprog_to_cudaq_kernel emits kernel-builder kernels like this.
    kernel, thetas = cudaq.make_kernel(list)
    q = kernel.qalloc(4)
    for i in range(4):
        kernel.h(q[i])
    for i in range(3):
        kernel.cx(q[i], q[i + 1])
        kernel.rz(thetas[i], q[i + 1])
        kernel.cx(q[i], q[i + 1])
    for i in range(4):
        kernel.rx(thetas[3], q[i])
    H = sum((0.5 * spin.z(i) * spin.z(i + 1) for i in range(3)), -1.5 * spin.i(0))
    for p in ([0.1, 0.2, 0.3, 0.4], [1.0, -0.5, 2.0, 0.7]):
        ref = on("qpp-cpu", cudaq.observe, kernel, H, p).expectation()
        assert on("metal", cudaq.observe, kernel, H, p).expectation() == pytest.approx(ref, abs=1e-5)


def test_gpu_scratch_chunking_is_exact(monkeypatch):
    # Force one 8-term batch per dispatch: per-term sums must not change at all.
    monkeypatch.setenv("MSV_OBSERVE_DEVICE", "gpu")
    args, words, coefs, c0, _, met = problem(12, 101)
    H = hamiltonian(words, coefs, c0)

    def per_term():
        r = on("metal", cudaq.observe, scrambler, H, *args)
        return [r.expectation(word_op(w)) for w in words]

    whole = per_term()
    monkeypatch.setenv("MSV_PAULI_SCRATCH_BYTES", "1")
    chunked = per_term()
    assert chunked == whole
    for w, e in zip(words, chunked):
        assert e == pytest.approx(np_expect(met, 12, w), abs=TOL["gpu"])


def test_gpu_is_deterministic(monkeypatch):
    monkeypatch.setenv("MSV_OBSERVE_DEVICE", "gpu")
    args, words, coefs, c0, _, _ = problem(17, 30)
    H = hamiltonian(words, coefs, c0)
    vals = {on("metal", cudaq.observe, scrambler, H, *args).expectation() for _ in range(10)}
    assert len(vals) == 1


def test_shot_based_observe_still_samples():
    # With shots, CUDA-Q must keep its term-by-term sampling (real counts).
    cudaq.set_random_seed(13)
    H = 0.5 * spin.z(0) * spin.z(1) + 0.8 * spin.x(0) * spin.x(1) - 0.3 * spin.z(2)
    exact = on("qpp-cpu", cudaq.observe, bell_plus, H, 3).expectation()
    shots = 4000
    r = on("metal", cudaq.observe, bell_plus, H, 3, shots_count=shots)
    zz = r.counts(spin.z(0) * spin.z(1))
    assert sum(zz.count(b) for b in zz) == shots
    assert set(zz) <= {"00", "11"}
    sigma = (0.5 + 0.8 + 0.3) / np.sqrt(shots)
    assert abs(r.expectation() - exact) < 5 * sigma


_PATH_SCRIPT = """
import json, sys
import cudaq_metal, cudaq
from cudaq import spin

@cudaq.kernel
def k(n: int):
    q = cudaq.qvector(n)
    for i in range(n):
        ry(0.3 + 0.2 * i, q[i])
        rz(0.1 * i, q[i])
    for i in range(n - 1):
        x.ctrl(q[i], q[i + 1])

cudaq.set_target("metal")
H = (0.5 * spin.z(0) * spin.z(3) + 0.7 * spin.x(1) * spin.x(2)
     - 0.3 * spin.y(0) * spin.y(4) + 0.2 * spin.x(5) * spin.y(6) * spin.z(7) + 1.0)
r = cudaq.observe(k, H, 8)
print("RESULT", json.dumps([r.expectation()] + [r.expectation(t) for t in H]))
"""


def _run_path_script(tmp_path, **env):
    script = tmp_path / "observe_path.py"
    script.write_text(_PATH_SCRIPT)
    # Import the same cudaq_metal as this process: a different installed copy
    # would install its own (older) backend binary on import.
    pkg_root = str(Path(cudaq_metal.__file__).resolve().parents[1])
    path = os.pathsep.join(filter(None, [pkg_root, os.environ.get("PYTHONPATH")]))
    full_env = dict(os.environ, CUDAQ_LOG_LEVEL="info", PYTHONPATH=path, **env)
    full_env.pop("MSV_OBSERVE_DEVICE", None)
    p = subprocess.run([sys.executable, str(script)], cwd=REPO, env=full_env,
                       capture_output=True, text=True, timeout=300)
    assert p.returncode == 0, p.stdout[-2000:] + p.stderr[-2000:]
    out = p.stdout + p.stderr
    values = json.loads(next(l for l in out.splitlines() if l.startswith("RESULT"))[7:])
    return out, values


def test_batched_path_is_used_and_matches_term_by_term(tmp_path):
    # CUDA-Q logs "Measure <term>" for each term on its term-by-term path.
    batched_log, batched = _run_path_script(tmp_path)
    assert "Measure" not in batched_log, "CUDA-Q did not hand observe to the backend"
    sampled_log, sampled = _run_path_script(tmp_path, CUDAQ_OBSERVE_FROM_SAMPLING="1")
    assert sampled_log.count("Measure") >= 4, "CUDAQ_OBSERVE_FROM_SAMPLING=1 was ignored"
    assert batched == pytest.approx(sampled, abs=1e-6)
