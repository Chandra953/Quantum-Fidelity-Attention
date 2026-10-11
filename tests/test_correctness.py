"""Correctness tests for the quantum attention layer, its simulator and CPLA.

Run from the repository root:  python -m pytest -q tests/
"""
import math
import os
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from quantum_attention import (QuantumAttentionLayer, QuantumOverlapEstimator,  # noqa: E402
                               QuantumOverlapFunction, feature_map, hardware_efficient_ansatz)
from experiments import simulate  # noqa: E402
import exp2  # noqa: E402

torch.set_default_dtype(torch.float64)


# ---------------------------------------------------------------------------
# Qiskit layer
# ---------------------------------------------------------------------------

def test_estimator_matches_statevector():
    """Estimator V2 overlap equals the exact |<psi_K|psi_Q>|^2 from Statevector."""
    from qiskit.quantum_info import Statevector
    n, L = 4, 2
    be = QuantumOverlapEstimator(n, L)
    rng = np.random.default_rng(1)
    a, b = rng.uniform(-np.pi, np.pi, n), rng.uniform(-np.pi, np.pi, n)
    tq, tk = rng.uniform(0, 2 * np.pi, n * L), rng.uniform(0, 2 * np.pi, n * L)

    def state(x, th):
        s, xv = feature_map(n, "x")
        w, tv = hardware_efficient_ansatz(n, L, "t")
        qc = s.compose(w).assign_parameters({**dict(zip(xv, x)), **dict(zip(tv, th))})
        return Statevector(qc)

    exact = abs(state(b, tk).inner(state(a, tq))) ** 2
    est = be.run(np.concatenate([a, tq, tk, b])[None, :])[0]
    assert abs(est - exact) < 1e-10


def test_parameter_shift_matches_finite_differences():
    n, L = 3, 2
    be = QuantumOverlapEstimator(n, L)
    torch.manual_seed(0)
    aq, ak = torch.rand(2, n), torch.rand(2, n)
    tq = torch.rand(n * L, requires_grad=True)
    tk = torch.rand(n * L)
    QuantumOverlapFunction.apply(aq, ak, tq, tk, be).sum().backward()
    eps = 1e-5
    for i in range(n * L):
        tp, tm = tq.detach().clone(), tq.detach().clone()
        tp[i] += eps
        tm[i] -= eps
        f = lambda t: QuantumOverlapFunction.apply(aq, ak, t, tk, be).sum().item()  # noqa: E731
        assert abs(tq.grad[i].item() - (f(tp) - f(tm)) / (2 * eps)) < 1e-6


def test_layer_output_shape_and_attention_rows():
    torch.manual_seed(0)
    layer = QuantumAttentionLayer(embedding_dim=8, n_qubits=2, n_layers=1)
    x = torch.randn(2, 3, 8)
    out, attn = layer(x, return_attn=True)
    assert out.shape == (2, 3, 8)
    assert torch.allclose(attn.sum(-1), torch.ones(2, 3))


# ---------------------------------------------------------------------------
# Statevector simulator used by the experiments
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("n,L", [(2, 1), (2, 2), (3, 2), (4, 2)])
def test_simulator_matches_qiskit(n, L):
    be = QuantumOverlapEstimator(n, L)
    rng = np.random.default_rng(n * 10 + L)
    a = torch.tensor(rng.uniform(-np.pi, np.pi, (4, n)))
    b = torch.tensor(rng.uniform(-np.pi, np.pi, (4, n)))
    tq = torch.tensor(rng.uniform(0, 2 * np.pi, n * L))
    tk = torch.tensor(rng.uniform(0, 2 * np.pi, n * L))
    qiskit_scores = QuantumOverlapFunction.apply(a, b, tq, tk, be)
    sim_scores = (simulate(a, tq, n, L) * simulate(b, tk, n, L)).sum(1) ** 2
    assert torch.allclose(qiskit_scores, sim_scores, atol=1e-12)


def test_simulator_states_are_normalized():
    phi = torch.rand(5, 4) * 2 * math.pi
    psi = simulate(phi, torch.rand(8), 4, 2)
    assert torch.allclose(psi.norm(dim=1), torch.ones(5))


# ---------------------------------------------------------------------------
# Propositions 1 and 2, and the CPLA layer
# ---------------------------------------------------------------------------

def test_proposition1_fidelity_equals_cosine_product():
    """Without an ansatz the fidelity is prod_k cos^2((a_k - b_k)/2)."""
    phi = (torch.rand(6, 4) * 2 - 1) * math.pi
    psi = simulate(phi, torch.zeros(4), 4, 1)   # zero angles: W = CNOT ring, which cancels in W^T W
    fid = (psi @ psi.T) ** 2
    cp = torch.cos((phi[:, None] - phi[None]) / 2).pow(2).prod(-1)
    assert torch.allclose(fid, cp, atol=1e-12)


def test_proposition2_explicit_feature_map():
    assert exp2.check_feature_map() < 1e-12


def test_cpla_equals_pairwise_normalized_kernel():
    """Linear-time CPLA equals explicit pairwise sum_j s_ij v_j / sum_j s_ij."""
    torch.manual_seed(0)
    layer = exp2.CosineAttention(8, n=3, linear=True)
    x = torch.randn(2, 5, 8)
    y = layer(x)
    phi = math.pi * torch.tanh(layer.to_angles(x))
    s = torch.cos((phi.unsqueeze(2) - phi.unsqueeze(1)) / 2).pow(2).prod(-1)
    ref = layer.out((s @ layer.value(x)) / (s.sum(-1, keepdim=True) + 1e-6))
    assert torch.allclose(y, ref, atol=1e-8)


def test_padding_mask_is_respected():
    """Changing padded tokens must not change outputs at real positions."""
    torch.manual_seed(0)
    for name in ["dot16", "cosine", "cosine-linear", "quantum", "elu-linear"]:
        layer = exp2.make_attn(name, 8)
        x = torch.randn(1, 5, 8)
        mask = torch.tensor([[True, True, True, False, False]])
        x2 = x.clone()
        x2[:, 3:] = torch.randn(1, 2, 8) * 10
        assert torch.allclose(layer(x, mask)[:, :3], layer(x2, mask)[:, :3], atol=1e-8), name
