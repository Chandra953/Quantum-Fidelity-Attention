"""
Quantum-Accelerated Transformers: a hybrid Quantum Attention Layer
===================================================================

Reference implementation for "Appendix A: Simulated Implementation of
Quantum-Accelerated Attention" (PyTorch + Qiskit >= 1.0).

Core idea
---------
Classical dot-product attention scores token i against token j with
    score_ij = <q_i, k_j> / sqrt(d)
Here that similarity is replaced by a *quantum state overlap* (fidelity):
    score_ij = |<psi_K(x_j) | psi_Q(x_i)>|^2
where
    |psi_Q(x)> = W_Q(theta_Q) S(x) |0...0>     (query state)
    |psi_K(x)> = W_K(theta_K) S(x) |0...0>     (key state)
S(x) is an angle-encoding feature map and W_Q / W_K are trainable
hardware-efficient ansatzes (RY layers + CNOT ring). They play the role of
the W_Q / W_K projection matrices in a classical Transformer.

The overlap is estimated with the "compute-uncompute" circuit
    U_ij = S(x_j)^dag W_K^dag W_Q S(x_i)
    score_ij = <0| U_ij^dag  P0  U_ij |0>,   P0 = |0...0><0...0|
which is exactly the expectation value of the projector P0, so it is
measured with Qiskit's modern Estimator primitive (StatevectorEstimator,
V2 interface). Gradients are computed with the exact parameter-shift rule
inside a custom torch.autograd.Function, so PyTorch backpropagates through
the quantum circuit into both the quantum parameters and the upstream
classical layers.

NOTE ON COMPLEXITY (important for the paper): this layer evaluates one
overlap per (i, j) token pair, i.e. N^2 circuit evaluations per sequence,
and each one on real hardware needs many shots. It demonstrates that a
quantum similarity kernel can be trained end-to-end inside attention; it
does NOT demonstrate an asymptotic speedup over classical attention.
"""

from __future__ import annotations

import math
import time
from functools import reduce

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.primitives import StatevectorEstimator
from qiskit.quantum_info import SparsePauliOp


# ---------------------------------------------------------------------------
# 1. Circuit construction
# ---------------------------------------------------------------------------

def feature_map(n_qubits: int, prefix: str) -> tuple[QuantumCircuit, ParameterVector]:
    """Angle encoding S(x): one RY rotation per qubit.

    Classical -> Hilbert space transition happens HERE: a real vector
    x in R^n (after the classical projection) becomes rotation angles, and
    the n-qubit register moves from |0...0> to the product state
        ⊗_k ( cos(x_k/2)|0> + sin(x_k/2)|1> ),
    a unit vector in the 2^n-dimensional complex Hilbert space C^(2^n).
    """
    x = ParameterVector(prefix, n_qubits)
    qc = QuantumCircuit(n_qubits, name=f"S({prefix})")
    for k in range(n_qubits):
        qc.ry(x[k], k)
    return qc, x


def hardware_efficient_ansatz(n_qubits: int, n_layers: int, prefix: str
                              ) -> tuple[QuantumCircuit, ParameterVector]:
    """W(theta): alternating layers of trainable RY rotations and a CNOT ring.

    The CNOT ring entangles the qubits, so the state is no longer a product
    state: this is what lets the model use the full 2^n-dim space rather than
    n independent 2-dim spaces.
    """
    theta = ParameterVector(prefix, n_qubits * n_layers)
    qc = QuantumCircuit(n_qubits, name=f"W({prefix})")
    for layer in range(n_layers):
        for k in range(n_qubits):
            qc.ry(theta[layer * n_qubits + k], k)
        for k in range(n_qubits):
            qc.cx(k, (k + 1) % n_qubits)
    return qc, theta


def build_overlap_circuit(n_qubits: int, n_layers: int):
    """Compute-uncompute circuit U = S(x_k)^dag W_K^dag W_Q S(x_q).

    Returns the circuit plus the parameter vectors in a fixed logical order
    (x_q, theta_q, theta_k, x_k).
    """
    s_q, x_q = feature_map(n_qubits, "xq")
    s_k, x_k = feature_map(n_qubits, "xk")
    w_q, th_q = hardware_efficient_ansatz(n_qubits, n_layers, "tq")
    w_k, th_k = hardware_efficient_ansatz(n_qubits, n_layers, "tk")

    qc = QuantumCircuit(n_qubits)
    qc.compose(s_q, inplace=True)              # encode query token
    qc.compose(w_q, inplace=True)              # query "projection"
    qc.compose(w_k.inverse(), inplace=True)    # un-apply key "projection"
    qc.compose(s_k.inverse(), inplace=True)    # un-encode key token
    return qc, (x_q, th_q, th_k, x_k)


def zero_projector(n_qubits: int) -> SparsePauliOp:
    """P0 = |0..0><0..0| = ⊗ (I + Z)/2, expressed as a Pauli sum for the Estimator."""
    single = SparsePauliOp.from_list([("I", 0.5), ("Z", 0.5)])
    return reduce(lambda a, b: a.tensor(b), [single] * n_qubits).simplify()


# ---------------------------------------------------------------------------
# 2. Quantum backend wrapper (Estimator + parameter-shift gradients)
# ---------------------------------------------------------------------------

class QuantumOverlapEstimator:
    """Evaluates batches of overlaps with the Estimator V2 primitive.

    All circuit evaluations for a forward or backward pass are sent as ONE
    primitive unified bloc (PUB) with a 2-D parameter array, which is how
    Qiskit >= 1.0 broadcasts a single circuit over many parameter bindings.
    """

    def __init__(self, n_qubits: int, n_layers: int, precision: float = 0.0, seed=None):
        self.n_qubits, self.n_layers = n_qubits, n_layers
        self.circuit, vecs = build_overlap_circuit(n_qubits, n_layers)
        self.observable = zero_projector(n_qubits)
        # precision=0 -> exact statevector expectation values.
        # precision>0 -> adds shot-like Gaussian noise (≈ 1/sqrt(shots)).
        self.estimator = StatevectorEstimator(default_precision=precision, seed=seed)

        # Map our logical parameter order onto Qiskit's internal ordering.
        logical = [p for v in vecs for p in v]
        circuit_order = list(self.circuit.parameters)
        self._perm = [logical.index(p) for p in circuit_order]
        self.sizes = [len(v) for v in vecs]          # [n, n*L, n*L, n]

    def run(self, values: np.ndarray) -> np.ndarray:
        """values: (K, P) in logical order -> overlaps (K,)."""
        values = values[:, self._perm]
        job = self.estimator.run([(self.circuit, self.observable, values)])
        evs = np.asarray(job.result()[0].data.evs, dtype=np.float64)
        return np.clip(evs, 0.0, 1.0)


class QuantumOverlapFunction(torch.autograd.Function):
    """Custom autograd bridge: PyTorch <-> Qiskit.

    forward : fidelity |<psi_K(x_k)|psi_Q(x_q)>|^2 for P token pairs
    backward: exact parameter-shift rule. Every parameter enters the circuit
              through a single RY (generator eigenvalues ±1/2), so
              d f / d a = [ f(a + pi/2) - f(a - pi/2) ] / 2   exactly.
    """

    @staticmethod
    def forward(ctx, ang_q, ang_k, theta_q, theta_k, backend: QuantumOverlapEstimator):
        P = ang_q.shape[0]
        tq = theta_q.detach().expand(P, -1)
        tk = theta_k.detach().expand(P, -1)
        values = torch.cat([ang_q.detach(), tq, tk, ang_k.detach()], dim=1)
        values = values.double().cpu().numpy()
        out = backend.run(values)
        ctx.backend = backend
        ctx.save_for_backward(torch.from_numpy(values))
        return torch.from_numpy(out).to(ang_q.dtype)

    @staticmethod
    def backward(ctx, grad_out):
        (values,) = ctx.saved_tensors
        backend = ctx.backend
        P, M = values.shape
        v = values.numpy()

        # Build 2*M shifted copies of every pair in one batch: (P, 2M, M)
        shifts = np.concatenate([np.eye(M), -np.eye(M)], axis=0) * (math.pi / 2)
        batch = (v[:, None, :] + shifts[None, :, :]).reshape(-1, M)
        f = backend.run(batch).reshape(P, 2 * M)
        jac = 0.5 * (f[:, :M] - f[:, M:])                     # (P, M)

        g = torch.from_numpy(jac).to(grad_out.dtype) * grad_out[:, None]
        n, nl, _, _ = backend.sizes
        g_ang_q = g[:, :n]
        g_theta_q = g[:, n:n + nl].sum(0)                     # shared across pairs
        g_theta_k = g[:, n + nl:n + 2 * nl].sum(0)
        g_ang_k = g[:, n + 2 * nl:]
        return g_ang_q, g_ang_k, g_theta_q, g_theta_k, None


# ---------------------------------------------------------------------------
# 3. The PyTorch module
# ---------------------------------------------------------------------------

class QuantumAttentionLayer(nn.Module):
    """Drop-in single-head attention whose Q·K similarity is a quantum overlap.

    Input : (batch, seq_len, embedding_dim)
    Output: (batch, seq_len, embedding_dim)  -- same shape as nn.MultiheadAttention
    """

    def __init__(self, embedding_dim: int, n_qubits: int = 4, n_layers: int = 2,
                 causal: bool = False, precision: float = 0.0, seed=None):
        super().__init__()
        self.n_qubits, self.causal = n_qubits, causal

        # Classical compression: R^embedding_dim -> R^n_qubits.
        # Angle encoding needs one feature per qubit, so the embedding is
        # first projected down to n_qubits real numbers.
        self.to_angles = nn.Linear(embedding_dim, n_qubits)
        self.value = nn.Linear(embedding_dim, embedding_dim)
        self.out_proj = nn.Linear(embedding_dim, embedding_dim)

        # Trainable quantum parameters (the "quantum weights").
        self.theta_q = nn.Parameter(0.1 * torch.randn(n_qubits * n_layers))
        self.theta_k = nn.Parameter(0.1 * torch.randn(n_qubits * n_layers))
        # Fidelities live in [0, 1]; a learnable inverse temperature turns
        # them into sharp-enough softmax logits.
        self.log_scale = nn.Parameter(torch.tensor(math.log(5.0)))

        self.backend = QuantumOverlapEstimator(n_qubits, n_layers, precision, seed)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Classical vector -> rotation angles in (-pi, pi)."""
        return math.pi * torch.tanh(self.to_angles(x))

    def quantum_scores(self, x: torch.Tensor) -> torch.Tensor:
        """(B, N, D) -> (B, N, N) matrix of quantum overlaps."""
        B, N, _ = x.shape
        ang = self.encode(x)                                   # (B, N, n)
        ang_q = ang.unsqueeze(2).expand(B, N, N, -1).reshape(-1, self.n_qubits)
        ang_k = ang.unsqueeze(1).expand(B, N, N, -1).reshape(-1, self.n_qubits)
        s = QuantumOverlapFunction.apply(ang_q, ang_k, self.theta_q, self.theta_k,
                                         self.backend)
        return s.view(B, N, N)

    def forward(self, x: torch.Tensor, return_attn: bool = False):
        B, N, _ = x.shape
        scores = self.quantum_scores(x) * self.log_scale.exp()
        if self.causal:
            mask = torch.triu(torch.ones(N, N, dtype=torch.bool), diagonal=1)
            scores = scores.masked_fill(mask, float("-inf"))
        attn = F.softmax(scores, dim=-1)                       # (B, N, N)
        out = self.out_proj(attn @ self.value(x))              # (B, N, D)
        return (out, attn) if return_attn else out


class QuantumTransformerBlock(nn.Module):
    """Standard pre-norm Transformer block with the quantum attention inside."""

    def __init__(self, d: int, n_qubits: int = 4, n_layers: int = 2):
        super().__init__()
        self.ln1, self.ln2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.attn = QuantumAttentionLayer(d, n_qubits, n_layers)
        self.mlp = nn.Sequential(nn.Linear(d, 2 * d), nn.GELU(), nn.Linear(2 * d, d))

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        return x + self.mlp(self.ln2(x))


# ---------------------------------------------------------------------------
# 4. Sanity checks
# ---------------------------------------------------------------------------

def check_overlap_is_correct(layer: QuantumAttentionLayer):
    """Compare the Estimator result with a direct statevector calculation."""
    from qiskit.quantum_info import Statevector
    n, L = layer.n_qubits, len(layer.theta_q) // layer.n_qubits
    rng = np.random.default_rng(0)
    a, b = rng.uniform(-np.pi, np.pi, n), rng.uniform(-np.pi, np.pi, n)
    tq, tk = layer.theta_q.detach().numpy(), layer.theta_k.detach().numpy()

    def state(x, theta):
        s, xv = feature_map(n, "x"); w, tv = hardware_efficient_ansatz(n, L, "t")
        qc = s.compose(w)
        qc = qc.assign_parameters({**dict(zip(xv, x)), **dict(zip(tv, theta))})
        return Statevector(qc)

    exact = abs(state(b, tk).inner(state(a, tq))) ** 2
    est = layer.backend.run(np.concatenate([a, tq, tk, b])[None, :])[0]
    print(f"[check] overlap  Estimator={est:.6f}  statevector={exact:.6f}")
    assert abs(est - exact) < 1e-8


def check_gradients(layer: QuantumAttentionLayer):
    """Parameter-shift gradient vs. central finite differences."""
    torch.manual_seed(1)
    aq = torch.rand(3, layer.n_qubits, dtype=torch.float64, requires_grad=True)
    ak = torch.rand(3, layer.n_qubits, dtype=torch.float64, requires_grad=True)
    tq = layer.theta_q.detach().double().requires_grad_()
    tk = layer.theta_k.detach().double().requires_grad_()
    QuantumOverlapFunction.apply(aq, ak, tq, tk, layer.backend).sum().backward()

    eps, i = 1e-5, 2
    def f(t):
        return QuantumOverlapFunction.apply(aq.detach(), ak.detach(), t, tk.detach(),
                                            layer.backend).sum().item()
    tp, tm = tq.detach().clone(), tq.detach().clone()
    tp[i] += eps; tm[i] -= eps
    fd = (f(tp) - f(tm)) / (2 * eps)
    print(f"[check] dL/dtheta_q[{i}]  param-shift={tq.grad[i].item():.6f}  "
          f"finite-diff={fd:.6f}")
    assert abs(tq.grad[i].item() - fd) < 1e-5


# ---------------------------------------------------------------------------
# 5. Mock training loop
# ---------------------------------------------------------------------------

def make_dummy_dataset(n_samples: int, seq_len: int, d: int, vocab: int, seed: int = 0):
    """Toy sequence task that *requires* token-to-token comparison.

    Each sequence is built from a small random "vocabulary" of embeddings.
    Label = 1 if some token appears twice in the sequence, else 0.
    A model can only solve this by comparing tokens with each other, which
    is exactly what the attention scores do.
    """
    g = torch.Generator().manual_seed(seed)
    table = torch.randn(vocab, d, generator=g)
    X, y = [], []
    for k in range(n_samples):
        if k % 2 == 0:      # all distinct
            ids = torch.randperm(vocab, generator=g)[:seq_len]
        else:               # force one repeat
            ids = torch.randperm(vocab, generator=g)[:seq_len - 1]
            ids = torch.cat([ids, ids[torch.randint(0, seq_len - 1, (1,), generator=g)]])
            ids = ids[torch.randperm(seq_len, generator=g)]
        X.append(table[ids] + 0.05 * torch.randn(seq_len, d, generator=g))
        y.append(float(k % 2))
    return torch.stack(X), torch.tensor(y)


class ToyClassifier(nn.Module):
    def __init__(self, d: int, n_qubits: int, n_layers: int):
        super().__init__()
        self.attn = QuantumAttentionLayer(d, n_qubits, n_layers)
        self.ln = nn.LayerNorm(d)
        self.head = nn.Sequential(nn.Linear(d, 16), nn.GELU(), nn.Linear(16, 1))

    def forward(self, x):
        h = self.ln(x + self.attn(x))
        return self.head(h.max(dim=1).values).squeeze(-1)


def train():
    torch.manual_seed(0)
    D, N_TOKENS, N_QUBITS, N_LAYERS = 8, 4, 4, 2
    X, y = make_dummy_dataset(n_samples=16, seq_len=N_TOKENS, d=D, vocab=8)

    model = ToyClassifier(D, N_QUBITS, N_LAYERS)
    check_overlap_is_correct(model.attn)
    check_gradients(model.attn)

    opt = torch.optim.Adam(model.parameters(), lr=0.03)
    theta0 = torch.cat([model.attn.theta_q, model.attn.theta_k]).detach().clone()
    batch_size, epochs = 8, 20

    print(f"\nTraining: {len(X)} sequences, seq_len={N_TOKENS}, embed_dim={D}, "
          f"{N_QUBITS} qubits, {N_LAYERS} ansatz layers")
    print(f"Quantum circuit: depth={model.attn.backend.circuit.depth()}, "
          f"params/circuit={model.attn.backend.circuit.num_parameters}\n")

    history = []
    for epoch in range(1, epochs + 1):
        t0, perm = time.time(), torch.randperm(len(X))
        tot_loss, correct, qgrad = 0.0, 0, 0.0
        for s in range(0, len(X), batch_size):
            idx = perm[s:s + batch_size]
            logits = model(X[idx])
            loss = F.binary_cross_entropy_with_logits(logits, y[idx])
            opt.zero_grad()
            loss.backward()               # gradients flow THROUGH the quantum circuit
            qgrad += torch.cat([model.attn.theta_q.grad, model.attn.theta_k.grad]).norm().item()
            opt.step()
            tot_loss += loss.item() * len(idx)
            correct += ((logits > 0).float() == y[idx]).sum().item()
        avg = tot_loss / len(X)
        history.append(avg)
        drift = (torch.cat([model.attn.theta_q, model.attn.theta_k]) - theta0).norm().item()
        print(f"epoch {epoch:2d} | loss {avg:.4f} | acc {correct / len(X):.2f} | "
              f"|grad θ_quantum| {qgrad:.4f} | |Δθ_quantum| {drift:.4f} | "
              f"{time.time() - t0:.1f}s")

    print(f"\nLoss: {history[0]:.4f} -> {history[-1]:.4f}")
    print(f"Quantum parameters moved by |Δθ| = {drift:.4f} from initialization")

    with torch.no_grad():
        _, attn = model.attn(X[1:2], return_attn=True)
    print("\nLearned quantum attention for one sequence that contains a repeat:")
    print(np.array2string(attn[0].numpy(), precision=2, suppress_small=True))
    return history


if __name__ == "__main__":
    train()
