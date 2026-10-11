"""
Experiments for "Trainable Quantum-Kernel Attention".

Uses an exact PyTorch statevector simulator of the SAME circuit as
quantum_attention.py (RY angle encoding, RY+CNOT-ring ansatz, fidelity
score). All gates are real (RY, CNOT), so states are real vectors and
autograd gives exact gradients. Equivalence with the Qiskit Estimator +
parameter-shift implementation is checked in `validate()`.

Outputs CSV files in ./results/.
"""
import csv
import math
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.set_default_dtype(torch.float64)
OUT = "results"
os.makedirs(OUT, exist_ok=True)


# ---------------------------------------------------------------------------
# Exact statevector simulator (real amplitudes)
# ---------------------------------------------------------------------------

def _ry(state, angle, k, n):
    """Apply RY(angle) on qubit k. state: (M, 2,...,2); angle: (M,) or scalar."""
    s = state.movedim(k + 1, 1)
    a0, a1 = s[:, 0], s[:, 1]
    shape = (-1,) + (1,) * (n - 1)
    c = torch.cos(angle / 2).reshape(shape) if torch.is_tensor(angle) else math.cos(angle / 2)
    si = torch.sin(angle / 2).reshape(shape) if torch.is_tensor(angle) else math.sin(angle / 2)
    out = torch.stack([c * a0 - si * a1, si * a0 + c * a1], dim=1)
    return out.movedim(1, k + 1)


def _cx(state, c, t, n):
    s = state.movedim((c + 1, t + 1), (1, 2))
    s = torch.stack([s[:, 0], s[:, 1].flip(1)], dim=1)
    return s.movedim((1, 2), (c + 1, t + 1))


def simulate(angles, theta, n, L):
    """|psi> = W(theta) S(angles) |0>.  angles: (M, n), theta: (n*L,). Returns (M, 2^n)."""
    M = angles.shape[0]
    st = torch.zeros((M,) + (2,) * n, dtype=angles.dtype)
    st.view(M, -1)[:, 0] = 1.0
    for k in range(n):
        st = _ry(st, angles[:, k], k, n)
    for l in range(L):
        for k in range(n):
            st = _ry(st, theta[l * n + k].expand(M), k, n)
        if n > 1:
            for k in range(n):                  # CNOT ring, identical to the Qiskit ansatz
                st = _cx(st, k, (k + 1) % n, n)
    return st.reshape(M, -1)


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

class QuantumKernelAttention(nn.Module):
    def __init__(self, d, n=4, L=2, train_theta=True, shared=False, noise=0.0):
        super().__init__()
        self.n, self.L, self.shared, self.noise = n, L, shared, noise
        self.to_angles = nn.Linear(d, n)
        self.value, self.out = nn.Linear(d, d), nn.Linear(d, d)
        self.theta_q = nn.Parameter(0.1 * torch.randn(n * L), requires_grad=train_theta)
        self.theta_k = nn.Parameter(0.1 * torch.randn(n * L), requires_grad=train_theta and not shared)
        self.log_scale = nn.Parameter(torch.tensor(math.log(5.0)))
        self.test_noise = 0.0

    def scores(self, x):
        B, N, _ = x.shape
        ang = math.pi * torch.tanh(self.to_angles(x)).reshape(B * N, self.n)
        q = simulate(ang, self.theta_q, self.n, self.L).reshape(B, N, -1)
        k = simulate(ang, self.theta_q if self.shared else self.theta_k, self.n, self.L).reshape(B, N, -1)
        s = (q @ k.transpose(1, 2)) ** 2                       # fidelity |<k_j|q_i>|^2
        eps = self.noise if self.training else self.test_noise
        if eps > 0:                                            # shot noise on each estimate
            s = (s + eps * torch.randn_like(s).detach()).clamp(0, 1)
        return s

    def forward(self, x):
        a = F.softmax(self.scores(x) * self.log_scale.exp(), dim=-1)
        return self.out(a @ self.value(x))


class ClassicalAttention(nn.Module):
    def __init__(self, d, dk=4):
        super().__init__()
        self.q, self.k = nn.Linear(d, dk), nn.Linear(d, dk)
        self.value, self.out = nn.Linear(d, d), nn.Linear(d, d)
        self.dk = dk

    def forward(self, x):
        s = self.q(x) @ self.k(x).transpose(1, 2) / math.sqrt(self.dk)
        return self.out(F.softmax(s, -1) @ self.value(x))


class ClassicalKernelAttention(nn.Module):
    """Classical similarity kernels that share the fidelity's shape.

    kind='cosine': s_ij = prod_k cos^2((phi_ik - phi_jk)/2) -- exactly the
                   fidelity of the angle-encoded product state with no ansatz,
                   computed classically.
    kind='rbf'   : s_ij = exp(-||q_i - k_j||^2), separate 4-dim projections.
    """
    def __init__(self, d, n=4, kind="cosine"):
        super().__init__()
        self.kind = kind
        if kind == "cosine":
            self.to_angles = nn.Linear(d, n)
        else:
            self.q, self.k = nn.Linear(d, n), nn.Linear(d, n)
        self.value, self.out = nn.Linear(d, d), nn.Linear(d, d)
        self.log_scale = nn.Parameter(torch.tensor(math.log(5.0)))

    def forward(self, x):
        if self.kind == "cosine":
            phi = math.pi * torch.tanh(self.to_angles(x))
            s = torch.cos((phi.unsqueeze(2) - phi.unsqueeze(1)) / 2).pow(2).prod(-1)
        else:
            s = torch.exp(-(self.q(x).unsqueeze(2) - self.k(x).unsqueeze(1)).pow(2).sum(-1))
        a = F.softmax(s * self.log_scale.exp(), dim=-1)
        return self.out(a @ self.value(x))


class NoAttention(nn.Module):
    def forward(self, x):
        return torch.zeros_like(x)


class Classifier(nn.Module):
    def __init__(self, d, attn):
        super().__init__()
        self.attn, self.ln = attn, nn.LayerNorm(d)
        self.pos = nn.Parameter(0.1 * torch.randn(8, d))       # learned positions, N <= 8
        self.head = nn.Sequential(nn.Linear(d, 32), nn.GELU(), nn.Linear(32, 1))

    def forward(self, x):
        x = x + self.pos[: x.shape[1]]
        h = self.ln(x + self.attn(x))
        return self.head(h.max(1).values).squeeze(-1)


def attn_params(m):
    return sum(p.numel() for p in m.attn.parameters() if p.requires_grad)


# ---------------------------------------------------------------------------
# Tasks (each requires comparing tokens with each other)
# ---------------------------------------------------------------------------

def make_task(name, n_samples, seed, split="train", d=16, vocab=12, noise=0.1):
    """Same token vocabulary (embedding table) for train and test of a given
    seed; the sequences themselves are drawn independently per split."""
    table = torch.randn(vocab, d, generator=torch.Generator().manual_seed(1000 + seed))
    g = torch.Generator().manual_seed(2000 + seed + (0 if split == "train" else 500))
    X, y = [], []
    for i in range(n_samples):
        lab = i % 2
        if name == "repeat":            # does any token occur twice? (N=6)
            N = 6
            ids = torch.randperm(vocab, generator=g)[:N]
            if lab:
                a, b = torch.randperm(N, generator=g)[:2]
                ids[b] = ids[a]
        elif name == "match-first":     # does token 0 occur again later? (N=6)
            N = 6
            ids = torch.randperm(vocab, generator=g)[:N]
            if lab:
                j = int(torch.randint(1, N, (1,), generator=g))
                ids[j] = ids[0]
            else:                       # distractor: a repeat that does NOT involve token 0
                a, b = (torch.randperm(N - 1, generator=g)[:2] + 1)
                ids[b] = ids[a]
        elif name == "two-pairs":       # at least two distinct repeated pairs? (N=8)
            N = 8
            ids = torch.randperm(vocab, generator=g)[:N]
            pos = torch.randperm(N, generator=g)
            ids[pos[1]] = ids[pos[0]]   # always one pair
            if lab:
                ids[pos[3]] = ids[pos[2]]
        X.append(table[ids] + noise * torch.randn(len(ids), d, generator=g))
        y.append(float(lab))
    p = torch.randperm(n_samples, generator=g)
    return torch.stack(X)[p], torch.tensor(y)[p]


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def build(model_name, d, n=4, L=2, noise=0.0):
    if model_name == "quantum":
        return Classifier(d, QuantumKernelAttention(d, n, L, noise=noise))
    if model_name == "quantum-frozen":
        return Classifier(d, QuantumKernelAttention(d, n, L, train_theta=False))
    if model_name == "quantum-shared":
        return Classifier(d, QuantumKernelAttention(d, n, L, shared=True))
    if model_name == "classical":
        return Classifier(d, ClassicalAttention(d, dk=n))
    if model_name == "classical-cosine":
        return Classifier(d, ClassicalKernelAttention(d, n, "cosine"))
    if model_name == "classical-rbf":
        return Classifier(d, ClassicalKernelAttention(d, n, "rbf"))
    if model_name == "none":
        return Classifier(d, NoAttention())
    raise ValueError(model_name)


def accuracy(model, X, y):
    model.eval()
    with torch.no_grad():
        return ((model(X) > 0).double() == y).double().mean().item()


def train_one(model, Xtr, ytr, epochs=60, lr=1e-2, bs=32, seed=0):
    torch.manual_seed(seed)
    opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=lr)
    for _ in range(epochs):
        model.train()
        for i in torch.randperm(len(Xtr)).split(bs):
            loss = F.binary_cross_entropy_with_logits(model(Xtr[i]), ytr[i])
            opt.zero_grad(); loss.backward(); opt.step()
    return model


# ---------------------------------------------------------------------------
# Experiments
# ---------------------------------------------------------------------------

def validate():
    """Statevector sim vs Qiskit Estimator + parameter shift (quantum_attention.py)."""
    from quantum_attention import QuantumOverlapEstimator, QuantumOverlapFunction
    rng = np.random.default_rng(0)
    rows = []
    for n, L in [(2, 1), (2, 2), (3, 2), (4, 2), (6, 3)]:
        be = QuantumOverlapEstimator(n, L)
        a = torch.tensor(rng.uniform(-np.pi, np.pi, (8, n)))
        b = torch.tensor(rng.uniform(-np.pi, np.pi, (8, n)))
        tq = torch.tensor(rng.uniform(0, 2 * np.pi, n * L), requires_grad=True)
        tk = torch.tensor(rng.uniform(0, 2 * np.pi, n * L), requires_grad=True)
        s_est = QuantumOverlapFunction.apply(a, b, tq, tk, be)
        s_est.sum().backward()
        g_est = torch.cat([tq.grad, tk.grad]).clone()
        tq.grad = tk.grad = None
        s_sim = ((simulate(a, tq, n, L) * simulate(b, tk, n, L)).sum(1)) ** 2
        s_sim.sum().backward()
        g_sim = torch.cat([tq.grad, tk.grad])
        rows.append(dict(n=n, L=L,
                         max_score_err=float((s_est - s_sim.detach()).abs().max()),
                         max_grad_err=float((g_est - g_sim).abs().max())))
        print("validate", rows[-1], flush=True)
    write_csv("validation.csv", rows)


def exp_accuracy(seeds=5, models=("quantum", "quantum-frozen", "classical", "none"),
                 out="accuracy.csv"):
    rows = []
    d = 16
    for task in ["repeat", "match-first", "two-pairs"]:
        for s in range(seeds):
            Xtr, ytr = make_task(task, 512, seed=s)
            Xte, yte = make_task(task, 512, seed=s, split="test")
            for m in models:
                torch.manual_seed(s)
                model = build(m, d)
                t0 = time.time()
                train_one(model, Xtr, ytr, seed=s)
                rows.append(dict(task=task, model=m, seed=s, attn_params=attn_params(model),
                                 train_acc=accuracy(model, Xtr, ytr),
                                 test_acc=accuracy(model, Xte, yte),
                                 seconds=round(time.time() - t0, 1)))
                print(rows[-1], flush=True)
    write_csv(out, rows)


def exp_kernels():
    exp_accuracy(models=("classical-cosine", "classical-rbf"), out="accuracy_kernels.csv")


def exp_gradient_variance(samples=200):
    """Variance of d s / d theta_Q[0] at random init vs qubits and depth."""
    rows = []
    for n in [2, 4, 6, 8, 10]:
        for L in [1, 2, 4, 8]:
            torch.manual_seed(0)
            g = []
            for _ in range(samples):
                a = torch.rand(1, n) * 2 * math.pi
                b = torch.rand(1, n) * 2 * math.pi
                tq = (torch.rand(n * L) * 2 * math.pi).requires_grad_()
                tk = torch.rand(n * L) * 2 * math.pi
                s = ((simulate(a, tq, n, L) * simulate(b, tk, n, L)).sum()) ** 2
                (gq,) = torch.autograd.grad(s, tq)
                g.append(gq[0].item())
            rows.append(dict(n=n, L=L, grad_var=float(np.var(g))))
            print(rows[-1], flush=True)
    write_csv("gradient_variance.csv", rows)


def exp_qubits(seeds=3):
    """Test accuracy on 'two-pairs' vs qubit count (L=2)."""
    rows = []
    for n in [2, 4, 6, 8]:
        for s in range(seeds):
            Xtr, ytr = make_task("two-pairs", 512, seed=s)
            Xte, yte = make_task("two-pairs", 512, seed=s, split="test")
            for m in ["quantum", "classical"]:
                torch.manual_seed(s)
                model = build(m, 16, n=n, L=2)
                train_one(model, Xtr, ytr, seed=s)
                rows.append(dict(n=n, model=m, seed=s, attn_params=attn_params(model),
                                 test_acc=accuracy(model, Xte, yte)))
                print(rows[-1], flush=True)
    write_csv("qubits.csv", rows)


def exp_noise(seeds=3):
    """Shot noise: train with noisy scores (std eps), test with the same eps."""
    rows = []
    for eps in [0.0, 0.01, 0.03, 0.1, 0.3]:
        for s in range(seeds):
            Xtr, ytr = make_task("two-pairs", 512, seed=s)
            Xte, yte = make_task("two-pairs", 512, seed=s, split="test")
            torch.manual_seed(s)
            model = build("quantum", 16, noise=eps)
            train_one(model, Xtr, ytr, seed=s)
            model.attn.test_noise = eps
            torch.manual_seed(999)
            rows.append(dict(eps=eps, approx_shots=(round(0.25 / eps ** 2) if eps else "exact"),
                             seed=s, test_acc=accuracy(model, Xte, yte)))
            print(rows[-1], flush=True)
    write_csv("noise.csv", rows)


def write_csv(name, rows):
    with open(os.path.join(OUT, name), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)


if __name__ == "__main__":
    which = sys.argv[1:] or ["validate", "gradvar", "accuracy", "qubits", "noise"]
    fns = dict(validate=validate, gradvar=exp_gradient_variance, accuracy=exp_accuracy,
               qubits=exp_qubits, noise=exp_noise, kernels=exp_kernels)
    for w in which:
        print(f"=== {w} ===", flush=True)
        fns[w]()
