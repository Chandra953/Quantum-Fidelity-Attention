"""
Second experiment suite (AI-focused) for the quantum-kernel attention paper.

Adds:
  * dot-product attention with a wider query/key dimension (dk = 16)
  * cosine-product attention in LINEAR time via its explicit feature map
        s_ij = prod_k cos^2((phi_ik - phi_jk)/2) = (psi_i . psi_j)^2 = Phi_i . Phi_j,
        psi(x) = ⊗_k [cos(phi_k/2), sin(phi_k/2)]  (2^n dims),  Phi = psi ⊗ psi (4^n dims)
  * linear attention with elu+1 features (Katharopoulos et al., 2020)
  * SST-2 sentence classification (real text)
  * wall-clock scaling vs sequence length

Every attention module takes (x, mask) with mask[b, j] = True for real tokens.
Writes CSVs to ./results/.
"""
import csv
import math
import os
import sys
import time
from collections import Counter

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from experiments import simulate, make_task

torch.set_default_dtype(torch.float64)
OUT = "results"
os.makedirs(OUT, exist_ok=True)
NEG = -1e9


# ---------------------------------------------------------------------------
# Attention variants
# ---------------------------------------------------------------------------

class _Base(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.value, self.out = nn.Linear(d, d), nn.Linear(d, d)

    def softmax_attend(self, scores, x, mask):
        if mask is not None:
            scores = scores.masked_fill(~mask[:, None, :], NEG)
        return self.out(F.softmax(scores, -1) @ self.value(x))

    def linear_attend(self, phi_q, phi_k, x, mask):
        """y_i = sum_j (phi_q_i . phi_k_j) v_j / sum_j (phi_q_i . phi_k_j); O(N D d)."""
        if mask is not None:
            phi_k = phi_k * mask[..., None]
        v = self.value(x)
        kv = phi_k.transpose(1, 2) @ v                    # (B, D, d)
        z = phi_k.sum(1)                                  # (B, D)
        num = phi_q @ kv                                  # (B, N, d)
        den = (phi_q * z[:, None, :]).sum(-1, keepdim=True) + 1e-6
        return self.out(num / den)


class DotAttention(_Base):
    def __init__(self, d, dk=4):
        super().__init__(d)
        self.q, self.k, self.dk = nn.Linear(d, dk), nn.Linear(d, dk), dk

    def forward(self, x, mask=None):
        return self.softmax_attend(self.q(x) @ self.k(x).transpose(1, 2) / math.sqrt(self.dk), x, mask)


def product_state(phi):
    """psi(x) = ⊗_k [cos(phi_k/2), sin(phi_k/2)]; phi: (..., n) -> (..., 2^n)."""
    c, s = torch.cos(phi / 2), torch.sin(phi / 2)
    psi = torch.stack([c[..., 0], s[..., 0]], -1)
    for k in range(1, phi.shape[-1]):
        nxt = torch.stack([c[..., k], s[..., k]], -1)
        psi = (psi[..., :, None] * nxt[..., None, :]).flatten(-2)
    return psi


class CosineAttention(_Base):
    """Fidelity of the angle-encoded product state, computed classically."""
    def __init__(self, d, n=4, linear=False):
        super().__init__(d)
        self.to_angles, self.linear = nn.Linear(d, n), linear
        if not linear:
            self.log_scale = nn.Parameter(torch.tensor(math.log(5.0)))

    def forward(self, x, mask=None):
        phi = math.pi * torch.tanh(self.to_angles(x))
        if self.linear:
            psi = product_state(phi)                                  # (B, N, 2^n)
            Phi = (psi[..., :, None] * psi[..., None, :]).flatten(-2) # (B, N, 4^n)
            return self.linear_attend(Phi, Phi, x, mask)
        # prod_k cos^2((a_k-b_k)/2) == (psi(a).psi(b))^2, computed as one matrix product
        psi = product_state(phi)
        s = (psi @ psi.transpose(1, 2)) ** 2
        return self.softmax_attend(s * self.log_scale.exp(), x, mask)


class QuantumAttention(_Base):
    """Fidelity with trainable query/key ansatzes (exact statevector simulation)."""
    def __init__(self, d, n=4, L=2, train_theta=True):
        super().__init__(d)
        self.n, self.L = n, L
        self.to_angles = nn.Linear(d, n)
        self.theta_q = nn.Parameter(0.1 * torch.randn(n * L), requires_grad=train_theta)
        self.theta_k = nn.Parameter(0.1 * torch.randn(n * L), requires_grad=train_theta)
        self.log_scale = nn.Parameter(torch.tensor(math.log(5.0)))

    def forward(self, x, mask=None):
        B, N, _ = x.shape
        ang = math.pi * torch.tanh(self.to_angles(x)).reshape(B * N, self.n)
        q = simulate(ang, self.theta_q, self.n, self.L).reshape(B, N, -1)
        k = simulate(ang, self.theta_k, self.n, self.L).reshape(B, N, -1)
        s = (q @ k.transpose(1, 2)) ** 2
        return self.softmax_attend(s * self.log_scale.exp(), x, mask)


class EluLinearAttention(_Base):
    def __init__(self, d, dk=16):
        super().__init__(d)
        self.q, self.k = nn.Linear(d, dk), nn.Linear(d, dk)

    def forward(self, x, mask=None):
        return self.linear_attend(F.elu(self.q(x)) + 1, F.elu(self.k(x)) + 1, x, mask)


class NoAttention(nn.Module):
    def forward(self, x, mask=None):
        return torch.zeros_like(x)


def make_attn(name, d):
    return {
        "quantum": lambda: QuantumAttention(d),
        "quantum-frozen": lambda: QuantumAttention(d, train_theta=False),
        "cosine": lambda: CosineAttention(d),
        "cosine-linear": lambda: CosineAttention(d, linear=True),
        "dot4": lambda: DotAttention(d, 4),
        "dot16": lambda: DotAttention(d, 16),
        "elu-linear": lambda: EluLinearAttention(d, 16),
        "none": lambda: NoAttention(),
    }[name]()


def n_params(m):
    return sum(p.numel() for p in m.parameters() if p.requires_grad)


# ---------------------------------------------------------------------------
# Classifiers
# ---------------------------------------------------------------------------

class SeqClassifier(nn.Module):
    """Same head as experiments.py: positions, residual attention, LN, max-pool, MLP."""
    def __init__(self, d, attn, max_len=8):
        super().__init__()
        self.attn, self.ln = attn, nn.LayerNorm(d)
        self.pos = nn.Parameter(0.1 * torch.randn(max_len, d))
        self.head = nn.Sequential(nn.Linear(d, 32), nn.GELU(), nn.Linear(32, 1))

    def forward(self, x, mask=None):
        x = x + self.pos[: x.shape[1]]
        h = self.ln(x + self.attn(x, mask))
        if mask is not None:
            h = h.masked_fill(~mask[..., None], NEG)
        return self.head(h.max(1).values).squeeze(-1)


class TextClassifier(nn.Module):
    def __init__(self, vocab, d, attn, max_len):
        super().__init__()
        self.emb = nn.Embedding(vocab, d, padding_idx=0)
        self.body = SeqClassifier(d, attn, max_len)
        self.drop = nn.Dropout(0.3)

    def forward(self, ids):
        mask = ids != 0
        return self.body(self.drop(self.emb(ids)), mask)


# ---------------------------------------------------------------------------
# Synthetic tasks: new models only (5 seeds, same protocol as experiments.py)
# ---------------------------------------------------------------------------

def run_synthetic(models=("dot16", "cosine-linear", "elu-linear"), seeds=5):
    rows = []
    for task in ["repeat", "match-first", "two-pairs"]:
        for s in range(seeds):
            Xtr, ytr = make_task(task, 512, seed=s)
            Xte, yte = make_task(task, 512, seed=s, split="test")
            for m in models:
                torch.manual_seed(s)
                model = SeqClassifier(16, make_attn(m, 16))
                opt = torch.optim.Adam(model.parameters(), lr=1e-2)
                torch.manual_seed(s)
                for _ in range(60):
                    model.train()
                    for i in torch.randperm(len(Xtr)).split(32):
                        loss = F.binary_cross_entropy_with_logits(model(Xtr[i]), ytr[i])
                        opt.zero_grad(); loss.backward(); opt.step()
                model.eval()
                with torch.no_grad():
                    acc = ((model(Xte) > 0).double() == yte).double().mean().item()
                rows.append(dict(task=task, model=m, seed=s, attn_params=n_params(model.attn),
                                 test_acc=acc))
                print(rows[-1], flush=True)
    write_csv("accuracy_v2.csv", rows)


# ---------------------------------------------------------------------------
# SST-2
# ---------------------------------------------------------------------------

def load_sst2(path="data/sst2", max_len=48):
    def read(f):
        xs, ys = [], []
        for line in open(os.path.join(path, f), encoding="utf-8"):
            line = line.rstrip("\n")
            if not line:
                continue
            text, lab = line.rsplit("\t", 1)
            xs.append(text.lower().split()); ys.append(float(lab))
        return xs, ys
    tr, dv, te = read("train.tsv"), read("dev.tsv"), read("test.tsv")
    cnt = Counter(w for s in tr[0] for w in s)
    vocab = {w: i + 2 for i, (w, c) in enumerate(cnt.most_common()) if c >= 2}   # 0 pad, 1 unk

    def enc(split):
        ids = torch.zeros(len(split[0]), max_len, dtype=torch.long)
        for i, s in enumerate(split[0]):
            t = [vocab.get(w, 1) for w in s][:max_len]
            ids[i, :len(t)] = torch.tensor(t)
        return ids, torch.tensor(split[1])
    return enc(tr), enc(dv), enc(te), len(vocab) + 2


def run_sst2(models=("quantum", "quantum-frozen", "cosine", "cosine-linear", "dot4", "dot16",
                     "elu-linear", "none"), seeds=5, d=64, epochs=10, max_len=48):
    torch.set_default_dtype(torch.float32)
    (Xtr, ytr), (Xdv, ydv), (Xte, yte), V = load_sst2(max_len=max_len)
    print(f"SST-2: train {len(Xtr)}, dev {len(Xdv)}, test {len(Xte)}, vocab {V}", flush=True)
    rows = []
    for s in range(seeds):
        for m in models:
            torch.manual_seed(s)
            model = TextClassifier(V, d, make_attn(m, d), max_len).float()
            opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
            best_dev, best_test, t0 = -1, None, time.time()
            for ep in range(epochs):
                model.train()
                g = torch.Generator().manual_seed(100 * s + ep)
                for i in torch.randperm(len(Xtr), generator=g).split(64):
                    loss = F.binary_cross_entropy_with_logits(model(Xtr[i]), ytr[i].float())
                    opt.zero_grad(); loss.backward(); opt.step()
                model.eval()
                with torch.no_grad():
                    dev = ((model(Xdv) > 0).float() == ydv).float().mean().item()
                    test = ((model(Xte) > 0).float() == yte).float().mean().item()
                if dev > best_dev:                     # model selection on dev only
                    best_dev, best_test = dev, test
            rows.append(dict(model=m, seed=s, attn_params=n_params(model.body.attn),
                             dev_acc=best_dev, test_acc=best_test,
                             seconds=round(time.time() - t0, 1)))
            print(rows[-1], flush=True)
    write_csv("sst2.csv", rows)
    torch.set_default_dtype(torch.float64)


# ---------------------------------------------------------------------------
# Wall-clock scaling
# ---------------------------------------------------------------------------

def run_timing(lengths=(64, 256, 1024, 2048, 4096), d=64, reps=5):
    torch.set_default_dtype(torch.float32)
    torch.set_num_threads(1)
    rows = []
    for m in ["dot16", "cosine", "quantum", "cosine-linear", "elu-linear"]:
        torch.manual_seed(0)
        attn = make_attn(m, d).float().eval()
        for N in lengths:
            x = torch.randn(1, N, d)
            with torch.no_grad():
                attn(x)                                          # warm-up
                ts = []
                for _ in range(reps):
                    t0 = time.perf_counter(); attn(x); ts.append(time.perf_counter() - t0)
            rows.append(dict(model=m, N=N, ms=round(1000 * float(np.median(ts)), 3)))
            print(rows[-1], flush=True)
    write_csv("timing.csv", rows)
    torch.set_default_dtype(torch.float64)


def check_feature_map():
    """Linear-time feature map reproduces the pairwise cosine kernel exactly."""
    torch.manual_seed(0)
    phi = (torch.rand(5, 4) * 2 - 1) * math.pi
    s = torch.cos((phi[:, None] - phi[None]) / 2).pow(2).prod(-1)
    psi = product_state(phi)
    Phi = (psi[:, :, None] * psi[:, None, :]).flatten(1)
    err = (Phi @ Phi.T - s).abs().max().item()
    print("feature-map max error", err)
    assert err < 1e-12
    return err


def write_csv(name, rows):
    with open(os.path.join(OUT, name), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)


if __name__ == "__main__":
    which = sys.argv[1:] or ["check", "timing", "synthetic", "sst2"]
    fns = dict(check=check_feature_map, timing=run_timing, synthetic=run_synthetic, sst2=run_sst2)
    for w in which:
        print(f"=== {w} ===", flush=True)
        fns[w]()
