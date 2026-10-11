"""Generate all figures for the IEEE paper from ../results/*.csv."""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "results")
FIG = os.path.join(HERE, "figures")
ASSETS = os.path.join(HERE, "..", "assets")
os.makedirs(FIG, exist_ok=True)
os.makedirs(ASSETS, exist_ok=True)


def save(fig, name):
    """Vector PDF for the paper, PNG for the README."""
    fig.savefig(os.path.join(FIG, name + ".pdf"), bbox_inches="tight")
    fig.savefig(os.path.join(ASSETS, name + ".png"), bbox_inches="tight", dpi=200)
    plt.close(fig)

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Times", "Times New Roman", "DejaVu Serif"],
    "font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8, "legend.fontsize": 7,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "axes.spines.top": False,
    "axes.spines.right": False, "pdf.fonttype": 42,
})
BLUE, ORANGE, GREEN, GREY = "#1f5fbf", "#d4622a", "#2e8b57", "#9a9a9a"
COL, DCOL = 3.5, 7.16

# model key -> (label, colour); one source of truth for every figure
MODELS = {
    "quantum": ("Quantum fidelity (ours)", BLUE),
    "quantum-frozen": ("Quantum, frozen ansatz", GREY),
    "classical-cosine": ("CP kernel, softmax", ORANGE),
    "cosine-linear": ("CP kernel, linear (CPLA)", GREEN),
    "classical-rbf": ("RBF kernel", GREY),
    "classical": ("Dot product, $d_k{=}4$", GREY),
    "dot16": ("Dot product, $d_k{=}16$", GREY),
    "elu-linear": ("Linear attn., elu+1", GREY),
    "none": ("No attention", GREY),
}


def load_synthetic():
    a = pd.concat([pd.read_csv(os.path.join(RES, f)) for f in
                   ["accuracy.csv", "accuracy_kernels.csv", "accuracy_v2.csv"]])
    return a[["task", "model", "seed", "test_acc", "attn_params"]]


def fig_circuit():
    sys.path.insert(0, os.path.join(HERE, "..", "src"))
    from quantum_attention import build_overlap_circuit
    qc, _ = build_overlap_circuit(4, 1)
    qc.measure_all()
    f = qc.draw("mpl", fold=-1, scale=0.55, style={"name": "bw"})
    save(f, "circuit")


def fig_accuracy():
    d = load_synthetic()
    order = list(MODELS)
    tasks = ["repeat", "match-first", "two-pairs"]
    fig, axes = plt.subplots(1, 3, figsize=(DCOL, 2.45), sharey=True)
    for ax, t in zip(axes, tasks):
        for i, m in enumerate(order):
            v = d[(d.task == t) & (d.model == m)].test_acc.values
            lab, c = MODELS[m]
            y = len(order) - 1 - i
            ax.scatter(v, np.full_like(v, y - 0.25), s=5, color=c, alpha=0.45, lw=0)
            ax.errorbar(v.mean(), y, xerr=v.std(ddof=1), fmt="o", color=c, ms=3.5, lw=1.1)
            if c != GREY:
                ax.text(min(v.mean() + v.std(ddof=1), 1.0) + 0.012, y, f"{v.mean():.2f}",
                        va="center", fontsize=6.5, color=c)
        ax.set_xlim(0.5, 1.08)
        ax.set_xticks([0.5, 0.75, 1.0])
        ax.set_title(t)
        ax.grid(axis="x", color="#e5e5e5", lw=0.6)
        ax.set_axisbelow(True)
    axes[0].set_yticks(range(len(order)))
    axes[0].set_yticklabels([MODELS[m][0] for m in order][::-1])
    axes[1].set_xlabel("Test accuracy (0.5 = chance)")
    fig.tight_layout(w_pad=0.8)
    save(fig, "accuracy")


def fig_timing():
    t = pd.read_csv(os.path.join(RES, "timing.csv"))
    style = {"dot16": ("Softmax dot product", GREY, "s--"),
             "cosine": ("CP kernel, softmax", ORANGE, "o-"),
             "cosine-linear": ("CPLA (linear)", GREEN, "o-"),
             "elu-linear": ("Linear attn., elu+1", GREY, "^:")}
    fig, ax = plt.subplots(figsize=(COL, 2.45))
    for m, (lab, c, ls) in style.items():
        s = t[t.model == m].sort_values("N")
        ax.loglog(s.N, s.ms, ls, color=c, ms=3, lw=1.2, label=lab)
    ns = np.array(sorted(t.N.unique()), dtype=float)
    ref = t[(t.model == "dot16")].sort_values("N").ms.values
    ax.loglog(ns, ref[-1] * (ns / ns[-1]) ** 2, color="#cccccc", lw=0.8, zorder=0)
    ax.text(ns[-2] * 0.62, ref[-1] * (ns[-2] / ns[-1]) ** 2 * 1.9, r"$\propto N^2$", fontsize=6.5,
            color="#888", ha="right")
    ax.set_xlabel("Sequence length $N$")
    ax.set_ylabel("Forward time (ms, 1 CPU thread)")
    ax.legend(frameon=False, loc="lower left", bbox_to_anchor=(0.0, 1.0), ncol=2, columnspacing=1.0)
    ax.grid(color="#e5e5e5", lw=0.6, which="major")
    fig.tight_layout()
    save(fig, "timing")


def fig_gradvar():
    g = pd.read_csv(os.path.join(RES, "gradient_variance.csv"))
    fig, ax = plt.subplots(figsize=(COL, 2.1))
    Ls = sorted(g.L.unique())
    for i, L in enumerate(Ls):
        s = g[g.L == L].sort_values("n")
        ax.semilogy(s.n, s.grad_var, "o-", color=BLUE, alpha=0.3 + 0.7 * i / (len(Ls) - 1),
                    ms=3, lw=1.2, label=f"L = {L}")
    r = g[(g.n == 4) & (g.L == 2)].iloc[0]
    ax.plot(r.n, r.grad_var, "o", mfc="none", mec="k", ms=8, mew=0.8)
    ax.annotate("setting used\nin Sec. V-A", (r.n, r.grad_var), (5.0, 1.5e-2),
                fontsize=6.5, arrowprops=dict(arrowstyle="-", lw=0.6))
    ax.set_xlabel("Qubits $n$")
    ax.set_ylabel(r"Var$[\partial s/\partial\theta]$")
    ax.set_xticks([2, 4, 6, 8, 10])
    ax.legend(title="Ansatz layers", frameon=False, title_fontsize=7)
    ax.grid(color="#e5e5e5", lw=0.6, which="major")
    fig.tight_layout()
    save(fig, "gradvar")


def fig_noise():
    n = pd.read_csv(os.path.join(RES, "noise.csv"))
    q = pd.read_csv(os.path.join(RES, "qubits.csv"))
    base = q[(q.n == 4) & (q.model == "classical")].test_acc.mean()
    eps = sorted(n.eps.unique())
    fig, ax = plt.subplots(figsize=(COL, 2.0))
    xs = np.arange(len(eps))
    for x, e in zip(xs, eps):
        v = n[n.eps == e].test_acc.values
        ax.scatter(np.full_like(v, x + 0.15), v, s=6, color=BLUE, alpha=0.45, lw=0)
        ax.errorbar(x, v.mean(), yerr=v.std(ddof=1), fmt="o", color=BLUE, ms=4, lw=1.2)
        ax.text(x - 0.12, v.mean(), f"{v.mean():.2f}", ha="right", va="center", fontsize=6.5)
    ax.axhline(base, ls="--", color=GREY, lw=0.9)
    ax.text(len(eps) - 0.6, base - 0.035, f"dot product, noise-free ({base:.2f})",
            ha="right", fontsize=6.5, color="#555")
    shots = ["exact"] + [f"~{round(0.25 / e ** 2):,}" for e in eps[1:]]
    ax.set_xticks(xs)
    ax.set_xticklabels([f"{'0' if e == 0 else e}\n{s}" for e, s in zip(eps, shots)])
    ax.set_xlabel(r"Score noise $\varepsilon$ / approx. shots per score")
    ax.set_ylabel("Test accuracy")
    ax.set_ylim(0.6, 1.0)
    ax.set_xlim(-0.75, len(eps) - 0.5)
    ax.grid(axis="y", color="#e5e5e5", lw=0.6)
    fig.tight_layout()
    save(fig, "noise")


if __name__ == "__main__":
    which = sys.argv[1:] or ["accuracy", "timing", "gradvar", "noise", "circuit"]
    for w in which:
        globals()[f"fig_{w}"]()
    print("figures written to", FIG)
