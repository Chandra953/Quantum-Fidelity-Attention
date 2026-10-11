# From Quantum Fidelity to Linear-Time Attention

**An Exact Classical Reduction of Quantum Self-Attention**

Chandra Sekhar Singuru · Independent Researcher · [ORCID 0009-0004-0219-1960](https://orcid.org/0009-0004-0219-1960)

📄 Paper (IEEE conference format): [`paper/Quantum_Fidelity_Linear_Attention.pdf`](paper/Quantum_Fidelity_Linear_Attention.pdf)

---

## Summary

Quantum self-attention models replace the Transformer's dot-product score with the fidelity between quantum states that encode two tokens. This project asks where their reported accuracy gains come from, and whether they need a quantum computer.

- **Exact classical reduction.** For angle-encoded states, the quantum fidelity score equals a classical **cosine-product (CP) kernel**, and any fidelity-based attention is an inner product of explicit `4^n`-dimensional features.
- **CPLA: a linear-time attention layer.** The reduction yields cosine-product linear attention, a classical layer whose cost grows linearly with sequence length.
- **Controlled experiments.** Nine attention variants are compared, including a quantum layer built on the Qiskit Estimator primitive with exact parameter-shift gradients.

## Key results

| Finding | Evidence |
|---|---|
| Quantum attention beats dot-product attention on token-matching tasks by 0.13–0.31 accuracy | 3 tasks × 5 seeds (`results/accuracy*.csv`) |
| A classical CP kernel matches the quantum layer within seed variance | paired per-seed comparison |
| CPLA matches or exceeds the quantum layer on 2 of 3 tasks, ~22× faster than softmax attention at 4,096 tokens | `results/timing.csv` |
| Wider dot-product and standard linear attention do not close the gap | `results/accuracy_v2.csv` |
| On SST-2 sentiment, dot-product attention is best; CP-kernel layers trail by 1.1–2.4 points | `results/sst2.csv` |
| Gradient variance of the quantum layer falls ~10,000× from 2 to 10 qubits | `results/gradient_variance.csv` |

## Repository layout

```
src/
  quantum_attention.py   Qiskit reference layer: Estimator V2, parameter-shift autograd, checks
  experiments.py         exact statevector simulator, synthetic tasks, gradient/qubit/noise studies
  exp2.py                CPLA, wider baselines, SST-2, timing
results/                 raw CSVs behind every number in the paper
paper/                   IEEEtran LaTeX source, BibTeX, figure script, figures, compiled PDF
scripts/                 data download and full reproduction
```

## Reproduce

```bash
pip install -r requirements.txt
bash scripts/download_sst2.sh        # SST-2 sentence split (6,920 / 872 / 1,821)
bash scripts/reproduce.sh            # all experiments, figures and the PDF (~1–2 h on a laptop CPU)
```

Individual steps:

```bash
python src/quantum_attention.py                                          # Qiskit layer demo + checks
python src/experiments.py validate gradvar accuracy kernels qubits noise
python src/exp2.py check timing synthetic sst2
cd paper && python make_figures.py && latexmk -pdf paper.tex
```

Timing depends on hardware; run it on an otherwise idle machine. The simulator in `experiments.py` is validated against the Qiskit implementation to within 1e-15 in scores and gradients.

## Citation

If you use this work, please cite it (see also [`CITATION.cff`](CITATION.cff)):

```bibtex
@misc{singuru2026fidelity,
  author = {Singuru, Chandra Sekhar},
  title  = {From Quantum Fidelity to Linear-Time Attention: An Exact Classical Reduction of Quantum Self-Attention},
  year   = {2026},
  howpublished = {\url{https://github.com/Chandra953/Quantum-Fidelity-Attention}}
}
```

## Disclosure

Generative AI tools assisted with code development, experiment design and drafting. The author reviewed, verified and takes responsibility for all content.

## License

Code is released under the MIT License (see [`LICENSE`](LICENSE)). The paper text and figures are © 2026 Chandra Sekhar Singuru.
