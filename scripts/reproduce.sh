#!/usr/bin/env bash
# Reproduce every experiment, figure and the paper PDF. Run from the repository root.
set -euo pipefail
[ -d data/sst2 ] || bash scripts/download_sst2.sh
python src/experiments.py validate gradvar accuracy kernels qubits noise
python src/exp2.py check timing synthetic sst2
cd paper
python make_figures.py
latexmk -pdf -interaction=nonstopmode paper.tex
echo "Done: paper/paper.pdf"
