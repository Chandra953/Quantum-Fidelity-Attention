#!/usr/bin/env bash
# Reproduce every test, experiment, figure, the README results and the paper PDF.
# Run from the repository root.
set -euo pipefail
mkdir -p logs
[ -d data/sst2 ] || bash scripts/download_sst2.sh

echo "== tests"
python -m pytest -q tests/ | tee logs/tests.txt

echo "== experiments"
python src/experiments.py validate gradvar accuracy kernels qubits noise
python src/exp2.py check synthetic sst2
python src/exp2.py timing          # run last, on an otherwise idle machine

echo "== figures, README, paper"
(cd paper && python make_figures.py)
python scripts/summarize.py logs/tests.txt
(cd paper && latexmk -pdf -interaction=nonstopmode paper.tex >/dev/null)
echo "Done: README.md updated, paper/paper.pdf built"
