#!/usr/bin/env bash
# Download the SST-2 sentence-level split (Socher et al., 2013) used in the paper.
set -euo pipefail
BASE=https://raw.githubusercontent.com/clairett/pytorch-sentiment-classification/master/data/SST2
mkdir -p data/sst2
for f in train dev test; do
  curl -fsSL "$BASE/$f.tsv" -o "data/sst2/$f.tsv"
done
wc -l data/sst2/*.tsv
