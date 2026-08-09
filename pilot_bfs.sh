#!/usr/bin/env bash
# Pilot: does BFS subgraph sampling improve ground-truth cluster recovery?
#
# Three arms at equal coverage (5 epochs) plus a matched-compute control that
# gives uniform sampling the same number of updates as bfs_frac=1.0, so that
# "BFS helped" is separated from "more gradient steps helped".
set -u
cd "$(dirname "$0")"
PY=./.venv/bin/python
COMMON="--k 729 --d 64 --lr 0.1 --likelihood bernoulli --seed 0"
EPOCHS=5
MATCHED_UPDATES=$((5 * 226))

for bf in 0.0 0.5 1.0; do
  echo "########## ARM bfs_frac=${bf} epochs=${EPOCHS} ##########"
  $PY train_lv_vsbm.py $COMMON --epochs $EPOCHS --bfs-frac "$bf" \
    --out-prefix "pilot_bfs${bf}" 2>&1 | grep -aE "(^loss=|^Saved|edges_seen)"
done

echo "########## ARM bfs_frac=0.0 matched-compute (${MATCHED_UPDATES} updates) ##########"
$PY train_lv_vsbm.py $COMMON --epochs 9999 --target-updates $MATCHED_UPDATES --bfs-frac 0.0 \
  --out-prefix "pilot_matched" 2>&1 | grep -aE "(^loss=|^Saved|edges_seen)"

echo "########## GROUND TRUTH ##########"
$PY evaluate_clustering.py --pred \
  pilot_bfs0.0_assignment_dict.npy \
  pilot_bfs0.5_assignment_dict.npy \
  pilot_bfs1.0_assignment_dict.npy \
  pilot_matched_assignment_dict.npy
echo "PILOT_DONE"
