#!/bin/bash
# One-shot held-out evaluation under verifier v2 (final method = main after the merge of loop/tg-method-1112).
# 1) validation histories again (network events now stamped when the t_g data are ready); 2) held-out histories
# (64 synthetic test plates, 28 real scans), one process, idle machine; 3) FE on t_g for the held-out plates (pool).
cd "$(dirname "$0")/../../.."
O=results/runs/heldout-v2
git rev-parse HEAD > $O/commit.txt
for S in val test real; do
  VH_TG=1 VH_SET=$S .venv/bin/python -u paper/figures/val_histories.py > $O/histories_$S.log 2>&1 || { echo "histories $S failed" >> $O/status.txt; exit 1; }
  echo "histories $S done" >> $O/status.txt
done
cd cnla && ../.venv/bin/python -u fe_tg_heldout.py --out ../$O --workers 6 > ../$O/fe.log 2>&1 && echo "fe done" >> ../$O/status.txt || echo "fe failed" >> ../$O/status.txt
