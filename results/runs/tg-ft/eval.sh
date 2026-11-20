#!/bin/bash
# After the two fine-tunes: network-only evals (parallel), then full-method evals (sequential, idle machine), then
# the LB ceiling probe. Method = nlo_refine with THICK_MODEL "tg", best linear minorant, exact final LB scaling.
cd "$(dirname "$0")/../.."
L=loops/tg-ft
until { grep -q '^done' $L/train_tg1.log && grep -q '^done' $L/train_tg0.log; } || grep -q Traceback $L/train_tg1.log $L/train_tg0.log; do sleep 15; done
grep -q Traceback $L/train_tg1.log $L/train_tg0.log && { echo "training crashed" > $L/eval.status; exit 1; }
declare -A CK=([base]=cnla/data/nlo_fvs_ft30k.pt [tg1]=cnla/data/nlo_tgft6k_tg1.pt [tg0]=cnla/data/nlo_tgft6k_tg0.pt)
for k in base tg1 tg0; do
  .venv/bin/python -u loops/scratch/tg_ckpt_eval.py ${CK[$k]} $L/net_$k.csv 32 --net-only > $L/net_$k.log 2>&1 &
done
wait
echo "net done" > $L/eval.status
for k in base tg1 tg0; do
  .venv/bin/python -u loops/scratch/tg_ckpt_eval.py ${CK[$k]} $L/full_$k.csv 32 > $L/full_$k.log 2>&1
  echo "full $k done" >> $L/eval.status
done
.venv/bin/python -u loops/scratch/tg_floor.py 0 1 2 5 6 23 > loops/scratch/tg_floor.log 2>&1
echo "all done" >> $L/eval.status
