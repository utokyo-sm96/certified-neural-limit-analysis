#!/bin/bash
# Leave-one-bridge-out (review R4): 30k steps from scratch on one generator family each, same recipe as
# data/nlo_ckpt_final.pt (30k, both families) which serves as the control. GPU; 2 CPU pool workers.
cd "$(dirname "$0")/../../cnla"
for fam in ananai amarube; do
  NLO_FAMILY=$fam NLO_POOL_WORKERS=2 NLO_STEPS=30000 NLO_CKPT_OUT=data/nlo_lobo_${fam}30k.pt \
    ../.venv/bin/python -u nlo_train.py > ../loops/lobo/train_${fam}.log 2>&1
done
echo "all done" >> ../loops/lobo/train_amarube.log
