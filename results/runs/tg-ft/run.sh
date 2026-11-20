#!/bin/bash
# t_g-aware fine-tune (NLO_TG=1) and matched control (NLO_TG=0): 6k steps from nlo_fvs_ft30k.pt, LR 2e-4, seed 0.
cd "$(dirname "$0")/../../cnla"
for TG in 1 0; do
  NLO_POOL_WORKERS=4 NLO_INIT=data/nlo_fvs_ft30k.pt NLO_FVS=1 NLO_TG=$TG NLO_STEPS=6000 NLO_SEED=0 \
    NLO_CKPT_OUT=data/nlo_tgft6k_tg$TG.pt ../.venv/bin/python -u nlo_train_fvs.py > ../loops/tg-ft/train_tg$TG.log 2>&1 &
done
wait
