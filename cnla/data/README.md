# Data directory

Filled by `python scripts/download_assets.py` from the GitHub release v1.0.0:

| file | content |
|---|---|
| `nlo_fvs_ft30k.pt` | final network (U-Net with the cubic head) |
| `nlo_ckpt_100k.pt` | base network ($10^5$ steps), starting point of the fine-tune |
| `synthetic_plates.npz` | 1000 synthetic corroded plates from the generator |
| `nlo_split.npz`, `nlo_val.npz`, `nlo_test.npz` | split (904 / 32 / 64) and the validation and test plates |
| `nlo_val_ref_tg.csv`, `nlo_test_ref_tg.csv` | adaptive finite element reference brackets on $t_g$ |
| `generator_calibration.csv` | parameters of the corrosion generator |

`--extra` adds the networks of the ablation and the leave-one-bridge-out study. Training scripts write their caches
here as well.
