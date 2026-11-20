#!/usr/bin/env python3
"""Fixed train / validation / test split of the 1,000 synthetic plates for the neural limit operator.

    validation  32 plates  (the loop metric; FE references in nlo_val_ref.csv)
    test        64 plates  (evaluated once, at the end; FE references in nlo_test_ref.csv)
    train      904 plates  (label-free training: no reference solution is ever used)

The real scans and the tensile tests are not part of any split; they are the held-out experimental check.
Output: data/nlo_split.npz (index arrays), data/nlo_val.npz, data/nlo_test.npz (plates with sigma_y).
"""
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
SIGMA_Y = {"ananai": 281.6, "amarube": 269.4}          # MPa, coupon means (material_properties.csv)
SEED = 2026


def main():
    d = np.load(os.path.join(DATA, "synthetic_plates.npz"), allow_pickle=True)
    n = len(d["family"])
    perm = np.random.default_rng(SEED).permutation(n)
    idx = dict(idx_val=np.sort(perm[:32]), idx_test=np.sort(perm[32:96]), idx_train=np.sort(perm[96:]))
    np.savez(os.path.join(DATA, "nlo_split.npz"), **idx)
    x, y, t, fam = d["x_mm"], d["y_mm"], d["thickness_mm"], d["family"]
    for name in ("val", "test"):
        ii = idx[f"idx_{name}"]
        np.savez(os.path.join(DATA, f"nlo_{name}.npz"),
                 x_mm=np.array([x[i].astype(float) for i in ii], dtype=object),
                 y_mm=np.array([y[i].astype(float) for i in ii], dtype=object),
                 thickness_mm=np.array([t[i].astype(float) for i in ii], dtype=object),
                 family=fam[ii], sigma_y_MPa=np.array([SIGMA_Y[f] for f in fam[ii]]),
                 name=np.array([f"syn{i:04d}" for i in ii]), index=ii)
        print(f"{name}: {len(ii)} plates, {np.sum(fam[ii] == 'ananai')} Ananai / {np.sum(fam[ii] == 'amarube')} Amarube")
    print(f"train: {len(idx['idx_train'])} plates")


if __name__ == "__main__":
    main()
