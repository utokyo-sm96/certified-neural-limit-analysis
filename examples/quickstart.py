#!/usr/bin/env python3
"""Certified neural limit analysis of one synthetic plate: the final network and the certified refinement give a
bracket on the plastic collapse load, and the independent verifier recomputes both bounds for the thickness model t_g.
Usage: python examples/quickstart.py [plate index in the validation set, default 0]"""
import csv
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "cnla"))
import nlo_refine as R  # noqa: E402
import nlo_train_fvs  # noqa: E402
from nlo_refine_eval import SQ3, tri_area  # noqa: E402
from verify_v2 import verify_v2  # noqa: E402

DATA = os.path.join(ROOT, "cnla", "data")


def main():
    i = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    R.THICK_MODEL = "tg"                                     # certify every bound for the common thickness model
    d = np.load(os.path.join(DATA, "nlo_val.npz"), allow_pickle=True)
    name = str(d["name"][i])
    xs, ys, t = d["x_mm"][i], d["y_mm"][i], np.asarray(d["thickness_mm"][i], float)
    sy = float(d["sigma_y_MPa"][i])
    plate = dict(xs=xs, ys=ys, t=t, sigma_y=sy)
    print(f"plate {name}: {xs[-1]:.0f} x {ys[-1]:.0f} mm, thickness {t.min():.2f}-{t.max():.2f} mm, "
          f"sigma_y = {sy:.1f} MPa")
    model = nlo_train_fvs.load(os.path.join(DATA, "nlo_fvs_ft30k.pt"))
    R.bounds(model, plate)                                    # warm-up: plate-independent factorizations, GPU kernels
    t0 = time.time()
    out = R.bounds(model, plate)
    dt = time.time() - t0
    # independent verification (verify_v2 recomputes both bounds; it never scales a field up)
    u, l = out["ub_res"], out["lb_res"]
    S = np.zeros((len(u["tri"]), 3, 3))
    S[:, :, 0] = 1.0
    ub, _, eu, _ = verify_v2(dict(nodes=np.asarray(u["nodes"], float), tri=np.asarray(u["tri"], int),
                                  thick=np.asarray(u["thick"], float), v=np.asarray(u["v"], float), ub=float(u["ub"]),
                                  stress=S, lb=0.0, status_ub="Solved", status_lb="Solved"), xs, ys, t, sy)
    nl, tl, thl = np.asarray(l["nodes"], float), np.asarray(l["tri"], int), np.asarray(l["thick"], float)
    vl = np.stack([nl[:, 0] / xs[-1], np.zeros(len(nl))], 1)
    ubd = float(np.sum(sy * (2 / SQ3) * (1 / xs[-1]) * tri_area(nl, tl) * thl[tl].mean(1)))
    _, lb, el, _ = verify_v2(dict(nodes=nl, tri=tl, thick=thl, v=vl, ub=ubd, stress=np.asarray(l["stress"], float),
                                  lb=float(l["lb"]), status_ub="Solved", status_lb="Solved"), xs, ys, t, sy)
    print(f"certified bracket: [{lb / 1e3:.2f}, {ub / 1e3:.2f}] kN, relative gap {100 * (ub - lb) / ub:.2f} %, "
          f"solver time {dt:.1f} s, verifier errors: {eu + el or 'none'}")
    refs = {r["name"]: r for r in csv.DictReader(open(os.path.join(DATA, "nlo_val_ref_tg.csv")))}
    if name in refs:
        lo, hi = float(refs[name]["LB_ref_N"]), float(refs[name]["UB_ref_N"])
        ok = lb <= hi * (1 + 1e-6) and ub >= lo * (1 - 1e-6)
        print(f"FE reference (adaptive, level 4): [{lo / 1e3:.2f}, {hi / 1e3:.2f}] kN, consistent: {ok}")


if __name__ == "__main__":
    main()
