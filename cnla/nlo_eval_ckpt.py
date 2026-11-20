#!/usr/bin/env python3
"""Evaluate any trained NLO checkpoint end to end (not a loop evaluator; for learning curves and final models).

For each plate of a split: certified bracket of the network alone, and of the hybrid method
(nlo_refine.bounds: mirror combination + local cone-program corrections), both certified by the frozen
step-1 verifier (nlo_refine_eval.certify_ub / certify_lb) and compared with the step-1 FE references.

Usage: python nlo_eval_ckpt.py --ckpt data/nlo_ckpt_long.pt --split val [--out ../loops/<tag>/eval_val.csv]
The test split is held-out data: evaluate it once per final model only.
"""
import argparse
import csv
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--split", default="val", choices=["val", "test"])
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    import nlo_refine
    import nlo_train
    from nlo_refine_eval import certify_lb, certify_ub
    model = nlo_train.load(a.ckpt)
    d = np.load(os.path.join(HERE, "data", f"nlo_{a.split}.npz"), allow_pickle=True)
    refs = {r["name"]: r for r in csv.DictReader(open(os.path.join(HERE, "data", f"nlo_{a.split}_ref.csv")))}
    plates = [dict(xs=d["x_mm"][i], ys=d["y_mm"][i], t=d["thickness_mm"][i], sigma_y=float(d["sigma_y_MPa"][i]))
              for i in range(len(d["name"]))]
    nlo_refine.bounds(model, plates[0])                                  # warm-up
    net = nlo_train.predict(model, plates)
    rows = []
    for name, p, r in zip(d["name"], plates, net):
        rf = refs[str(name)]
        nub, e1 = certify_ub(r, p["xs"], p["ys"], p["t"], p["sigma_y"])
        nlb, e2 = certify_lb(r, p["xs"], p["ys"], p["t"], p["sigma_y"])
        t1 = time.time()
        out = nlo_refine.bounds(model, p)
        dt = time.time() - t1
        ub, e3 = certify_ub(out["ub_res"], p["xs"], p["ys"], p["t"], p["sigma_y"])
        lb, e4 = certify_lb(out["lb_res"], p["xs"], p["ys"], p["t"], p["sigma_y"])
        rows.append(dict(name=str(name), net_gap=(nub - nlb) / nub, gap=(ub - lb) / ub,
                         ub_excess=ub / float(rf["UB_ref_N"]) - 1, lb_deficit=1 - lb / float(rf["LB_ref_N"]),
                         seconds=dt, certified="yes" if not (e1 or e2 or e3 or e4) else ";".join(e1 + e2 + e3 + e4)))
        print(f"{name}: net {100 * rows[-1]['net_gap']:.2f}%  hybrid {100 * rows[-1]['gap']:.2f}%  {dt:.1f}s  {rows[-1]['certified']}", flush=True)
    if a.out:
        with open(a.out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
    k = lambda key: np.mean([r[key] for r in rows])
    print("---")
    print(f"split {a.split} n={len(rows)} certified={all(r['certified'] == 'yes' for r in rows)}")
    print(f"network gap mean {100 * k('net_gap'):.2f}%   hybrid gap mean {100 * k('gap'):.2f}%  "
          f"(UB +{100 * k('ub_excess'):.2f}%, LB -{100 * k('lb_deficit'):.2f}%)  time {k('seconds'):.2f} s/plate")


if __name__ == "__main__":
    main()
