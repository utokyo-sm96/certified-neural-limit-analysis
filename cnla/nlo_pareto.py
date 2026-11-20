#!/usr/bin/env python3
"""Speed/accuracy curve of the certified hybrid over time budgets (paper figure; not a loop evaluator).

For each budget level the hybrid's knobs (nlo_refine module constants) are set by a simple, untuned schedule, and the
32 validation plates are run one at a time in this process, timed exactly as in the loop evaluators (network included),
and certified independently (nlo_refine_eval.certify_ub / certify_lb with the frozen step-1 verifier).
The machine must be otherwise idle. Output: one CSV row per (level, plate).

Usage: python nlo_pareto.py --ckpt data/nlo_fvs_ft30k.pt --out ../results/runs/pareto/pareto_val.csv [--levels comb,b1,...]
"""
import argparse
import csv
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

# level: (LB_SECONDS, LB_MAX_ROUNDS, LB_MAX_DOF, UB_ROUNDS, UB_MAX_NODES, UB_MAX_ITER); b3 = the loop method
LEVELS = {
    "comb": (0.0, 0, 6000, 0, 16000, 15),       # mirror ensemble + convex combinations only
    "b1": (0.6, 8, 4000, 1, 8000, 15),
    "b2": (1.2, 8, 6000, 1, 12000, 15),
    "b3": (2.3, 8, 6000, 1, 16000, 15),
    "b4": (5.0, 20, 8000, 2, 24000, 15),
    "b5": (10.0, 40, 12000, 2, 32000, None),
    "b6": (20.0, 80, 16000, 3, 48000, None),
}
# plan C global stage on top of the default hybrid (b3): level -> (base level, arm, ADMM seconds, repair seconds)
GLOBAL = {
    "g1": ("b3", "zp", 1.0, 2.5),
    "g2": ("b3", "zp", 2.0, 2.5),
    "g3": ("b3", "zp", 3.0, 2.5),
    "f5": ("b3", "fvs", 5.0, 2.5),
}
# plus the global UB stage: level -> (global level, UB ADMM seconds)
GLOBAL_UB = {"g1u2": ("g1", 2.0), "g1u4": ("g1", 4.0), "f5u4": ("f5", 4.0)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="data/nlo_fvs_ft30k.pt")
    ap.add_argument("--out", required=True)
    ap.add_argument("--levels", default=",".join(LEVELS))   # also: g1, g2, g3, f5 (global stage, plan C)
    a = ap.parse_args()
    import nlo_refine as R
    import nlo_train
    import nlo_train_fvs
    from nlo_refine_eval import certify_lb, certify_ub
    ck = os.path.join(HERE, a.ckpt) if not os.path.isabs(a.ckpt) else a.ckpt
    model = nlo_train_fvs.load(ck)
    if not model.use_fvs:
        model = nlo_train.load(ck)
    d = np.load(os.path.join(HERE, "data", "nlo_val.npz"), allow_pickle=True)
    plates = [dict(xs=d["x_mm"][i], ys=d["y_mm"][i], t=d["thickness_mm"][i], sigma_y=float(d["sigma_y_MPa"][i]))
              for i in range(len(d["name"]))]
    R.bounds(model, plates[0])                                           # warm-up (caches, CUDA)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    new = not os.path.exists(a.out)
    f = open(a.out, "a", newline="")
    w = csv.DictWriter(f, fieldnames=["level", "lb_seconds", "lb_max_rounds", "lb_max_dof", "ub_rounds", "ub_max_nodes",
                                      "ub_max_iter", "global_arm", "global_seconds", "repair_seconds", "ub_global_seconds", "name", "ub_N", "lb_N",
                                      "gap", "seconds", "certified"])
    if new:
        w.writeheader()
    for lev in a.levels.split(","):
        glev, usec = GLOBAL_UB.get(lev, (lev, 0.0))
        base, garm, gsec, rsec = GLOBAL.get(glev, (glev, "zp", 0.0, 2.5))
        R.UB_GLOBAL_SECONDS = usec
        cfg = LEVELS[base]
        R.LB_SECONDS, R.LB_MAX_ROUNDS, R.LB_MAX_DOF, R.UB_ROUNDS, R.UB_MAX_NODES, R.UB_MAX_ITER = cfg
        R.GLOBAL_ARM, R.GLOBAL_SECONDS, R.REPAIR_SECONDS = garm, gsec, rsec
        R.GLOBAL_ITERS, R.UB_GLOBAL_ITERS = None, None                   # this sweep uses time budgets
        if gsec > 0 or usec > 0:
            R.bounds(model, plates[0])                                   # warm-up: offline setup of the global arm
        gaps, times = [], []
        for name, p in zip(d["name"], plates):
            t1 = time.time()
            out = R.bounds(model, p)
            dt = time.time() - t1
            ub, e1 = certify_ub(out["ub_res"], p["xs"], p["ys"], p["t"], p["sigma_y"])
            lb, e2 = certify_lb(out["lb_res"], p["xs"], p["ys"], p["t"], p["sigma_y"])
            gaps.append((ub - lb) / ub)
            times.append(dt)
            w.writerow(dict(level=lev, lb_seconds=cfg[0], lb_max_rounds=cfg[1], lb_max_dof=cfg[2], ub_rounds=cfg[3],
                            ub_max_nodes=cfg[4], ub_max_iter=cfg[5], global_arm=garm, global_seconds=gsec,
                            repair_seconds=rsec, ub_global_seconds=usec, name=str(name), ub_N=ub, lb_N=lb, gap=gaps[-1],
                            seconds=dt, certified="yes" if not (e1 or e2) else ";".join(e1 + e2)))
            f.flush()
        print(f"{lev}: gap mean {100 * np.mean(gaps):.3f}%  median {100 * np.median(gaps):.3f}%  "
              f"time mean {np.mean(times):.2f}s  max {np.max(times):.2f}s", flush=True)
    f.close()


if __name__ == "__main__":
    main()
