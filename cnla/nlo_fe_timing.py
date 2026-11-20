#!/usr/bin/env python3
"""Single-process timing of the step-1 adaptive FE limit analysis on a sample of a split (for the speed/accuracy
figure; the stored reference times were measured with 8-10 parallel workers). Bounds are re-verified.
Usage: python nlo_fe_timing.py --split val --n 8 --out ../results/runs/pareto/fe_timing_val.csv"""
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
    ap.add_argument("--split", default="val")
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    from eval_limit_analysis import verify
    from limit_analysis import solve_bounds
    d = np.load(os.path.join(HERE, "data", f"nlo_{a.split}.npz"), allow_pickle=True)
    idx = np.linspace(0, len(d["name"]) - 1, a.n).round().astype(int)       # evenly spread sample, fixed
    rows = []
    for i in idx:
        xs, ys, t, sy = d["x_mm"][i], d["y_mm"][i], d["thickness_mm"][i], float(d["sigma_y_MPa"][i])
        t1 = time.time()
        r = solve_bounds(xs, ys, t, sy, grips="rigid")
        dt = time.time() - t1
        ub, lb, errs, _ = verify(r, xs, ys, t, sy, "rigid")
        rows.append(dict(name=str(d["name"][i]), ub_N=ub, lb_N=lb, gap=(ub - lb) / ub, seconds=dt,
                         certified="yes" if not errs else ";".join(errs)))
        print(f"{rows[-1]['name']}: gap {100 * rows[-1]['gap']:.3f}%  {dt:.1f}s  {rows[-1]['certified']}", flush=True)
    with open(a.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"--- FE single process: gap mean {100 * np.mean([r['gap'] for r in rows]):.3f}%  "
          f"time mean {np.mean([r['seconds'] for r in rows]):.1f}s  median {np.median([r['seconds'] for r in rows]):.1f}s")


if __name__ == "__main__":
    main()
