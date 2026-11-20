#!/usr/bin/env python3
"""Frozen evaluator of loop tg-method-1112 (review R1): the full method with THICK_MODEL = "tg" on the 32 VALIDATION
plates, every bracket certified for the common thickness model t_g by verifier v2 (verify_v2.py).

Gates (any failure -> gates: FAIL):
  G1 every LB and UB certified by verify_v2 (no errors) and LB <= UB;
  G2 weak duality against the FE references on t_g (data/nlo_val_ref_tg.csv, from loops/scratch/fe_levels.py, level 4):
     LB <= UB_ref (1 + 1e-6) and UB >= LB_ref (1 - 1e-6);
  G3 time per plate (nlo_refine.bounds, wall clock): mean <= 15 s, max <= 30 s.
Metric: mean certified gap (UB - LB)/UB (lower is better).
Usage: python nlo_tg_eval.py [n_plates]   (prints the summary block for loop_tool.py)"""
import csv
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)


def main():
    t_start = time.time()
    import nlo_refine as R
    import nlo_train_fvs
    from nlo_refine_eval import SQ3, tri_area
    from verify_v2 import verify_v2
    R.THICK_MODEL = "tg"
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 32
    d = np.load(os.path.join(HERE, "data", "nlo_val.npz"), allow_pickle=True)
    ref_path = os.path.join(HERE, "data", "nlo_val_ref_tg.csv")
    refs = {r["name"]: r for r in csv.DictReader(open(ref_path))} if os.path.exists(ref_path) else {}
    model = nlo_train_fvs.load(os.path.join(HERE, "data", "nlo_fvs_ft30k.pt"))
    p0 = dict(xs=d["x_mm"][0], ys=d["y_mm"][0], t=d["thickness_mm"][0], sigma_y=float(d["sigma_y_MPa"][0]))
    R.bounds(model, p0)                                                   # warm-up (offline factorisations)
    gaps, times, fails, lbdef, ubex = [], [], [], [], []
    for i in range(min(n, len(d["name"]))):
        name = str(d["name"][i])
        xs, ys, t = d["x_mm"][i], d["y_mm"][i], np.asarray(d["thickness_mm"][i], float)
        sy = float(d["sigma_y_MPa"][i])
        t0 = time.time()
        out = R.bounds(model, dict(xs=xs, ys=ys, t=t, sigma_y=sy))
        dt = time.time() - t0
        u, l = out["ub_res"], out["lb_res"]
        nodes, tri = np.asarray(u["nodes"], float), np.asarray(u["tri"], int)
        S = np.zeros((len(tri), 3, 3))
        S[:, :, 0] = 1.0
        ub, _, eu, _ = verify_v2(dict(nodes=nodes, tri=tri, thick=np.asarray(u["thick"], float), v=np.asarray(u["v"], float),
                                      ub=float(u["ub"]), stress=S, lb=0.0, status_ub="Solved", status_lb="Solved"), xs, ys, t, sy)
        nl, tl = np.asarray(l["nodes"], float), np.asarray(l["tri"], int)
        thl = np.asarray(l["thick"], float)
        L = xs[-1]
        vl = np.stack([nl[:, 0] / L, np.zeros(len(nl))], 1)
        ubd = float(np.sum(sy * (2 / SQ3) * (1 / L) * tri_area(nl, tl) * thl[tl].mean(1)))
        _, lb, el, _ = verify_v2(dict(nodes=nl, tri=tl, thick=thl, v=vl, ub=ubd, stress=np.asarray(l["stress"], float),
                                      lb=float(l["lb"]), status_ub="Solved", status_lb="Solved"), xs, ys, t, sy)
        errs = eu + el
        if lb > ub * (1 + 1e-6):
            errs.append("LB > UB")
        if name in refs:
            ub_ref, lb_ref = float(refs[name]["UB_ref_N"]), float(refs[name]["LB_ref_N"])
            if lb > ub_ref * (1 + 1e-6) or ub < lb_ref * (1 - 1e-6):
                errs.append("inconsistent with the FE reference on t_g")
            lbdef.append(1 - lb / lb_ref)
            ubex.append(ub / ub_ref - 1)
        gaps.append((ub - lb) / ub)
        times.append(dt)
        if errs:
            fails.append(f"{name}: {errs[0]}")
        print(f"{name}: [{lb / 1e3:.3f}, {ub / 1e3:.3f}] kN  gap {100 * gaps[-1]:.3f} %  {dt:.1f} s  {';'.join(errs)}", flush=True)
    gaps, times = np.array(gaps), np.array(times)
    gate = []
    if fails:
        gate.append(f"G1/G2 {len(fails)} plates ({fails[0]})")
    if times.mean() > 15 or times.max() > 30:
        gate.append(f"G3 time mean {times.mean():.2f}s max {times.max():.2f}s")
    if len(refs) < len(gaps):
        gate.append("G2 references on t_g missing")
    print("---")
    print(f"mean_gap: {gaps.mean():.6f}")
    print("gates: PASS" if not gate else "gates: FAIL " + "; ".join(gate))
    print(f"runtime_s: {time.time() - t_start:.1f}")
    print(f"median_gap: {np.median(gaps):.6f}")
    print(f"max_gap: {gaps.max():.6f}")
    print(f"time_mean_s: {times.mean():.2f}")
    print(f"time_max_s: {times.max():.2f}")
    if lbdef:
        print(f"lb_deficit: {np.mean(lbdef):.6f}")
        print(f"ub_excess: {np.mean(ubex):.6f}")


if __name__ == "__main__":
    main()
