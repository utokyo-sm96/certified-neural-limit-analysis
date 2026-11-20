#!/usr/bin/env python3
"""FE limit analysis on the common thickness model t_g for the HELD-OUT plates (64 synthetic test plates, 28 real scans):
the references of the held-out evaluation under verifier v2, and the refinement-level brackets for the equal-accuracy
comparison. Same procedure as results/probes/thickness-model/fe_levels.py (pass 1 = UB on the scan grid locating the plastic
zone, then both bounds on meshes refined to a maximum level 0..4; level 4 = the reference solver), every bracket
certified by verifier v2. Runs in a worker pool; a few validation plates with known idle timings (levels 0-2,
fe_levels_0_32_L2.csv) run in the same pool to measure the slowdown under load.
Outputs: data/nlo_test_ref_tg.csv, data/real_ref_tg.csv (level-4 brackets) and <out>/fe_levels_heldout.csv (all levels).
Usage: python fe_tg_heldout.py --out ../results/runs/heldout-v2 [--workers 6] [--n-val 6]"""
import argparse
import csv
import os
import sys
import time
from multiprocessing import Pool

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "paper", "figures"))


def levels(job):
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    import limit_analysis as LA
    from thickness_tg import tg_eval
    from verify_v2 import verify_v2
    sset, name, xs, ys, t, sy, lmax = job
    L, W = xs[-1], ys[-1]
    t0 = float(t.max())
    tol = 1e-6 * max(L, W)
    interp = lambda q: tg_eval(np.asarray(q, float), xs, ys, t)
    clock = time.perf_counter()
    nodes, tri = LA.grid_mesh(xs, ys)
    ub0 = LA.upper_bound(nodes, tri, interp(nodes), t0, L, "rigid", tol)
    ncell = (len(xs) - 1) * (len(ys) - 1)
    cell_diss = ub0["diss"][:ncell] + ub0["diss"][ncell:]
    rank = np.argsort(cell_diss)[::-1]
    if cell_diss.max() < LA.UNIFORM_RATIO * cell_diss.mean():
        rank = rank[:0]
    spent = time.perf_counter() - clock
    rows = []
    for cap in range(lmax + 1):
        level = np.zeros(ncell, int)
        for lv, frac in ((1, LA.REFINE_FRACTION), (2, LA.LEVEL2_FRACTION), (3, LA.LEVEL3_FRACTION), (4, LA.LEVEL4_FRACTION)):
            if lv <= cap:
                level[rank[:int(frac * ncell)]] = lv
        c0 = time.perf_counter()
        nodes, tri = LA.adaptive_mesh(xs, ys, level.reshape(len(xs) - 1, len(ys) - 1))
        thick = interp(nodes)
        ub = LA.upper_bound(nodes, tri, thick, t0, L, "rigid", tol)
        lb = LA.lower_bound(nodes, tri, thick, t0, L, W, "rigid", tol)
        spent += time.perf_counter() - c0
        scale = sy * t0
        res = dict(nodes=nodes, tri=tri, thick=thick, ub=ub["ub_hat"] * scale, v=ub["v"], status_ub=ub["status"],
                   lb=lb["lb_hat"] * scale, stress=lb["stress_hat"] * scale, status_lb=lb["status"])
        ubc, lbc, errs, _ = verify_v2(res, xs, ys, t, sy)
        rows.append(dict(set=sset, plate=name, level=cap, triangles=len(tri), t_cum=spent, lb=lbc, ub=ubc,
                         gap=(ubc - lbc) / ubc, certified="yes" if not errs else ";".join(errs)))
    print(f"{sset:5s} {name}: " + "  ".join(f"L{r['level']} {100 * r['gap']:.2f}% {r['t_cum']:.0f}s" for r in rows)
          + ("" if all(r["certified"] == "yes" for r in rows) else "  NOT CERTIFIED"), flush=True)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--n-val", type=int, default=6)
    a = ap.parse_args()
    from val_histories import load_set
    jobs = []
    for s in ("real", "test"):
        jobs += [(s, p["name"], np.asarray(p["xs"], float), np.asarray(p["ys"], float), np.asarray(p["t"], float), p["sigma_y"], 4)
                 for p in load_set(s)]
    jobs += [("val", p["name"], np.asarray(p["xs"], float), np.asarray(p["ys"], float), np.asarray(p["t"], float), p["sigma_y"], 2)
             for p in load_set("val")[:a.n_val]]
    os.makedirs(a.out, exist_ok=True)
    with Pool(a.workers, maxtasksperchild=4) as pool:
        out = pool.map(levels, jobs, chunksize=1)
    rows = [r for rr in out for r in rr]
    with open(os.path.join(a.out, "fe_levels_heldout.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    for s, fn in (("test", "nlo_test_ref_tg.csv"), ("real", "real_ref_tg.csv")):
        with open(os.path.join(HERE, "data", fn), "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["name", "LB_ref_N", "UB_ref_N", "gap_ref", "seconds_cum", "triangles"])
            for r in rows:
                if r["set"] == s and r["level"] == 4:
                    w.writerow([r["plate"], r["lb"], r["ub"], r["gap"], r["t_cum"], r["triangles"]])
    idle = {(r["plate"], int(r["level"])): float(r["t_cum"])
            for r in csv.DictReader(open(os.path.join(HERE, "..", "docs", "probes", "thickness-model", "fe_levels_0_32_L2.csv")))}
    fac = [r["t_cum"] / idle[(r["plate"], r["level"])] for r in rows if r["set"] == "val" and (r["plate"], r["level"]) in idle]
    print(f"done: {len(rows)} brackets, certified {sum(r['certified'] == 'yes' for r in rows)}; load factor (val plates, "
          f"levels 0-2) median {np.median(fac):.2f} (range {min(fac):.2f}-{max(fac):.2f})")


if __name__ == "__main__":
    main()
