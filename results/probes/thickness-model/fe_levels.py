#!/usr/bin/env python3
"""Review point R3 (matched accuracy): adaptive FE limit analysis as a sequence of refinement levels with a stopping
criterion. For each plate: pass 1 (UB on the scan grid, locates the plastic zone), then both bounds on meshes refined to
a maximum level of 0, 1, 2, 3, 4 (4 = the reference solver). Records the cumulative single-process time and the
certified bracket (verifier v2, thickness model t_g) after each level, so that the time to reach a gap threshold with
"stop when the certified gap is below the threshold" can be read off. Validation plates only.
Usage: python loops/scratch/fe_levels.py [first] [last] > loops/scratch/fe_levels.log"""
import csv
import os
import sys
import time

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
PF = os.path.join(ROOT, "cnla")
sys.path.insert(0, PF)


def main():
    import limit_analysis as LA
    from thickness_tg import tg_eval
    from verify_v2 import verify_v2
    a = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    b = int(sys.argv[2]) if len(sys.argv) > 2 else 32
    d = np.load(os.path.join(PF, "data", "nlo_val.npz"), allow_pickle=True)
    out = os.path.join(os.path.dirname(__file__), f"fe_levels_{a}_{b}.csv")
    rows = []
    for i in range(a, min(b, len(d["name"]))):
        name = str(d["name"][i])
        xs, ys, t = (np.asarray(d[k][i], float) for k in ("x_mm", "y_mm", "thickness_mm"))
        sy = float(d["sigma_y_MPa"][i])
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
        t_pass1 = time.perf_counter() - clock
        spent = t_pass1
        for cap in (0, 1, 2, 3, 4):
            level = np.zeros(ncell, int)
            for lv, frac in ((1, LA.REFINE_FRACTION), (2, LA.LEVEL2_FRACTION), (3, LA.LEVEL3_FRACTION), (4, LA.LEVEL4_FRACTION)):
                if lv <= cap:
                    level[rank[:int(frac * ncell)]] = lv
            c0 = time.perf_counter()
            nodes, tri = LA.adaptive_mesh(xs, ys, level.reshape(len(xs) - 1, len(ys) - 1))
            thick = interp(nodes)
            ub = LA.upper_bound(nodes, tri, thick, t0, L, "rigid", tol)
            lb = LA.lower_bound(nodes, tri, thick, t0, L, W, "rigid", tol)
            solve = time.perf_counter() - c0
            spent += solve
            scale = sy * t0
            res = dict(nodes=nodes, tri=tri, thick=thick, ub=ub["ub_hat"] * scale, v=ub["v"], status_ub=ub["status"],
                       lb=lb["lb_hat"] * scale, stress=lb["stress_hat"] * scale, status_lb=lb["status"])
            c1 = time.perf_counter()
            ubc, lbc, errs, _ = verify_v2(res, xs, ys, t, sy)
            ver = time.perf_counter() - c1
            gap = (ubc - lbc) / ubc
            rows.append(dict(plate=name, level=cap, triangles=len(tri), t_pass1=t_pass1, t_solve=solve, t_cum=spent,
                             t_verify=ver, lb=lbc, ub=ubc, gap=gap, certified="yes" if not errs else ";".join(errs)))
            print(f"{name} level<={cap}: {len(tri):7d} tri  cum {spent:6.1f} s (verify {ver:.1f} s)  "
                  f"[{lbc / 1e3:.3f}, {ubc / 1e3:.3f}] kN  gap {100 * gap:.3f} %  {rows[-1]['certified']}", flush=True)
        with open(out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)


if __name__ == "__main__":
    main()
