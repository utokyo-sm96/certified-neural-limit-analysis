#!/usr/bin/env python3
"""Probe R1 (review): how much do certified bounds change when both are certified for ONE thickness model, t_g
(verifier v2: piecewise-linear interpolant on the scan grid with alternating diagonals), instead of the piecewise-linear
interpolant of the bilinear scan interpolant on each result's own mesh (verifier v1)? Validation plates
only. For each plate: network alone, full method (nlo_refine default), adaptive FE (limit_analysis.solve_bounds).
Usage: python loops/scratch/thickness_probe.py [n_plates] > loops/scratch/thickness_probe.log"""
import csv
import os
import sys
import time

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
PF = os.path.join(ROOT, "cnla")
sys.path.insert(0, PF)


def main():
    import nlo_refine as R
    import nlo_train_fvs
    from limit_analysis import solve_bounds
    from nlo_refine_eval import SQ3, tri_area
    from verify_v2 import verify_v2
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 8
    d = np.load(os.path.join(PF, "data", "nlo_val.npz"), allow_pickle=True)
    model = nlo_train_fvs.load(os.path.join(PF, "data", "nlo_fvs_ft30k.pt"))
    rows = []

    def both(u, l, xs, ys, t, sy):
        """v1 and v2 bounds of an UB result u and an LB result l (dicts with nodes, tri, thick, v / stress)."""
        nodes, tri = np.asarray(u["nodes"], float), np.asarray(u["tri"], int)
        S = np.zeros((len(tri), 3, 3))
        S[:, :, 0] = 1.0
        ru = dict(nodes=nodes, tri=tri, thick=np.asarray(u["thick"], float), v=np.asarray(u["v"], float), ub=float(u["ub"]),
                  stress=S, lb=0.0, status_ub="Solved", status_lb="Solved")
        ub2, _, eu, iu = verify_v2(ru, xs, ys, t, sy)
        nl, tl = np.asarray(l["nodes"], float), np.asarray(l["tri"], int)
        thl = np.asarray(l["thick"], float)
        L = xs[-1]
        vl = np.stack([nl[:, 0] / L, np.zeros(len(nl))], 1)
        ubd = float(np.sum(sy * (2 / SQ3) * (1 / L) * tri_area(nl, tl) * thl[tl].mean(1)))
        rl = dict(nodes=nl, tri=tl, thick=thl, v=vl, ub=ubd, stress=np.asarray(l["stress"], float), lb=float(l["lb"]),
                  status_ub="Solved", status_lb="Solved")
        _, lb2, el, il = verify_v2(rl, xs, ys, t, sy)
        return dict(ub1=iu["ub_v1"], ub2=ub2, lb1=il["lb_v1"], lb2=lb2, errs=";".join(eu + el))

    for i in range(n):
        name = str(d["name"][i])
        xs, ys, t = d["x_mm"][i], d["y_mm"][i], d["thickness_mm"][i]
        sy = float(d["sigma_y_MPa"][i])
        p = dict(xs=xs, ys=ys, t=t, sigma_y=sy)
        net = nlo_train_fvs.predict(model, [p])[0]
        a = both(net, net, xs, ys, t, sy)
        out = R.bounds(model, p)
        b = both(out["ub_res"], out["lb_res"], xs, ys, t, sy)
        t0 = time.time()
        fe = solve_bounds(xs, ys, t, sy)
        tfe = time.time() - t0
        c = both(fe, fe, xs, ys, t, sy)
        fg = solve_bounds(xs, ys, t, sy, thick_model="tg")
        e = both(fg, fg, xs, ys, t, sy)
        print(f"   FE solved on t_g: own [{fg['lb'] / 1e3:.4f}, {fg['ub'] / 1e3:.4f}] vs v2 [{e['lb2'] / 1e3:.4f}, "
              f"{e['ub2'] / 1e3:.4f}] kN (must agree)", flush=True)
        c = e
        for meth, r in (("network", a), ("full", b), ("fe_tg", e)):
            rows.append(dict(plate=name, method=meth, **r))
            g1 = (r["ub1"] - r["lb1"]) / r["ub1"]
            g2 = (r["ub2"] - r["lb2"]) / r["ub2"]
            print(f"{name} {meth:8s} UB {r['ub1'] / 1e3:9.3f} -> {r['ub2'] / 1e3:9.3f} ({100 * (r['ub2'] / r['ub1'] - 1):+.3f} %)  "
                  f"LB {r['lb1'] / 1e3:9.3f} -> {r['lb2'] / 1e3:9.3f} ({100 * (r['lb2'] / r['lb1'] - 1):+.3f} %)  gap "
                  f"{100 * g1:.3f} -> {100 * g2:.3f} %  {r['errs']}", flush=True)
        print(f"   FE time {tfe:.1f} s; FE v2 bracket [{c['lb2'] / 1e3:.3f}, {c['ub2'] / 1e3:.3f}] vs full v2 "
              f"[{b['lb2'] / 1e3:.3f}, {b['ub2'] / 1e3:.3f}] consistent: {b['lb2'] <= c['ub2'] and b['ub2'] >= c['lb2']}",
              flush=True)
    with open(os.path.join(os.path.dirname(__file__), "thickness_probe.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


if __name__ == "__main__":
    main()
