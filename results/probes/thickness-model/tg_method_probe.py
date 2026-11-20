#!/usr/bin/env python3
"""Probe R1b: does a t_g-aware refinement recover the lower bound under verifier v2?
The refinement's nodal capacities are replaced by c_node = t_g(node) - max_{T containing node} delta_T, where
delta_T = max_T (t_lin - t_g) over the exact pieces of T; then the vertex test with c is sufficient for t_g on every
triangle (the linear interpolant of c is <= t_lin - delta_T <= t_g on T). The network's inputs are unchanged (they use
nlo_core.plate_on_mesh directly); only nlo_refine's capacity array changes (monkeypatched). Every bound is certified by
verifier v2. Validation plates only. Usage: python loops/scratch/tg_method_probe.py [n] > loops/scratch/tg_method_probe.log"""
import csv
import os
import sys
import time

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
PF = os.path.join(ROOT, "cnla")
sys.path.insert(0, PF)


def safe_capacity(mesh, nodes, xs, ys, t, tri=None):
    from thickness_tg import tg_eval, tg_quantities
    tri = mesh.tri if tri is None else np.asarray(tri, int)
    _, own, lam, tpt = tg_quantities(nodes, tri, xs, ys, t)
    tgn = tg_eval(nodes, xs, ys, t)
    delta = np.full(len(tri), -np.inf)
    np.maximum.at(delta, own, (lam * tgn[tri[own]]).sum(1) - tpt)
    delta = np.maximum(delta, 0.0)
    dn = np.zeros(len(nodes))
    np.maximum.at(dn, tri.ravel(), np.repeat(delta, 3))
    return tgn - dn


def main():
    import nlo_refine as R
    import nlo_train_fvs
    from nlo_core import plate_on_mesh
    from nlo_refine_eval import SQ3, tri_area
    from verify_v2 import verify_v2
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 8
    only = sys.argv[2].split(",") if len(sys.argv) > 2 else None
    d = np.load(os.path.join(PF, "data", "nlo_val.npz"), allow_pickle=True)
    model = nlo_train_fvs.load(os.path.join(PF, "data", "nlo_fvs_ft30k.pt"))
    mesh = R.canon()["mesh"]
    orig = R.nlo_train.plate_on_mesh
    orig_mb = R.mirror_back
    rows = []
    for i in range(len(d["name"]) if only else n):
        name = str(d["name"][i])
        if only and name not in only:
            continue
        xs, ys, t = d["x_mm"][i], d["y_mm"][i], np.asarray(d["thickness_mm"][i], float)
        sy = float(d["sigma_y_MPa"][i])
        pl = plate_on_mesh(mesh, xs, ys, t)
        cap = safe_capacity(mesh, pl["nodes"], xs, ys, t)

        def patched(m, xs_, ys_, t_, _cap=cap):
            out = orig(m, xs_, ys_, t_)
            return dict(out, thick=_cap)
        def mirror_back(r, fx, fy, L_, W_, _xs=xs, _ys=ys, _t=t, _sy=sy):
            """Mirror fields rescaled to the same t_g-safe capacities as the refinement (comparable lower bounds)."""
            o = orig_mb(r, fx, fy, L_, W_)
            c_ = safe_capacity(mesh, o["nodes"], _xs, _ys, _t, tri=o["tri"])
            S_ = np.asarray(o["stress"], float)
            f_ = np.sqrt(np.maximum(S_[..., 0] ** 2 - S_[..., 0] * S_[..., 1] + S_[..., 1] ** 2 + 3 * S_[..., 2] ** 2, 0))
            r_cap = (f_ / (_sy * c_[o["tri"]])).max()
            r_own = (f_ / (_sy * np.asarray(o["thick"])[o["tri"]])).max()
            return dict(o, stress=S_ / r_cap, lb=o["lb"] * r_own / r_cap)
        R.nlo_train.plate_on_mesh = patched
        R.mirror_back = mirror_back
        t0 = time.time()
        out = R.bounds(model, dict(xs=xs, ys=ys, t=t, sigma_y=sy))
        dt = time.time() - t0
        R.nlo_train.plate_on_mesh = orig
        R.mirror_back = orig_mb
        u, l = out["ub_res"], out["lb_res"]
        nodes, tri = np.asarray(u["nodes"], float), np.asarray(u["tri"], int)
        S = np.zeros((len(tri), 3, 3))
        S[:, :, 0] = 1.0
        bil = orig(mesh, xs, ys, t)["thick"]                             # v1 structure check expects bilinear nodes
        thick_u = bil if len(nodes) == len(bil) else np.asarray(u["thick"], float)
        ub2, _, eu, _ = verify_v2(dict(nodes=nodes, tri=tri, thick=thick_u, v=np.asarray(u["v"], float), ub=float(u["ub"]),
                                       stress=S, lb=0.0, status_ub="Solved", status_lb="Solved"), xs, ys, t, sy)
        nl, tl = np.asarray(l["nodes"], float), np.asarray(l["tri"], int)
        thick_l = bil if len(nl) == len(bil) else np.asarray(l["thick"], float)
        L = xs[-1]
        vl = np.stack([nl[:, 0] / L, np.zeros(len(nl))], 1)
        ubd = float(np.sum(sy * (2 / SQ3) * (1 / L) * tri_area(nl, tl) * thick_l[tl].mean(1)))
        _, lb2, el, _ = verify_v2(dict(nodes=nl, tri=tl, thick=thick_l, v=vl, ub=ubd, stress=np.asarray(l["stress"], float),
                                       lb=float(l["lb"]), status_ub="Solved", status_lb="Solved"), xs, ys, t, sy)
        rows.append(dict(plate=name, lb2=lb2, ub2=ub2, gap2=(ub2 - lb2) / ub2, seconds=dt, errs=";".join(eu + el)))
        print(f"{name}: t_g-aware full method v2 [{lb2 / 1e3:.3f}, {ub2 / 1e3:.3f}] kN gap {100 * rows[-1]['gap2']:.3f} %  "
              f"{dt:.1f} s  {rows[-1]['errs']}", flush=True)
    with open(os.path.join(os.path.dirname(__file__), "tg_method_probe.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


if __name__ == "__main__":
    main()
