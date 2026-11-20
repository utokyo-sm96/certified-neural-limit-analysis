#!/usr/bin/env python3
"""Probe evaluator (not frozen): any FVS checkpoint under the common thickness model t_g on the VALIDATION plates.
Same pipeline and certification as cnla/nlo_tg_eval.py (nlo_refine.THICK_MODEL = "tg", verifier v2), plus the
network-alone bracket (identity mirror, scaled to its exact v2 yield ratio). Prints per-plate rows and means; writes csv.
Usage: python loops/scratch/tg_ckpt_eval.py CKPT OUT_CSV [n_plates] [--net-only]"""
import csv
import os
import sys
import time

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
PF = os.path.join(ROOT, "cnla")
sys.path.insert(0, PF)


def main():
    import torch
    import nlo_refine as R
    import nlo_train_fvs
    from nlo_core import dissipation, plate_on_mesh
    from nlo_fvs import to_res_fvs
    from nlo_refine_eval import SQ3, tri_area
    from verify_v2 import verify_v2
    from thickness_tg import tg_quantities
    from tg_capacity import tg_ratio
    R.THICK_MODEL = "tg"
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    net_only = "--net-only" in sys.argv
    ckpt, out_csv = args[0], args[1]
    n = int(args[2]) if len(args) > 2 else 32
    d = np.load(os.path.join(PF, "data", "nlo_val.npz"), allow_pickle=True)
    refs = {r["name"]: r for r in csv.DictReader(open(os.path.join(PF, "data", "nlo_val_ref_tg.csv")))}
    model = nlo_train_fvs.load(ckpt)
    mesh = R.canon()["mesh"]
    tri = mesh.tri
    grab = {}
    o_can = R.to_canonical

    def w_can(r, L_, W_):
        vv, Nv = o_can(r, L_, W_)
        grab.setdefault("first", (vv.copy(), Nv.copy()))
        return vv, Nv
    R.to_canonical = w_can

    def v2_lb(nl, tl, thl, stress, lb, xs, ys, t, sy):
        vl = np.stack([nl[:, 0] / xs[-1], np.zeros(len(nl))], 1)
        ubd = float(np.sum(sy * (2 / SQ3) * (1 / xs[-1]) * tri_area(nl, tl) * thl[tl].mean(1)))
        _, lbc, e, _ = verify_v2(dict(nodes=nl, tri=tl, thick=thl, v=vl, ub=ubd, stress=stress, lb=lb,
                                      status_ub="Solved", status_lb="Solved"), xs, ys, t, sy)
        return lbc, e

    def v2_ub(nodes, tri_, th, v, ub, xs, ys, t, sy):
        S = np.zeros((len(tri_), 3, 3))
        S[:, :, 0] = 1.0
        ubc, _, e, _ = verify_v2(dict(nodes=nodes, tri=tri_, thick=th, v=v, ub=ub, stress=S, lb=0.0,
                                      status_ub="Solved", status_lb="Solved"), xs, ys, t, sy)
        return ubc, e

    p0 = dict(xs=d["x_mm"][0], ys=d["y_mm"][0], t=d["thickness_mm"][0], sigma_y=float(d["sigma_y_MPa"][0]))
    R.bounds(model, p0)                                                   # warm-up
    rows = []
    for i in range(min(n, len(d["name"]))):
        name = str(d["name"][i])
        xs, ys, t = d["x_mm"][i], d["y_mm"][i], np.asarray(d["thickness_mm"][i], float)
        sy = float(d["sigma_y_MPa"][i])
        pl = plate_on_mesh(mesh, xs, ys, t)
        nodes, th = pl["nodes"], pl["thick"]
        grab.clear()
        t0 = time.time()
        if net_only:                                                     # identity mirror only, as in R._bounds
            r0 = R.mirror_back(nlo_train_fvs.predict(model, [dict(xs=xs, ys=ys, t=t, sigma_y=sy)])[0], False, False,
                               xs[-1], ys[-1])
            R.to_canonical(r0, xs[-1], ys[-1])
            out = None
        else:
            out = R.bounds(model, dict(xs=xs, ys=ys, t=t, sigma_y=sy))
        dt = time.time() - t0
        vv, Nv = grab["first"]
        # network alone: LB from the identity-mirror stress field, UB from its velocity, both certified by v2
        rr = tg_ratio(Nv, tg_quantities(nodes, tri, xs, ys, t), sy)          # exact t_g yield (piece vertices)
        lb_n, e1 = v2_lb(nodes, tri, th, Nv / rr, float(R.lb_v(Nv, pl["W"]) / rr), xs, ys, t, sy)
        ub0 = float(dissipation(torch.tensor(vv[None]), torch.tensor(nodes[None]), torch.tensor(tri),
                                torch.tensor(th[None]))[0]) * sy
        ub_n, e2 = v2_ub(nodes, tri, th, vv, ub0, xs, ys, t, sy)
        row = dict(plate=name, net_lb=lb_n, net_ub=ub_n, net_gap=(ub_n - lb_n) / ub_n)
        errs = e1 + e2
        if out is not None:
            u, l = out["ub_res"], out["lb_res"]
            ub, eu = v2_ub(np.asarray(u["nodes"], float), np.asarray(u["tri"], int), np.asarray(u["thick"], float),
                           np.asarray(u["v"], float), float(u["ub"]), xs, ys, t, sy)
            lb, el = v2_lb(np.asarray(l["nodes"], float), np.asarray(l["tri"], int), np.asarray(l["thick"], float),
                           np.asarray(l["stress"], float), float(l["lb"]), xs, ys, t, sy)
            errs += eu + el
            row.update(lb=lb, ub=ub, gap=(ub - lb) / ub, time_s=dt)
        lr, ur = float(refs[name]["LB_ref_N"]), float(refs[name]["UB_ref_N"])
        row.update(lb_ref=lr, ub_ref=ur, errs=";".join(errs))
        rows.append(row)
        print(f"{name}: network gap {100 * row['net_gap']:.2f} % (LB def {100 * (1 - lb_n / lr):.2f}, UB ex "
              f"{100 * (ub_n / ur - 1):.2f})" + (f"  full gap {100 * row['gap']:.3f} % (LB def {100 * (1 - row['lb'] / lr):.2f}"
              f", UB ex {100 * (row['ub'] / ur - 1):.2f}) {dt:.1f} s" if out is not None else "") + f"  {row['errs']}", flush=True)
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    a = lambda k: np.mean([r[k] for r in rows])
    print(f"---\nckpt {ckpt}: {len(rows)} plates, errors {sum(bool(r['errs']) for r in rows)}")
    print(f"network: gap {100 * a('net_gap'):.3f} %  LB deficit {100 * np.mean([1 - r['net_lb'] / r['lb_ref'] for r in rows]):.3f} %"
          f"  UB excess {100 * np.mean([r['net_ub'] / r['ub_ref'] - 1 for r in rows]):.3f} %")
    if "gap" in rows[0]:
        print(f"full: gap {100 * a('gap'):.3f} % (median {100 * np.median([r['gap'] for r in rows]):.3f}, max "
              f"{100 * max(r['gap'] for r in rows):.3f})  LB deficit {100 * np.mean([1 - r['lb'] / r['lb_ref'] for r in rows]):.3f} %"
              f"  UB excess {100 * np.mean([r['ub'] / r['ub_ref'] - 1 for r in rows]):.3f} %  time {a('time_s'):.2f} s "
              f"(max {max(r['time_s'] for r in rows):.2f})")


if __name__ == "__main__":
    main()
