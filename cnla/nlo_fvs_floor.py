#!/usr/bin/env python3
"""Mesh floor of the C1-cubic (FVS) Airy space on the 256 x 80 reference mesh, for a split (report only).

For each plate: the best FVS lower bound on the reference mesh (nlo_fvs.optimal_lb_fvs, one global interior-point
SOCP, Clarabel), certified by the frozen step-1 verifier, against the best mesh UB already stored in
data/nlo_<split>_ref.csv (UB_mesh_N, step-1 UB solver on the same mesh). The FVS floor gap is the best bracket any
method can certify on this mesh: the ceiling for a global solver (plan C).

Usage: python nlo_fvs_floor.py --split val --out ../results/runs/pareto/fvs_floor_val.csv [--workers 4]
"""
import argparse
import csv
import os
import sys
import time

os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)


def job(args):
    split, i = args
    from eval_limit_analysis import verify
    from nlo_core import RefMesh, plate_on_mesh
    from nlo_fvs import FVS, optimal_lb_fvs, to_res_fvs
    d = np.load(os.path.join(HERE, "data", f"nlo_{split}.npz"), allow_pickle=True)
    xs, ys, t, sy, name = d["x_mm"][i], d["y_mm"][i], d["thickness_mm"][i], float(d["sigma_y_MPa"][i]), str(d["name"][i])
    mesh = RefMesh(256, 80)
    pl = plate_on_mesh(mesh, xs, ys, t)
    t0 = time.time()
    LB, Nv, status, sec = optimal_lb_fvs(mesh, pl, sy, FVS(mesh))
    v = np.zeros((len(pl["nodes"]), 2))
    v[:, 0] = pl["nodes"][:, 0] / pl["L"]
    _, lbc, errs, _ = verify(to_res_fvs(mesh, pl, v, Nv, sy), xs, ys, t, sy, "rigid")
    return dict(name=name, LB_fvs_N=lbc, status=status, solve_s=sec, total_s=time.time() - t0,
                certified="yes" if not errs else ";".join(errs))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="val")
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    from multiprocessing import Pool
    d = np.load(os.path.join(HERE, "data", f"nlo_{a.split}.npz"), allow_pickle=True)
    refs = {r["name"]: r for r in csv.DictReader(open(os.path.join(HERE, "data", f"nlo_{a.split}_ref.csv")))}
    with Pool(a.workers) as p:
        rows = p.map(job, [(a.split, i) for i in range(len(d["name"]))])
    for r in rows:
        rf = refs[r["name"]]
        ubm = float(rf["UB_mesh_N"])
        r.update(UB_mesh_N=ubm, LB_zp_N=float(rf["LB_mesh_N"]), gap_zp=float(rf["gap_mesh"]),
                 gap_fvs=(ubm - r["LB_fvs_N"]) / ubm, LB_ref_N=float(rf["LB_ref_N"]), UB_ref_N=float(rf["UB_ref_N"]),
                 gap_ref=float(rf["gap_ref"]), ref_s=float(rf["ref_s"]))
        print(f"{r['name']}: floor FVS {100 * r['gap_fvs']:.2f}%  ZP {100 * r['gap_zp']:.2f}%  FE {100 * r['gap_ref']:.2f}%  "
              f"({r['status']}, {r['solve_s']:.0f}s)  {r['certified']}", flush=True)
    with open(a.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    k = lambda key: np.mean([r[key] for r in rows])
    print(f"--- mean floor FVS {100 * k('gap_fvs'):.3f}%  ZP {100 * k('gap_zp'):.3f}%  adaptive FE {100 * k('gap_ref'):.3f}%  "
          f"FVS solve {k('solve_s'):.0f}s/plate ({a.workers} workers)")


if __name__ == "__main__":
    main()
