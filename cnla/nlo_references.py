#!/usr/bin/env python3
"""Reference solutions for the neural-limit-operator validation and test plates (run once, in setup).

For every plate:
  * step-1 FE limit analysis (adaptive mesh, limit_analysis.solve_bounds, rigid grips), certified by the
    frozen verifier: LB_ref <= collapse load <= UB_ref, bracket ~0.4 %;
  * the best bounds reachable on the NLO reference mesh (uniform union-jack NX x NY): optimal UB over
    piecewise-linear velocities, optimal LB over ZP-Airy constant-stress fields. These are a floor for any
    network that uses that mesh, and are reported only.

Usage: python nlo_references.py --split val --workers 10
Output: data/nlo_<split>_ref.csv
"""
import argparse
import csv
import os
import sys
import time

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
NX, NY = 256, 80


def work(args):
    import torch
    torch.set_num_threads(1)
    from eval_limit_analysis import verify
    from limit_analysis import solve_bounds
    from nlo_core import RefMesh, optimal_lb_zp, optimal_ub, plate_on_mesh, to_res
    name, xs, ys, t, sy, skip_mesh = args
    t1 = time.time()
    r = solve_bounds(xs, ys, t, sy, grips="rigid")
    ub, lb, errs, _ = verify(r, xs, ys, t, sy, "rigid")
    t_fe = time.time() - t1
    if skip_mesh:
        return dict(name=name, L_mm=xs[-1], W_mm=ys[-1], sigma_y=sy, mean_t=float(t.mean()),
                    LB_ref_N=lb, UB_ref_N=ub, gap_ref=(ub - lb) / ub, ref_ok="yes" if not errs else ";".join(errs),
                    ref_s=t_fe, UB_mesh_N=float("nan"), LB_mesh_N=float("nan"), gap_mesh=float("nan"),
                    mesh_ok="yes", mesh_status="skipped", mesh_s=0.0)
    mesh = RefMesh(NX, NY)
    pl = plate_on_mesh(mesh, xs, ys, t)
    t1 = time.time()
    ub_m, v, s1, _ = optimal_ub(mesh, pl, sy)
    lb_m, N, s2, _ = optimal_lb_zp(mesh, pl, sy)
    ub_mv, lb_mv, errs_m, _ = verify(to_res(mesh, pl, v, N, sy), xs, ys, t, sy, "rigid")
    t_m = time.time() - t1
    return dict(name=name, L_mm=xs[-1], W_mm=ys[-1], sigma_y=sy, mean_t=float(t.mean()),
                LB_ref_N=lb, UB_ref_N=ub, gap_ref=(ub - lb) / ub, ref_ok="yes" if not errs else ";".join(errs),
                ref_s=t_fe, UB_mesh_N=ub_mv, LB_mesh_N=lb_mv, gap_mesh=(ub_mv - lb_mv) / ub_mv,
                mesh_ok="yes" if not errs_m else ";".join(errs_m), mesh_status=f"{s1}/{s2}", mesh_s=t_m)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True, choices=["val", "test"])
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--skip-mesh", action="store_true", help="FE references only (no mesh-floor bounds)")
    a = ap.parse_args()
    d = np.load(os.path.join(HERE, "data", f"nlo_{a.split}.npz"), allow_pickle=True)
    jobs = [(str(d["name"][i]), d["x_mm"][i], d["y_mm"][i], d["thickness_mm"][i], float(d["sigma_y_MPa"][i]), a.skip_mesh)
            for i in range(len(d["name"]))]
    from multiprocessing import Pool
    rows = []
    with Pool(a.workers) as pool:
        for r in pool.imap_unordered(work, jobs):
            rows.append(r)
            print(f"{r['name']}: ref [{r['LB_ref_N'] / 1e3:.2f}, {r['UB_ref_N'] / 1e3:.2f}] kN {100 * r['gap_ref']:.2f}% "
                  f"{r['ref_ok']}; mesh floor {100 * r['gap_mesh']:.2f}% {r['mesh_ok']} ({r['ref_s']:.0f}s+{r['mesh_s']:.0f}s)",
                  flush=True)
    rows.sort(key=lambda r: r["name"])
    out = os.path.join(HERE, "data", f"nlo_{a.split}_ref.csv")
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        for r in rows:
            w.writerow({k: (f"{v:.6g}" if isinstance(v, float) else v) for k, v in r.items()})
    print(f"wrote {out}; all certified: {all(r['ref_ok'] == 'yes' and r['mesh_ok'] == 'yes' for r in rows)}; "
          f"mean ref gap {100 * np.mean([r['gap_ref'] for r in rows]):.2f}%, "
          f"mean mesh floor {100 * np.mean([r['gap_mesh'] for r in rows]):.2f}%")


if __name__ == "__main__":
    main()
