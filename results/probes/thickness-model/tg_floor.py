"""Probe: ceiling of the lower bound on the network's 256x80 mesh under the common thickness model t_g.
Global interior-point optimum over all C1-cubic (FVS) Airy fields with t_g-safe per-vertex capacities
(nlo_refine.tg_context), certified by verifier v2; compared with the corrected method's LB and the FE reference LB on t_g.
Usage: python loops/scratch/tg_floor.py i [i ...]"""
import csv, os, sys, time
os.environ.setdefault("OMP_NUM_THREADS", "2")
import numpy as np
sys.path.insert(0, "cnla")
from multiprocessing import Pool


def job(i):
    import nlo_refine as R
    from nlo_core import RefMesh, plate_on_mesh
    from nlo_fvs import FVS, optimal_lb_fvs, to_res_fvs
    from verify_v2 import verify_v2
    d = np.load("cnla/data/nlo_val.npz", allow_pickle=True)
    xs, ys, t, sy, name = d["x_mm"][i], d["y_mm"][i], np.asarray(d["thickness_mm"][i], float), float(d["sigma_y_MPa"][i]), str(d["name"][i])
    mesh = RefMesh(256, 80)
    pl = plate_on_mesh(mesh, xs, ys, t)
    cap = R.tg_context(pl["nodes"], mesh.tri, xs, ys, t)["tv"]
    t0 = time.time()
    LB, Nv, status, sec = optimal_lb_fvs(mesh, pl, sy, FVS(mesh), cap=cap)
    v = np.zeros((len(pl["nodes"]), 2)); v[:, 0] = pl["nodes"][:, 0] / pl["L"]
    res = to_res_fvs(mesh, pl, v, Nv, sy)                                 # UB part only; LB at exact t_g yield (the
    from thickness_tg import tg_quantities
    from tg_capacity import tg_ratio                     # bilinear scaling of to_res_fvs lost 4.6 %)
    rr = tg_ratio(Nv * sy, tg_quantities(pl["nodes"], mesh.tri, xs, ys, t), sy)
    res.update(stress=Nv * sy / rr, lb=R.lb_v(Nv * sy, pl["W"]) / rr)
    _, lb2, errs, _ = verify_v2(res, xs, ys, t, sy)
    return dict(name=name, lb_floor=lb2, status=status, solve_s=sec, errs=";".join(errs))


if __name__ == "__main__":
    idx = [int(a) for a in sys.argv[1:]]
    with Pool(3) as p:
        rows = p.map(job, idx)
    ref = {r["name"]: r for r in csv.DictReader(open("cnla/data/nlo_val_ref_tg.csv"))}
    ev = {}
    for r in csv.DictReader(open("paper/figures/cache/val_histories_tg.csv")):
        if r["stage"] == "final" and r["bound"] == "lb":
            ev[r["plate"]] = float(r["value"])
    for r in rows:
        lr = float(ref[r["name"]]["LB_ref_N"])
        print(f"{r['name']}: LB deficit vs FE ref: floor {100 * (1 - r['lb_floor'] / lr):.2f} %, corrected method "
              f"{100 * (1 - ev[r['name']] / lr):.2f} %  ({r['status']}, {r['solve_s']:.0f} s) {r['errs']}", flush=True)
    with open("loops/scratch/tg_floor.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
