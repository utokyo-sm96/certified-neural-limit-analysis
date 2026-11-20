"""Network removal under the final t_g formulation (review 2, point 4): the final refinement (nlo_refine THICK_MODEL "tg",
loop tg-method-1112 configuration) started from trivial admissible fields instead of the network's: velocity
v = (x/L, 0) and uniform axial stress, on all four mirror images. Validation plates only; every bracket certified by
verifier v2. Output: coldstart_tg.csv."""
import csv
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import OUT, PF, plate_sets, v2_lb, v2_ub  # noqa: E402


def main():
    import nlo_refine as R
    import nlo_train_fvs as NF
    from nlo_core import plate_on_mesh
    from nlo_fvs import to_res_fvs
    R.THICK_MODEL = "tg"
    model = NF.load(os.path.join(PF, "data", "nlo_fvs_ft30k.pt"))
    real_predict = NF.predict

    def trivial_predict(model_, plates, batch=16):
        real_predict(model_, plates[:1])                                 # builds the cached mesh
        mesh = NF._C["mesh"]
        out = []
        for p in plates:
            pl = plate_on_mesh(mesh, p["xs"], p["ys"], p["t"])
            v = np.zeros((len(pl["nodes"]), 2))
            v[:, 0] = pl["nodes"][:, 0] / pl["L"]
            Nv = np.zeros((mesh.T, 3, 3))
            Nv[..., 0] = 1.0
            out.append(to_res_fvs(mesh, pl, v, Nv, p["sigma_y"]))
        return out
    NF.predict = trivial_predict
    real_combine_ub = R.combine_ub

    def combine_ub_safe(vs, *a, **k):                                    # four identical mirror fields: the best
        try:                                                             # combination is the field itself
            return real_combine_ub(vs, *a, **k)
        except np.linalg.LinAlgError:
            return np.asarray(vs[0], float).copy()
    R.combine_ub = combine_ub_safe
    rows = []
    P = plate_sets(("val",))
    R.bounds(model, dict((k, P[0][1][k]) for k in ("xs", "ys", "t", "sigma_y")))   # warm-up
    for s, p in P:
        pp = dict((k, p[k]) for k in ("xs", "ys", "t", "sigma_y"))
        t0 = time.time()
        o = R.bounds(model, pp)
        dt = time.time() - t0
        u, l = o["ub_res"], o["lb_res"]
        ub, eu, _ = v2_ub(np.asarray(u["nodes"], float), np.asarray(u["tri"], int), np.asarray(u["v"], float), p["xs"], p["ys"],
                          p["t"], p["sigma_y"], np.asarray(u["thick"], float))
        lb, el, _ = v2_lb(np.asarray(l["nodes"], float), np.asarray(l["tri"], int), np.asarray(l["stress"], float), p["xs"],
                          p["ys"], p["t"], p["sigma_y"], np.asarray(l["thick"], float))
        rows.append(dict(plate=p["name"], lb=lb, ub=ub, gap=(ub - lb) / ub, seconds=dt, errors=";".join(el + eu)))
        print(f"{p['name']}: [{lb / 1e3:.2f}, {ub / 1e3:.2f}] kN  gap {100 * rows[-1]['gap']:.2f} %  {dt:.1f} s  {rows[-1]['errors']}",
              flush=True)
    with open(os.path.join(OUT, "coldstart_tg.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"done: cold start (trivial fields) mean gap {100 * np.mean([r['gap'] for r in rows]):.3f} % (median "
          f"{100 * np.median([r['gap'] for r in rows]):.2f}), time {np.mean([r['seconds'] for r in rows]):.1f} s, errors "
          f"{sum(bool(r['errors']) for r in rows)}")


if __name__ == "__main__":
    main()
