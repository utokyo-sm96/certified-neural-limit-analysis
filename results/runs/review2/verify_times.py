"""Verification time of the method's bounds under verifier v2 (time until a VERIFIED bound is available), per plate,
on the validation plates, the synthetic test plates and the real scans: the network's identity-mirror fields on the
canonical mesh (every event of the method lives on this mesh, so its verification costs the same), scaled to exact t_g
yield. Also a consistency check against the network events of the histories. Output: verify_times.csv."""
import csv
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import OUT, PF, network_fields, plate_sets, scaled_exact, v2_lb, v2_ub  # noqa: E402


def main():
    import nlo_refine as R
    import nlo_train_fvs
    from nlo_core import plate_on_mesh
    R.THICK_MODEL = "tg"
    model = nlo_train_fvs.load(os.path.join(PF, "data", "nlo_fvs_ft30k.pt"))
    rows = []
    for s, p in plate_sets():
        nodes, tri, vv, Nv = network_fields(model, p)
        th = plate_on_mesh(R.canon()["mesh"], p["xs"], p["ys"], p["t"])["thick"]
        lb, el, tl = v2_lb(nodes, tri, scaled_exact(Nv, nodes, tri, p), p["xs"], p["ys"], p["t"], p["sigma_y"], th)
        ub, eu, tu = v2_ub(nodes, tri, vv, p["xs"], p["ys"], p["t"], p["sigma_y"], th)
        rows.append(dict(set=s, plate=p["name"], net_lb=lb, net_ub=ub, t_verify_lb=tl, t_verify_ub=tu, errors=";".join(el + eu)))
        print(f"{s:5s} {p['name']}: net [{lb / 1e3:.2f}, {ub / 1e3:.2f}] kN  verify LB {tl:.2f} s  UB {tu:.2f} s  {rows[-1]['errors']}",
              flush=True)
    with open(os.path.join(OUT, "verify_times.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    for s in ("val", "test", "real"):
        S = [r for r in rows if r["set"] == s]
        print(f"{s}: verify LB mean {np.mean([r['t_verify_lb'] for r in S]):.2f} s (max {max(r['t_verify_lb'] for r in S):.2f}), "
              f"UB mean {np.mean([r['t_verify_ub'] for r in S]):.2f} s (max {max(r['t_verify_ub'] for r in S):.2f}), "
              f"errors {sum(bool(r['errors']) for r in S)}")


if __name__ == "__main__":
    main()
