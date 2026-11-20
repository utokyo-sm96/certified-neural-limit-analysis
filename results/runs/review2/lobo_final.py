"""Leave-one-bridge-out with the FINAL method (review 2): the three 3e4-step networks of results/runs/lobo (trained on
Ananai-family plates only, Amarube-family only, both = control) on the 28 real scans, network alone (identity mirror,
exact t_g scaling) and the final refinement (nlo_refine THICK_MODEL "tg"), every bracket certified by verifier v2.
Output: lobo_final.csv."""
import csv
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import OUT, PF, network_fields, plate_sets, scaled_exact, v2_lb, v2_ub  # noqa: E402

NETS = {"ananai_only": "nlo_lobo_ananai30k.pt", "amarube_only": "nlo_lobo_amarube30k.pt", "both_control": "nlo_ckpt_final.pt"}


def main():
    import nlo_refine as R
    import nlo_train
    from nlo_core import plate_on_mesh
    R.THICK_MODEL = "tg"
    P = plate_sets(("real",))
    rows = []
    for tag, fn in NETS.items():
        model = nlo_train.load(os.path.join(PF, "data", fn))
        R.bounds(model, dict((k, P[0][1][k]) for k in ("xs", "ys", "t", "sigma_y")))   # warm-up
        for s, p in P:
            pp = dict((k, p[k]) for k in ("xs", "ys", "t", "sigma_y"))
            nodes, tri, vv, Nv = network_fields(model, p, fvs=False)
            th = plate_on_mesh(R.canon()["mesh"], p["xs"], p["ys"], p["t"])["thick"]
            nlb, e1, _ = v2_lb(nodes, tri, scaled_exact(Nv, nodes, tri, p), p["xs"], p["ys"], p["t"], p["sigma_y"], th)
            nub, e2, _ = v2_ub(nodes, tri, vv, p["xs"], p["ys"], p["t"], p["sigma_y"], th)
            t0 = time.time()
            o = R.bounds(model, pp)
            dt = time.time() - t0
            u, l = o["ub_res"], o["lb_res"]
            fub, e3, _ = v2_ub(np.asarray(u["nodes"], float), np.asarray(u["tri"], int), np.asarray(u["v"], float), p["xs"],
                               p["ys"], p["t"], p["sigma_y"], np.asarray(u["thick"], float))
            flb, e4, _ = v2_lb(np.asarray(l["nodes"], float), np.asarray(l["tri"], int), np.asarray(l["stress"], float), p["xs"],
                               p["ys"], p["t"], p["sigma_y"], np.asarray(l["thick"], float))
            fam = "ananai" if p["name"].startswith("AN") else "amarube"
            rows.append(dict(net=tag, family=fam, plate=p["name"], net_lb=nlb, net_ub=nub, net_gap=(nub - nlb) / nub,
                             full_lb=flb, full_ub=fub, full_gap=(fub - flb) / fub, seconds=dt, errors=";".join(e1 + e2 + e3 + e4)))
            print(f"{tag:13s} {p['name']}: network {100 * rows[-1]['net_gap']:6.2f} %  full {100 * rows[-1]['full_gap']:5.2f} %  "
                  f"{dt:5.1f} s  {rows[-1]['errors']}", flush=True)
            with open(os.path.join(OUT, "lobo_final.csv"), "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(rows[0]))
                w.writeheader()
                w.writerows(rows)
    for fam in ("ananai", "amarube"):
        for tag in NETS:
            S = [r for r in rows if r["family"] == fam and r["net"] == tag]
            print(f"{fam:8s} {tag:13s}: network {100 * np.mean([r['net_gap'] for r in S]):.2f} %  full "
                  f"{100 * np.mean([r['full_gap'] for r in S]):.2f} % (max {100 * max(r['full_gap'] for r in S):.2f})")
    print(f"done: {len(rows)} rows, errors {sum(bool(r['errors']) for r in rows)}")


if __name__ == "__main__":
    main()
