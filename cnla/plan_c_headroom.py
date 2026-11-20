#!/usr/bin/env python3
"""Plan C headroom: per-plate split of the hybrid's certified gap into the FVS mesh floor (unavoidable on this mesh)
and the optimisation error (what a global solver could remove). Inputs: results/runs/pareto/{pareto_val,fvs_floor_val}.csv.
Prints a summary and writes results/runs/pareto/plan_c_headroom.csv."""
import csv
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RUNS = os.path.join(HERE, "..", "docs", "runs", "pareto")


def main():
    par = list(csv.DictReader(open(os.path.join(RUNS, "pareto_val.csv"))))
    fl = {r["name"]: r for r in csv.DictReader(open(os.path.join(RUNS, "fvs_floor_val.csv")))}
    g = {(r["level"], r["name"]): 100 * float(r["gap"]) for r in par}
    names = sorted(fl)
    out = []
    for n in names:
        f = fl[n]
        out.append(dict(name=n, hybrid_b3=g[("b3", n)], hybrid_b6=g[("b6", n)], floor_fvs=100 * float(f["gap_fvs"]),
                        floor_zp=100 * float(f["gap_zp"]), fe=100 * float(f["gap_ref"]),
                        opt_err_b3=g[("b3", n)] - 100 * float(f["gap_fvs"]), opt_err_b6=g[("b6", n)] - 100 * float(f["gap_fvs"])))
    with open(os.path.join(RUNS, "plan_c_headroom.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(out[0]))
        w.writeheader()
        w.writerows(out)
    a = lambda k: np.array([r[k] for r in out])
    print(f"mean: hybrid b3 {a('hybrid_b3').mean():.3f}  b6 {a('hybrid_b6').mean():.3f}  FVS floor {a('floor_fvs').mean():.3f}  "
          f"ZP floor {a('floor_zp').mean():.3f}  FE {a('fe').mean():.3f}")
    print(f"median: hybrid b3 {np.median(a('hybrid_b3')):.3f}  FVS floor {np.median(a('floor_fvs')):.3f}  FE {np.median(a('fe')):.3f}")
    e = a("opt_err_b3")
    o = np.argsort(-e)
    print(f"optimisation error at b3: mean {e.mean():.3f} points; top-5 plates carry {e[o[:5]].sum() / e.sum() * 100:.0f} % of it; "
          f"plates with error > 2 points: {(e > 2).sum()}/32")
    for i in o[:6]:
        r = out[i]
        print(f"  {r['name']}: hybrid {r['hybrid_b3']:.2f} (b6 {r['hybrid_b6']:.2f})  floor FVS {r['floor_fvs']:.2f}  ZP {r['floor_zp']:.2f}  FE {r['fe']:.2f}")
    for frac in (0.5, 0.75, 1.0):
        print(f"if a global solver removed {int(frac * 100)} % of the optimisation error: mean gap {a('hybrid_b3').mean() - frac * e.mean():.2f} %")


if __name__ == "__main__":
    main()
