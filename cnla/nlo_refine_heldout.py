#!/usr/bin/env python3
"""One-shot held-out evaluation of the winning hybrid method of loop nlo-refine-1107
(frozen network -> certified mirror combination -> local LB/UB cone-program corrections).

Same certification and timing as the frozen loop evaluator (nlo_refine_eval.certify_ub / certify_lb).
FE references: data/nlo_test_ref.csv for the 64 synthetic test plates, and the step-1 FE brackets of the
real scans from loops/nlo-1106/heldout_nlo.csv (same solver, same settings), plus the measured yield loads.

Usage: python nlo_refine_heldout.py --data ../corroded_steel_plates_data_en --out ../loops/nlo-refine-1107
"""
import argparse
import csv
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    import nlo_refine
    import nlo_train
    from nlo_refine_eval import CKPT_FINAL, certify_lb, certify_ub
    from scans import load_surface, read_table, surface_to_grid
    model = nlo_train.load(CKPT_FINAL)
    plates = []
    d = np.load(os.path.join(HERE, "data", "nlo_test.npz"), allow_pickle=True)
    refs = {r["name"]: r for r in csv.DictReader(open(os.path.join(HERE, "data", "nlo_test_ref.csv")))}
    for i in range(len(d["name"])):
        rf = refs[str(d["name"][i])]
        plates.append(dict(set="synthetic-test", name=str(d["name"][i]), xs=d["x_mm"][i], ys=d["y_mm"][i],
                           t=d["thickness_mm"][i], sigma_y=float(d["sigma_y_MPa"][i]),
                           FE_LB=float(rf["LB_ref_N"]), FE_UB=float(rf["UB_ref_N"]), meas=""))
    prev = {r["name"]: r for r in csv.DictReader(open(os.path.join(HERE, "..", "loops", "nlo-1106", "heldout_nlo.csv")))
            if r["set"] == "real-scan"}
    tests = {r["specimen_id"]: r for r in read_table(os.path.join(a.data, "tensile_tests.csv"))}
    sig = {"ananai": 281.6, "amarube": 269.4}
    for fam in ("ananai", "amarube"):
        folder = os.path.join(a.data, "surfaces", fam)
        for fn in sorted(os.listdir(folder)):
            sid = fn[:-4]
            xs, ys, g = surface_to_grid(load_surface(os.path.join(folder, fn)))
            plates.append(dict(set="real-scan", name=sid, xs=xs, ys=ys, t=np.clip(g["thickness_mm"], 0.05, None),
                               sigma_y=sig[fam], FE_LB=1e3 * float(prev[sid]["FE_LB_kN"]),
                               FE_UB=1e3 * float(prev[sid]["FE_UB_kN"]), meas=tests.get(sid, {}).get("yield_load_kN", "")))
    nlo_refine.bounds(model, plates[0])                                  # warm-up (untimed)
    rows = []
    for p in plates:
        t1 = time.time()
        out = nlo_refine.bounds(model, dict(xs=p["xs"], ys=p["ys"], t=p["t"], sigma_y=p["sigma_y"]))
        dt = time.time() - t1
        ub, eu = certify_ub(out["ub_res"], p["xs"], p["ys"], p["t"], p["sigma_y"])
        lb, el = certify_lb(out["lb_res"], p["xs"], p["ys"], p["t"], p["sigma_y"])
        rows.append(dict(set=p["set"], name=p["name"], LB_kN=lb / 1e3, UB_kN=ub / 1e3, gap=(ub - lb) / ub,
                         certified="yes" if not (eu or el) else ";".join(eu + el),
                         dual_ok=lb <= p["FE_UB"] * (1 + 1e-6) and ub >= p["FE_LB"] * (1 - 1e-6),
                         FE_LB_kN=p["FE_LB"] / 1e3, FE_UB_kN=p["FE_UB"] / 1e3, seconds=dt, measured_yield_kN=p["meas"]))
        print(f"{p['name']}: [{lb / 1e3:.1f}, {ub / 1e3:.1f}] kN ({100 * rows[-1]['gap']:.1f}%)  FE [{p['FE_LB'] / 1e3:.1f}, "
              f"{p['FE_UB'] / 1e3:.1f}]  {dt:.1f}s  {rows[-1]['certified']}", flush=True)
    out = os.path.join(a.out, "heldout_refine.csv")
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        for r in rows:
            w.writerow({k: (f"{v:.6g}" if isinstance(v, float) else v) for k, v in r.items()})
    print("---")
    for s in ("synthetic-test", "real-scan"):
        R = [r for r in rows if r["set"] == s]
        g = np.array([r["gap"] for r in R])
        print(f"{s}: n={len(R)} certified={all(r['certified'] == 'yes' for r in R)} consistent-with-FE={all(r['dual_ok'] for r in R)}  "
              f"gap mean {100 * g.mean():.2f}% (median {100 * np.median(g):.2f}%, max {100 * g.max():.1f}%)  "
              f"UB +{100 * np.mean([r['UB_kN'] / r['FE_UB_kN'] - 1 for r in R]):.2f}%  "
              f"LB -{100 * np.mean([1 - r['LB_kN'] / r['FE_LB_kN'] for r in R]):.2f}%  "
              f"time {np.mean([r['seconds'] for r in R]):.2f} s (max {np.max([r['seconds'] for r in R]):.1f})")
    T = [r for r in rows if r["measured_yield_kN"] not in ("", None)]
    m = np.array([float(r["measured_yield_kN"]) for r in T])
    lo, hi = np.array([r["LB_kN"] for r in T]), np.array([r["UB_kN"] for r in T])
    q = m / (0.5 * (lo + hi))
    print(f"tensile tests (n={len(T)}): measured / mid-bracket = {q.mean():.3f} +- {q.std(ddof=1):.3f}, "
          f"MAPE {100 * np.mean(np.abs(q - 1)):.1f}%, inside bracket {int(np.sum((m >= lo) & (m <= hi)))}/{len(T)}")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
