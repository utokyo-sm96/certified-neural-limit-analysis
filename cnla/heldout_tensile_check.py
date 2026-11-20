#!/usr/bin/env python3
"""One-shot held-out check: certified limit-analysis bounds vs the 18 measured Ananai tensile yield loads.

Runs limit_analysis.solve_bounds on each scanned Ananai plate that has a tensile test (rigid welded
grips, sigma_y = 281.6 MPa, the Ananai coupon mean), certifies both bounds with the frozen evaluator's
independent verify(), and compares them with
  * the measured yield load (Chun et al. 2013, tensile_tests.csv), and
  * the net-section rule  sigma_y * W * t_minavg  (t_minavg = min over x-sections of the width-averaged t).
This data is never used inside an optimisation loop.

Usage: python heldout_tensile_check.py --data ../corroded_steel_plates_data_en --out ../loops/la-solver-1106
"""
import argparse
import csv
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from eval_limit_analysis import verify  # noqa: E402  (frozen independent verifier)
from limit_analysis import solve_bounds  # noqa: E402
from scans import load_surface, read_table, surface_to_grid  # noqa: E402

SIGMA_Y = 281.6   # MPa, Ananai coupon mean (material_properties.csv)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    tests = {r["specimen_id"]: r for r in read_table(os.path.join(a.data, "tensile_tests.csv"))}
    rows = []
    for sid in sorted(tests):
        path = os.path.join(a.data, "surfaces", "ananai", f"{sid}.csv")
        if not sid.startswith("AN-") or not os.path.exists(path):
            continue
        xs, ys, g = surface_to_grid(load_surface(path))
        t = np.clip(g["thickness_mm"], 0.05, None)
        t0 = time.time()
        res = solve_bounds(xs, ys, t, SIGMA_Y, grips="rigid")
        ub, lb, errs, info = verify(res, xs, ys, t, SIGMA_Y, "rigid")
        W = ys[-1]
        t_minavg = t.mean(axis=1).min()
        row = dict(specimen_id=sid, width_mm=W, mean_t_mm=t.mean(), min_avg_t_mm=t_minavg,
                   LB_kN=lb / 1e3, UB_kN=ub / 1e3, gap_pct=100 * (ub - lb) / ub,
                   net_section_kN=SIGMA_Y * W * t_minavg / 1e3,
                   measured_yield_kN=float(tests[sid]["yield_load_kN"]),
                   measured_max_kN=float(tests[sid]["max_tensile_load_kN"]),
                   verified="yes" if not errs else "NO: " + "; ".join(errs), seconds=time.time() - t0)
        rows.append(row)
        print(f"{sid}: LB {row['LB_kN']:7.1f}  UB {row['UB_kN']:7.1f}  (gap {row['gap_pct']:.2f}%)  "
              f"net-section {row['net_section_kN']:7.1f}  measured {row['measured_yield_kN']:7.1f} kN  "
              f"{row['verified']}  {row['seconds']:.0f}s", flush=True)

    with open(os.path.join(a.out, "heldout_tensile.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        for r in rows:
            w.writerow({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()})

    m = np.array([r["measured_yield_kN"] for r in rows])
    lb = np.array([r["LB_kN"] for r in rows])
    ub = np.array([r["UB_kN"] for r in rows])
    ns = np.array([r["net_section_kN"] for r in rows])
    mid = 0.5 * (lb + ub)

    def r2(pred):
        return 1 - np.sum((m - pred) ** 2) / np.sum((m - m.mean()) ** 2)

    print("---")
    print(f"plates: {len(rows)}, all certified: {all(r['verified'] == 'yes' for r in rows)}")
    print(f"measured / limit-analysis (mid of bracket): mean {np.mean(m / mid):.3f}  SD {np.std(m / mid, ddof=1):.3f}  "
          f"min {np.min(m / mid):.2f}  max {np.max(m / mid):.2f}  R2 {r2(mid):.3f}  MAPE {100 * np.mean(np.abs(m / mid - 1)):.1f}%")
    print(f"measured / net-section rule:               mean {np.mean(m / ns):.3f}  SD {np.std(m / ns, ddof=1):.3f}  "
          f"min {np.min(m / ns):.2f}  max {np.max(m / ns):.2f}  R2 {r2(ns):.3f}  MAPE {100 * np.mean(np.abs(m / ns - 1)):.1f}%")
    print(f"limit analysis / net-section:              mean {np.mean(mid / ns):.3f}  min {np.min(mid / ns):.3f}  max {np.max(mid / ns):.3f}")
    print(f"measured inside the certified bracket: {int(np.sum((m >= lb) & (m <= ub)))} / {len(rows)}; "
          f"mean bracket gap {np.mean(100 * (ub - lb) / ub):.2f}%")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ink2, grid_c, blue, orange = "#52514e", "#e4e3df", "#2a78d6", "#eb6834"
    fig, ax = plt.subplots(figsize=(5.6, 5.2), constrained_layout=True)
    lo, hi = 90, 240
    ax.plot([lo, hi], [lo, hi], color=ink2, lw=0.9, ls="--", zorder=1)
    ax.text(hi - 3, hi - 10, "prediction = measurement", color=ink2, ha="right", fontsize=8)
    ax.errorbar(m, mid, yerr=[mid - lb, ub - mid], fmt="o", ms=6.5, color=blue, mec="white", mew=1.2,
                ecolor=blue, elinewidth=1.5, capsize=0, zorder=3, label="limit analysis, certified bracket")
    ax.scatter(m, ns, marker="^", s=46, color=orange, edgecolor="white", linewidth=1.2, zorder=2,
               label="net-section rule  sigma_y W t_minavg")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel("measured yield load, kN (18 Ananai tensile tests)")
    ax.set_ylabel("predicted yield load, kN")
    ax.grid(color=grid_c, lw=0.6)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.legend(frameon=False, loc="upper left", fontsize=8.5)
    ax.set_title("Held-out check: limit analysis on the scanned plates", loc="left", fontsize=10)
    fig.savefig(os.path.join(a.out, "heldout_tensile.png"), dpi=150)
    print(f"wrote {os.path.join(a.out, 'heldout_tensile.csv')} and heldout_tensile.png")


if __name__ == "__main__":
    main()
