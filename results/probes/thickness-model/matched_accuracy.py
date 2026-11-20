#!/usr/bin/env python3
"""Review R3: time to reach a certified gap threshold, method vs adaptive FE with a stopping criterion.
FE: loops/scratch/fe_levels_0_32.csv (levels 0-4, cumulative single-process time, verifier v2 on t_g).
Method: paper/figures/cache/val_histories[_tg].csv (reported best-so-far bracket, every event certified).
Usage: python loops/scratch/matched_accuracy.py [tg]"""
import csv
import os
import sys

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SUF = "_tg" if len(sys.argv) > 1 and sys.argv[1] == "tg" else ""


def method_times(th):
    ev = {}
    for r in csv.DictReader(open(os.path.join(ROOT, "paper", "figures", "cache", f"val_histories{SUF}.csv"))):
        ev.setdefault(r["plate"], []).append((float(r["t"]), r["stage"], r["bound"], float(r["value"])))
    out = {}
    for p, E in ev.items():
        E.sort(key=lambda e: e[0])
        lb, ub, first = -np.inf, np.inf, np.inf
        for t, st, bd, v in E:
            if st in ("network", "admm_scaled"):
                continue
            lb, ub = (max(lb, v), ub) if bd == "lb" else (lb, min(ub, v))
            if np.isfinite(lb) and np.isfinite(ub) and (ub - lb) / ub <= th and not np.isfinite(first):
                first = t
        out[p] = first
    return out


def fe_times(th):
    R = list(csv.DictReader(open(os.path.join(ROOT, "loops", "scratch", os.environ.get("FE_CSV", "fe_levels_0_32.csv")))))
    out = {}
    for p in sorted(set(r["plate"] for r in R)):
        rr = sorted([r for r in R if r["plate"] == p], key=lambda r: int(r["level"]))
        out[p] = next((float(r["t_cum"]) for r in rr if float(r["gap"]) <= th), np.inf)
    return out


def main():
    print(f"method histories: val_histories{SUF}.csv")
    for th in (0.05, 0.02, 0.01):
        m, f = method_times(th), fe_times(th)
        common = sorted(set(m) & set(f))
        mt = np.array([m[p] for p in common])
        ft = np.array([f[p] for p in common])
        both = np.isfinite(mt) & np.isfinite(ft)
        ratio = ft[both] / mt[both]
        print(f"gap <= {100 * th:.0f} % on {len(common)} plates: method reached {np.isfinite(mt).sum()} (median "
              f"{np.median(mt[np.isfinite(mt)]) if np.isfinite(mt).any() else np.nan:.1f} s), FE reached {np.isfinite(ft).sum()} "
              f"(median {np.median(ft[np.isfinite(ft)]) if np.isfinite(ft).any() else np.nan:.1f} s); both: {both.sum()}, "
              f"FE/method time ratio median {np.median(ratio) if both.any() else np.nan:.1f} (range "
              f"{ratio.min() if both.any() else np.nan:.1f}-{ratio.max() if both.any() else np.nan:.1f})")


if __name__ == "__main__":
    main()
