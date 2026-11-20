#!/usr/bin/env python3
"""Frozen validation plates for the limit-analysis loop (synthetic only, fixed seed).

Six plates from corrosion_generator.py: for each bridge family, the synthetic plates closest
to the 10th, 50th and 90th percentile of mean thickness among 300 candidates. No real scan and
no test result is used, so the loop can never be tuned to the held-out experiments.

Usage: python make_la_validation_set.py --data ../corroded_steel_plates_data_en --out data/la_validation_plates.npz
"""
import argparse

import numpy as np

from corrosion_generator import calibrate, generate

SIGMA_Y = {"ananai": 281.6, "amarube": 269.4}      # coupon means, material_properties.csv


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=1106)
    a = ap.parse_args()
    plates = generate(calibrate(a.data), 300, seed=a.seed)
    chosen = []
    for fam in ("ananai", "amarube"):
        pool = [p for p in plates if p["family"] == fam]
        m = np.array([p["thickness"].mean() / p["t0"] for p in pool])
        for q in (10, 50, 90):
            chosen.append(pool[int(np.argmin(np.abs(m - np.percentile(m, q))))])

    def obj(key):
        o = np.empty(len(chosen), dtype=object)
        for i, p in enumerate(chosen):
            o[i] = np.asarray(p[key], dtype=np.float64)
        return o

    np.savez(a.out, x_mm=obj("xs"), y_mm=obj("ys"), thickness_mm=obj("thickness"),
             family=np.array([p["family"] for p in chosen]),
             t0_mm=np.array([p["t0"] for p in chosen]),
             sigma_y_MPa=np.array([SIGMA_Y[p["family"]] for p in chosen]),
             name=np.array([f"{p['family'][:2].upper()}-syn{i}" for i, p in enumerate(chosen)]))
    for p in chosen:
        print(f"{p['family']:8s} L={p['xs'][-1]:.0f} W={p['ys'][-1]:.0f} mean t/t0={p['thickness'].mean() / p['t0']:.3f}")
    print(f"-> {a.out}")


if __name__ == "__main__":
    main()
