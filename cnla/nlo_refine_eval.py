#!/usr/bin/env python3
"""FROZEN evaluator for loop nlo-refine-1107: certified collapse-load bracket at a fixed time budget.

The network is frozen (nlo_train.py and data/nlo_ckpt_final.pt from loop nlo-1106). The loop edits only
nlo_refine.py, which turns the network output for one plate into certified bounds:
    nlo_refine.bounds(model, plate) -> dict(ub_res=dict(nodes, tri, thick, v, ub),
                                            lb_res=dict(nodes, tri, thick, stress, lb))
Plates are processed one at a time, in this process, and each call is timed (network inference
included). The UB and the LB may live on different meshes. Each is certified separately by the
frozen step-1 verifier eval_limit_analysis.verify, with a trivially admissible partner field built here:
  * UB: the candidate velocities plus a uniform uniaxial stress Nxx = 1 N/mm (equilibrated, free edges
    traction-free, far below yield), so verify re-computes the dissipation independently;
  * LB: the candidate stresses plus the velocity v = (x/L, 0), whose dissipation is computed here in
    closed form; verify re-checks equilibrium, continuity, free edges, yield and the grip resultant.

Gates (any failure means discard):
  G1  uniform strip 236 x 76 x 10.5 mm: certified; UB >= sigma_y W t (1 - 1e-6); LB <= step-1 FE UB (1 + 1e-6)
  G2  every validation plate: UB and LB certified (no verifier errors)
  G3  weak duality against the step-1 FE references: LB <= UB_ref (1 + 1e-6), UB >= LB_ref (1 - 1e-6)
  G4  time: mean <= 5 s and max <= 10 s per plate (step-1 FE needs 45-60 s per plate)
  G5  no element edge > 2.0 mm in either mesh (no coarsening below the scan grid)
Metric: gap_mean = mean over the 32 validation plates of (UB - LB) / UB (lower is better).
"""
import csv
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from eval_limit_analysis import verify  # noqa: E402  (frozen step-1 verifier)

CKPT_FINAL = os.path.join(HERE, "data", "nlo_ckpt_final.pt")
VAL = os.path.join(HERE, "data", "nlo_val.npz")
REF = os.path.join(HERE, "data", "nlo_val_ref.csv")
TIME_MEAN, TIME_MAX, MAX_EDGE_MM = 5.0, 10.0, 2.0
SQ3 = np.sqrt(3.0)


def max_edge(res):
    p = np.asarray(res["nodes"])[np.asarray(res["tri"])]
    return float(max(np.linalg.norm(p[:, a] - p[:, b], axis=1).max() for a, b in ((0, 1), (1, 2), (2, 0))))


def tri_area(nodes, tri):
    p = nodes[tri]
    return 0.5 * ((p[:, 1, 0] - p[:, 0, 0]) * (p[:, 2, 1] - p[:, 0, 1]) - (p[:, 1, 1] - p[:, 0, 1]) * (p[:, 2, 0] - p[:, 0, 0]))


def certify_ub(res, xs, ys, t, sy):
    nodes, tri = np.asarray(res["nodes"], float), np.asarray(res["tri"], int)
    S = np.zeros((len(tri), 3, 3))
    S[:, :, 0] = 1.0                                                     # uniform uniaxial partner field
    r = dict(nodes=nodes, tri=tri, thick=np.asarray(res["thick"], float), v=np.asarray(res["v"], float),
             ub=float(res["ub"]), stress=S, lb=0.0, status_ub="Solved", status_lb="Solved")
    ub, _, errs, _ = verify(r, xs, ys, t, sy, "rigid")
    return ub, errs


def certify_lb(res, xs, ys, t, sy):
    nodes, tri = np.asarray(res["nodes"], float), np.asarray(res["tri"], int)
    th = np.asarray(res["thick"], float)
    L = xs[-1]
    v = np.stack([nodes[:, 0] / L, np.zeros(len(nodes))], 1)            # admissible partner velocity
    ub_dummy = float(np.sum(sy * (2 / SQ3) * (1 / L) * tri_area(nodes, tri) * th[tri].mean(1)))
    r = dict(nodes=nodes, tri=tri, thick=th, v=v, ub=ub_dummy, stress=np.asarray(res["stress"], float),
             lb=float(res["lb"]), status_ub="Solved", status_lb="Solved")
    _, lb, errs, _ = verify(r, xs, ys, t, sy, "rigid")
    return lb, errs


def main():
    t_start = time.time()
    gate_errs = []
    import nlo_train
    import nlo_refine
    model = nlo_train.load(CKPT_FINAL)

    # ---- G1 (also the untimed warm-up)
    from limit_analysis import solve_bounds
    sy, t0 = 281.6, 10.5
    xs, ys = np.arange(0, 236, 2.0), np.arange(0, 76, 2.0)
    tu = np.full((len(xs), len(ys)), t0)
    ub_fe, _, _, _ = verify(solve_bounds(xs, ys, tu, sy, grips="rigid"), xs, ys, tu, sy, "rigid")
    out = nlo_refine.bounds(model, dict(xs=xs, ys=ys, t=tu, sigma_y=sy))
    ub, eu = certify_ub(out["ub_res"], xs, ys, tu, sy)
    lb, el = certify_lb(out["lb_res"], xs, ys, tu, sy)
    ex = sy * ys[-1] * t0
    print(f"G1 uniform strip: UB/(sy W t)={ub / ex:.5f}  LB/(sy W t)={lb / ex:.5f}  FE UB={ub_fe / ex:.5f}  {eu + el}")
    if eu or el or ub < ex * (1 - 1e-6) or lb > ub_fe * (1 + 1e-6):
        gate_errs.append("G1")

    # ---- validation plates, one at a time, timed
    d = np.load(VAL, allow_pickle=True)
    refs = {row["name"]: row for row in csv.DictReader(open(REF))}
    gaps, times, ub_ex, lb_def = [], [], [], []
    for i in range(len(d["name"])):
        name = str(d["name"][i])
        xs, ys, t, sy = d["x_mm"][i], d["y_mm"][i], d["thickness_mm"][i], float(d["sigma_y_MPa"][i])
        t1 = time.time()
        out = nlo_refine.bounds(model, dict(xs=xs, ys=ys, t=t, sigma_y=sy))
        dt = time.time() - t1
        ub, eu = certify_ub(out["ub_res"], xs, ys, t, sy)
        lb, el = certify_lb(out["lb_res"], xs, ys, t, sy)
        rf = refs[name]
        ub_ref, lb_ref = float(rf["UB_ref_N"]), float(rf["LB_ref_N"])
        gap = (ub - lb) / ub
        gaps.append(gap)
        times.append(dt)
        ub_ex.append(ub / ub_ref - 1)
        lb_def.append(1 - lb / lb_ref)
        if eu or el:
            gate_errs.append(f"G2 {name}")
        if lb > ub_ref * (1 + 1e-6) or ub < lb_ref * (1 - 1e-6) or lb > ub * (1 + 1e-6):
            gate_errs.append(f"G3 {name}")
        if max(max_edge(out["ub_res"]), max_edge(out["lb_res"])) > MAX_EDGE_MM:
            gate_errs.append(f"G5 {name}")
        print(f"{name}  UB {ub / 1e3:7.2f}  LB {lb / 1e3:7.2f} kN  gap {100 * gap:6.2f}%  "
              f"(ref [{lb_ref / 1e3:.2f}, {ub_ref / 1e3:.2f}]: UB +{100 * ub_ex[-1]:.2f}%, LB -{100 * lb_def[-1]:.2f}%)  "
              f"{dt:5.2f}s  {(eu + el) if (eu or el) else ''}", flush=True)
    if np.mean(times) > TIME_MEAN or np.max(times) > TIME_MAX:
        gate_errs.append(f"G4 time mean {np.mean(times):.2f}s max {np.max(times):.2f}s")
    print("---")
    print(f"gap_mean: {np.mean(gaps):.6f}")
    print("gates: " + ("PASS" if not gate_errs else "FAIL " + ",".join(gate_errs[:6])))
    print(f"runtime_s: {time.time() - t_start:.1f}")
    print(f"gap_median: {np.median(gaps):.6f}")
    print(f"gap_max: {np.max(gaps):.6f}")
    print(f"ub_excess_mean: {np.mean(ub_ex):.6f}")
    print(f"lb_deficit_mean: {np.mean(lb_def):.6f}")
    print(f"time_mean_s: {np.mean(times):.3f}")
    print(f"time_max_s: {np.max(times):.3f}")


if __name__ == "__main__":
    main()
