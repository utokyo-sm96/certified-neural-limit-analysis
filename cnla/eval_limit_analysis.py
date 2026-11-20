#!/usr/bin/env python3
"""FROZEN evaluator for the limit-analysis loop. Do not edit inside a loop.

It calls limit_analysis.solve_bounds() and re-verifies every answer with its own independent code,
so a solver bug cannot pass as an improvement:
  * mesh: CCW triangles, conforming edges, boundary on the rectangle, area = L * W;
  * thickness: equals bilinear interpolation of the input grid at the mesh nodes;
  * upper bound: the velocity field meets the grip conditions, and its dissipation is recomputed
    here (UB_verified);
  * lower bound: equilibrium, traction continuity and free-edge tractions are recomputed here; the
    right-grip resultant is recomputed; the field is scaled down by the worst yield ratio, so
    LB_certified is statically and plastically admissible.

Gates (all must pass):
  G1  uniform strip, smooth grips: UB and LB within 1e-3 of the exact sigma_y * W * t
  G2  uniform strip, rigid grips: LB >= sigma_y * W * t * (1 - 1e-3) and LB <= UB
  G3  every validation plate: solver status Solved or AlmostSolved, mesh/thickness checks, |UB_solver/UB_verified - 1| < 1e-5,
      equilibrium / continuity / boundary residuals < 1e-6 (relative), yield ratio < 1 + 1e-4
  G4  weak duality: LB_certified <= UB_verified * (1 + 1e-6) on every plate
  G5  total runtime < 420 s
Metric: gap_mean = mean over plates of (UB_verified - LB_certified) / UB_verified  (lower is better).
"""
import os
import sys
import time

import numpy as np
from scipy.interpolate import RegularGridInterpolator

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from limit_analysis import solve_bounds  # noqa: E402

VAL = os.path.join(HERE, "data", "la_validation_plates.npz")
RUNTIME_LIMIT = 420.0


# ----------------------------------------------------------------------------- independent checks
def check_mesh(nodes, tri, L, W):
    p = nodes[tri]
    area = 0.5 * ((p[:, 1, 0] - p[:, 0, 0]) * (p[:, 2, 1] - p[:, 0, 1])
                  - (p[:, 1, 1] - p[:, 0, 1]) * (p[:, 2, 0] - p[:, 0, 0]))
    errs = []
    if area.min() <= 0:
        errs.append("non-CCW or degenerate triangle")
    if abs(area.sum() / (L * W) - 1) > 1e-9:
        errs.append(f"mesh area {area.sum():.6g} != L*W {L * W:.6g}")
    if nodes[:, 0].min() < -1e-9 or nodes[:, 0].max() > L + 1e-9 or nodes[:, 1].min() < -1e-9 or nodes[:, 1].max() > W + 1e-9:
        errs.append("node outside the plate")
    e = np.concatenate([tri[:, [0, 1]], tri[:, [1, 2]], tri[:, [2, 0]]])
    key = np.sort(e, axis=1)
    _, cnt = np.unique(key, axis=0, return_counts=True)
    if cnt.max() > 2:
        errs.append("edge shared by more than two triangles")
    uniq, cnt = np.unique(key, axis=0, return_counts=True)
    bnd = uniq[cnt == 1]
    m = 0.5 * (nodes[bnd[:, 0]] + nodes[bnd[:, 1]])
    tol = 1e-7 * max(L, W)
    on = (np.abs(m[:, 0]) < tol) | (np.abs(m[:, 0] - L) < tol) | (np.abs(m[:, 1]) < tol) | (np.abs(m[:, 1] - W) < tol)
    if not on.all():
        errs.append("boundary edge inside the plate (non-conforming mesh)")
    return area, errs


def verify(res, xs, ys, t, sy, grips):
    nodes, tri = np.asarray(res["nodes"], float), np.asarray(res["tri"], int)
    L, W = xs[-1], ys[-1]
    area, errs = check_mesh(nodes, tri, L, W)
    th = RegularGridInterpolator((xs, ys), t)(np.clip(nodes, [0, 0], [L, W]))
    if np.max(np.abs(np.asarray(res["thick"]) - th)) > 1e-6 * t.max():
        errs.append("thickness differs from bilinear interpolation of the grid")
    # Optimality is not required: both bounds are re-verified below, and a sub-optimal bound only
    # widens the gap (which the metric penalises). Hard failures (infeasible, max iterations) are not allowed.
    for k in ("status_ub", "status_lb"):
        if res[k] not in ("Solved", "AlmostSolved"):
            errs.append(f"{k}={res[k]}")
    tol = 1e-7 * max(L, W)
    left, right = np.abs(nodes[:, 0]) < tol, np.abs(nodes[:, 0] - L) < tol

    # ---- upper bound: grip conditions and dissipation recomputed here
    v = np.asarray(res["v"], float)
    vt = 1e-6
    if np.abs(v[left, 0]).max() > vt or np.abs(v[right, 0] - 1).max() > vt:
        errs.append("velocity violates grip condition (x)")
    if grips == "rigid" and (np.abs(v[left, 1]).max() > vt or np.abs(v[right, 1]).max() > vt):
        errs.append("velocity violates rigid grip condition (y)")
    p = nodes[tri]
    bx = np.stack([p[:, 1, 1] - p[:, 2, 1], p[:, 2, 1] - p[:, 0, 1], p[:, 0, 1] - p[:, 1, 1]], 1) / (2 * area[:, None])
    cy = np.stack([p[:, 2, 0] - p[:, 1, 0], p[:, 0, 0] - p[:, 2, 0], p[:, 1, 0] - p[:, 0, 0]], 1) / (2 * area[:, None])
    vx, vy = v[tri, 0], v[tri, 1]
    exx, eyy = (bx * vx).sum(1), (cy * vy).sum(1)
    exy = 0.5 * ((cy * vx).sum(1) + (bx * vy).sum(1))
    dens = np.sqrt(np.maximum(exx ** 2 + eyy ** 2 + exx * eyy + exy ** 2, 0.0))
    ub_ver = float(np.sum(sy * (2 / np.sqrt(3)) * dens * area * th[tri].mean(1)))
    if abs(res["ub"] / ub_ver - 1) > 1e-5:
        errs.append(f"UB reported {res['ub']:.6g} != recomputed {ub_ver:.6g}")

    # ---- lower bound: admissibility recomputed here
    S = np.asarray(res["stress"], float)                       # (T, 3, 3): vertex k, (Nxx, Nyy, Nxy)
    Nxx, Nyy, Nxy = S[:, :, 0], S[:, :, 1], S[:, :, 2]
    scale = np.abs(S).max()
    eq1 = (bx * Nxx).sum(1) + (cy * Nxy).sum(1)                # div N, constant per element
    eq2 = (bx * Nxy).sum(1) + (cy * Nyy).sum(1)
    h = np.sqrt(2 * area)
    r_eq = float(np.max(np.abs(np.concatenate([eq1 * h, eq2 * h]))) / scale)
    T = len(tri)
    el = np.tile(np.arange(T), 3)
    la, lb_ = np.repeat([0, 1, 2], T), np.repeat([1, 2, 0], T)
    ga, gb = tri[el, la], tri[el, lb_]
    d = nodes[gb] - nodes[ga]
    ln = np.linalg.norm(d, axis=1)
    n = np.stack([d[:, 1], -d[:, 0]], 1) / ln[:, None]

    def trac(e, k, nn):
        return np.stack([S[e, k, 0] * nn[:, 0] + S[e, k, 2] * nn[:, 1], S[e, k, 2] * nn[:, 0] + S[e, k, 1] * nn[:, 1]], 1)

    key = np.stack([np.minimum(ga, gb), np.maximum(ga, gb)], 1)
    _, inv, cnt = np.unique(key, axis=0, return_inverse=True, return_counts=True)
    order = np.argsort(inv, kind="stable")
    starts = np.concatenate([[0], np.cumsum(cnt)[:-1]])
    h1, h2 = order[starts[cnt == 2]], order[starts[cnt == 2] + 1]
    worst = 0.0
    for node_h1, k1 in ((ga[h1], la[h1]), (gb[h1], lb_[h1])):
        k2 = np.where(ga[h2] == node_h1, la[h2], lb_[h2])
        worst = max(worst, float(np.abs(trac(el[h1], k1, n[h1]) - trac(el[h2], k2, n[h1])).max()))
    r_cont = worst / scale
    hb = order[starts[cnt == 1]]
    mid = 0.5 * (nodes[ga[hb]] + nodes[gb[hb]])
    on_l, on_r = np.abs(mid[:, 0]) < tol, np.abs(mid[:, 0] - L) < tol
    free = ~(on_l | on_r)
    r_free = 0.0
    for k in (la[hb[free]], lb_[hb[free]]):
        r_free = max(r_free, float(np.abs(trac(el[hb[free]], k, n[hb[free]])).max()))
    if grips == "smooth":
        g = hb[on_l | on_r]
        for k in (la[g], lb_[g]):
            r_free = max(r_free, float(np.abs(trac(el[g], k, n[g])[:, 1]).max()))
    r_free /= scale
    if max(r_eq, r_cont, r_free) > 1e-6:
        errs.append(f"LB not admissible: eq {r_eq:.1e}, cont {r_cont:.1e}, bnd {r_free:.1e}")
    f = np.sqrt(np.maximum(Nxx ** 2 - Nxx * Nyy + Nyy ** 2 + 3 * Nxy ** 2, 0.0))
    ratio = float((f / (sy * th[tri])).max())
    if ratio > 1 + 1e-4:
        errs.append(f"yield violated by factor {ratio:.6f}")
    hr = hb[on_r]
    P = float(np.sum(0.5 * ln[hr] * (trac(el[hr], la[hr], n[hr])[:, 0] + trac(el[hr], lb_[hr], n[hr])[:, 0])))
    lb_cert = P / max(1.0, ratio)
    if lb_cert > ub_ver * (1 + 1e-6):
        errs.append(f"weak duality violated: LB {lb_cert:.6g} > UB {ub_ver:.6g}")
    return ub_ver, lb_cert, errs, dict(r_eq=r_eq, r_cont=r_cont, r_bnd=r_free, yield_ratio=ratio,
                                       nodes=len(nodes), tris=len(tri))


# ----------------------------------------------------------------------------- main
def main():
    t_start = time.time()
    gate_errs = []
    sy, t0 = 281.6, 10.5
    xs, ys = np.arange(0, 236, 2.0), np.arange(0, 76, 2.0)
    tu = np.full((len(xs), len(ys)), t0)
    exact = sy * ys[-1] * t0

    r = solve_bounds(xs, ys, tu, sy, grips="smooth")
    ub, lb, errs, _ = verify(r, xs, ys, tu, sy, "smooth")
    print(f"G1 uniform strip, smooth grips: UB/exact={ub / exact:.6f} LB/exact={lb / exact:.6f} {errs}")
    if errs or abs(ub / exact - 1) > 1e-3 or abs(lb / exact - 1) > 1e-3:
        gate_errs.append("G1")
    r = solve_bounds(xs, ys, tu, sy, grips="rigid")
    ub, lb, errs, _ = verify(r, xs, ys, tu, sy, "rigid")
    print(f"G2 uniform strip, rigid grips: UB/exact={ub / exact:.6f} LB/exact={lb / exact:.6f} {errs}")
    if errs or lb / exact < 1 - 1e-3 or lb > ub * (1 + 1e-6):
        gate_errs.append("G2")

    d = np.load(VAL, allow_pickle=True)
    gaps, times, nodes = [], [], []
    for i in range(len(d["name"])):
        xs, ys, t = d["x_mm"][i], d["y_mm"][i], d["thickness_mm"][i]
        sy = float(d["sigma_y_MPa"][i])
        t1 = time.time()
        r = solve_bounds(xs, ys, t, sy, grips="rigid")
        dt = time.time() - t1
        ub, lb, errs, info = verify(r, xs, ys, t, sy, "rigid")
        gap = (ub - lb) / ub
        gaps.append(gap)
        times.append(dt)
        nodes.append(info["nodes"])
        print(f"{d['name'][i]:12s} UB={ub / 1e3:8.2f} kN  LB={lb / 1e3:8.2f} kN  gap={100 * gap:6.3f}%  "
              f"{dt:6.1f}s (ub {r['time_ub']:.1f}s, lb {r['time_lb']:.1f}s)  nodes={info['nodes']}  "
              f"res eq/cont/bnd={info['r_eq']:.1e}/{info['r_cont']:.1e}/{info['r_bnd']:.1e}  "
              f"yield={info['yield_ratio']:.6f}  {errs if errs else ''}")
        if errs:
            gate_errs.append(f"G3/G4 {d['name'][i]}")
    total = time.time() - t_start
    if total > RUNTIME_LIMIT:
        gate_errs.append(f"G5 runtime {total:.0f}s")
    print("---")
    print(f"gap_mean: {np.mean(gaps):.6f}")
    print("gates: " + ("PASS" if not gate_errs else "FAIL " + ",".join(gate_errs)))
    print(f"runtime_s: {total:.1f}")
    print(f"gap_max: {np.max(gaps):.6f}")
    print(f"solve_time_mean_s: {np.mean(times):.1f}")
    print(f"nodes_mean: {np.mean(nodes):.0f}")


if __name__ == "__main__":
    main()
