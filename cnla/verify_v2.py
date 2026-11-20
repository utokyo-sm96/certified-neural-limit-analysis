#!/usr/bin/env python3
"""Verifier v2: certified bounds for ONE thickness model on every mesh, t_g (thickness_tg.py): the piecewise-linear
interpolant of the scan on the scan grid split by alternating diagonals.

v1 (eval_limit_analysis.verify, frozen) certifies the piecewise-linear interpolant of the bilinear scan interpolant on the
result's own mesh, so brackets on different meshes refer to slightly different thickness fields. v2 keeps the
thickness-independent checks of v1 (mesh, grip velocities, equilibrium, traction continuity at both ends of every
interior edge, free edges, grip resultant P) and recomputes both bounds for t_g, exactly:
  UB = sum_T (2/sqrt3) sigma_y D_T int_T t_g dA            (exact dissipation of the piecewise-linear velocity)
  LB = P / max(1, max_x f(N(x)) / (sigma_y t_g(x)))         x over the vertices of the pieces of every triangle on
which t_g is linear; f(N) - sigma_y t_g is convex on each piece, so this vertex test is exact.
Returns (ub, lb, errors, info)."""
import numpy as np

from eval_limit_analysis import verify
from thickness_tg import tg_quantities

SQ3 = np.sqrt(3.0)
V1_BOUND_CHECKS = ("thickness differs", "UB reported", "yield violated", "weak duality")


def verify_v2(res, xs, ys, t, sy, grips="rigid"):
    ub1, lb1, errs, info = verify(res, xs, ys, t, sy, grips)
    errs = [e for e in errs if not e.startswith(V1_BOUND_CHECKS)]       # replaced by the t_g checks below
    nodes, tri = np.asarray(res["nodes"], float), np.asarray(res["tri"], int)
    integral, own, lam, tpt = tg_quantities(nodes, tri, xs, ys, t)
    if np.any(tpt <= 0):
        errs.append("non-positive thickness in the certified model")
    p = nodes[tri]
    area = 0.5 * ((p[:, 1, 0] - p[:, 0, 0]) * (p[:, 2, 1] - p[:, 0, 1]) - (p[:, 1, 1] - p[:, 0, 1]) * (p[:, 2, 0] - p[:, 0, 0]))
    bx = np.stack([p[:, 1, 1] - p[:, 2, 1], p[:, 2, 1] - p[:, 0, 1], p[:, 0, 1] - p[:, 1, 1]], 1) / (2 * area[:, None])
    cy = np.stack([p[:, 2, 0] - p[:, 1, 0], p[:, 0, 0] - p[:, 2, 0], p[:, 1, 0] - p[:, 0, 0]], 1) / (2 * area[:, None])
    v = np.asarray(res["v"], float)
    vx, vy = v[tri, 0], v[tri, 1]
    exx, eyy = (bx * vx).sum(1), (cy * vy).sum(1)
    exy = 0.5 * ((cy * vx).sum(1) + (bx * vy).sum(1))
    dens = np.sqrt(np.maximum(exx ** 2 + eyy ** 2 + exx * eyy + exy ** 2, 0.0))
    ub2 = float(np.sum(sy * (2 / SQ3) * dens * integral))
    S = np.asarray(res["stress"], float)                                  # (T, 3 vertices, 3 components), linear on T
    N = np.einsum("pk,pkc->pc", lam, S[own])
    f = np.sqrt(np.maximum(N[:, 0] ** 2 - N[:, 0] * N[:, 1] + N[:, 1] ** 2 + 3 * N[:, 2] ** 2, 0.0))
    ratio2 = float((f / (sy * np.maximum(tpt, 1e-300))).max())
    P = lb1 * max(1.0, info["yield_ratio"])                              # grip resultant (v1 divides it by its ratio)
    lb2 = P / max(1.0, ratio2)
    if lb2 > ub2 * (1 + 1e-6):
        errs.append(f"weak duality violated (v2): LB {lb2:.6g} > UB {ub2:.6g}")
    info = dict(info, ub_v1=ub1, lb_v1=lb1, yield_ratio_v2=ratio2, n_test_points=len(tpt))
    return ub2, lb2, errs, info
