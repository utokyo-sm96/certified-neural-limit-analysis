#!/usr/bin/env python3
"""Method-side helpers for the common thickness model t_g (not part of the verifier; thickness_tg.py and verify_v2.py are
the ruler): t_g-safe vertex capacities for the vertex yield test on a mesh that is not aligned with the scan grid, and
the exact piece-vertex yield ratio used to scale a lower-bound field to t_g yield."""
import numpy as np

from thickness_tg import tg_eval, tg_quantities


def tg_minorant(nodes, tri, xs, ys, t, sweeps=1, quantities=None):
    """Per-triangle linear minorant of t_g for the vertex yield test (methods on a mesh not aligned with the scan grid).
    Returns c (T, 3): vertex values of a linear function m_T with m_T(p) <= t_g(p) at every piece vertex p of T, hence on
    all of T (t_g linear on each piece). With N linear on T, f(N(v_k)) <= sigma_y c_k at the 3 vertices implies
    f(N) <= sigma_y m_T <= sigma_y t_g on T (f convex). Start: the better (larger sum c, positive) of the additive
    minorant t_g(v) - delta_T and the multiplicative one alpha_T t_g(v); then coordinate ascent on sum c (each step keeps
    every piece-vertex constraint; one sweep reaches the fixed point on the probed plates). Always positive;
    quantities = tg_quantities(...) output if already computed."""
    nodes, tri = np.asarray(nodes, float), np.asarray(tri, int)
    _, own, lam, tpt = quantities if quantities is not None else tg_quantities(nodes, tri, xs, ys, t)
    T = len(tri)
    tv = tg_eval(nodes, xs, ys, t)[tri]
    tlin = (lam * tv[own]).sum(1)
    delta = np.full(T, -np.inf)
    np.maximum.at(delta, own, tlin - tpt)
    ca = tv - delta[:, None]
    alpha = np.full(T, np.inf)
    np.minimum.at(alpha, own, tpt / tlin)
    cm = tv * np.minimum(alpha, 1.0)[:, None]
    use_a = (ca.sum(1) >= cm.sum(1)) & (ca.min(1) > 0)
    c = np.where(use_a[:, None], ca, cm)
    for _ in range(sweeps):
        for k in range(3):
            rest = tpt - (lam * c[own]).sum(1) + lam[:, k] * c[own, k]
            pos = lam[:, k] > 1e-9
            bound = np.full(T, np.inf)
            np.minimum.at(bound, own[pos], rest[pos] / lam[pos, k])
            c[:, k] = np.where(np.isfinite(bound), np.maximum(c[:, k], bound), c[:, k])
    # round-off margin: the constraints are active at the optimum
    return c * (1 - 1e-9)


def tg_ratio(Nv, quantities, sy):
    """Exact yield ratio max_T max_{p in T} f(N(p)) / (sigma_y t_g(p)) of per-vertex linear stresses Nv (T, 3, 3) (N/mm),
    over the piece vertices (quantities = tg_quantities(...) of the same mesh). Dividing Nv by it puts the field exactly
    at yield on t_g (the scaling verifier v2 certifies; v2 itself never scales a field up)."""
    _, own, lam, tpt = quantities
    Np = (lam[:, :, None] * np.asarray(Nv, float)[own]).sum(1)
    f = np.sqrt(np.maximum(Np[:, 0] ** 2 - Np[:, 0] * Np[:, 1] + Np[:, 1] ** 2 + 3 * Np[:, 2] ** 2, 0))
    return float((f / (sy * tpt)).max())
