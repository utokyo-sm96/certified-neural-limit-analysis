#!/usr/bin/env python3
"""The certified thickness model t_g and exact certification quantities on any triangle mesh.

t_g is the piecewise-linear interpolant of the scan values on the scan grid split into two triangles per cell with
alternating diagonals (cell (i, j): diagonal from (x_i, y_j) to (x_i+1, y_j+1) if i + j is even, otherwise from
(x_i+1, y_j) to (x_i, y_j+1)); this is limit_analysis.grid_mesh / adaptive_mesh level 0, and every adaptive FE mesh is
a refinement of it, so the FE solver certifies t_g exactly when its nodal thickness is t_g. For a mesh that is not
aligned with the scan grid (the network's 256 x 80 reference mesh), each triangle T is split by the grid lines and the
cell diagonals into convex pieces on which t_g is linear. With a stress field N that is linear on T:
  - f(N) - sigma_y t_g is convex on each piece, so  f(N) <= sigma_y t_g  on T  iff it holds at every piece vertex;
  - int_T t_g dA is the sum over the pieces of area x mean of t_g at the fan-triangle vertices (exact, t_g linear).
Both are exact up to round-off: no conservative correction is needed.
"""
import numpy as np

from thickness_model import _clip, _pieces


def tg_eval(pts, xs, ys, t):
    """t_g at points (n, 2)."""
    xs, ys, t = np.asarray(xs, float), np.asarray(ys, float), np.asarray(t, float)
    x = np.clip(pts[:, 0], xs[0], xs[-1])
    y = np.clip(pts[:, 1], ys[0], ys[-1])
    i = np.clip(np.searchsorted(xs, x, side="right") - 1, 0, len(xs) - 2)
    j = np.clip(np.searchsorted(ys, y, side="right") - 1, 0, len(ys) - 2)
    u = (x - xs[i]) / (xs[i + 1] - xs[i])
    v = (y - ys[j]) / (ys[j + 1] - ys[j])
    ta, tb, tc, td = t[i, j], t[i + 1, j], t[i + 1, j + 1], t[i, j + 1]   # a=(0,0) b=(1,0) c=(1,1) d=(0,1)
    even = (i + j) % 2 == 0
    low = v <= u                                                          # even cells: diagonal a-c
    te = np.where(low, ta + u * (tb - ta) + v * (tc - tb), ta + u * (tc - td) + v * (td - ta))
    left = u + v <= 1                                                     # odd cells: diagonal b-d
    to = np.where(left, ta + u * (tb - ta) + v * (td - ta), tc + (1 - u) * (td - tc) + (1 - v) * (tb - tc))
    return np.where(even, te, to)


@np.errstate(invalid="ignore", divide="ignore")
def tg_pieces(nodes, tri, xs, ys):
    """Pieces of every triangle on which t_g is linear: polygons (n, m, 2), counts, owner triangle."""
    xs, ys = np.asarray(xs, float), np.asarray(ys, float)
    P, cnt, own, cell = _pieces(np.asarray(nodes, float), np.asarray(tri, int), xs, ys)
    i, j = cell[:, 0], cell[:, 1]
    even = (i + j) % 2 == 0
    ax_ = np.where(even, xs[i], xs[i + 1])                                # diagonal from A to B
    ay_ = ys[j]
    bx_ = np.where(even, xs[i + 1], xs[i])
    by_ = ys[j + 1]
    nx, ny = by_ - ay_, ax_ - bx_                                         # normal of the diagonal
    c = nx * ax_ + ny * ay_
    out_P, out_c, out_o = [], [], []
    for sgn in (1.0, -1.0):
        Q, k = _clip(P, cnt, sgn * nx, sgn * ny, sgn * c)
        keep = k >= 3
        out_P.append(Q[keep])
        out_c.append(k[keep])
        out_o.append(own[keep])
    m = max(q.shape[1] for q in out_P)
    out_P = [np.concatenate([q, np.zeros((len(q), m - q.shape[1], 2))], 1) for q in out_P]
    return np.concatenate(out_P), np.concatenate(out_c), np.concatenate(out_o)


def tg_quantities(nodes, tri, xs, ys, t):
    """Exact int_T t_g (per triangle) and the yield-test points: owner triangle, barycentric coordinates in the owner,
    t_g at the point. The points are the vertices of the pieces (they include the triangle vertices)."""
    nodes, tri = np.asarray(nodes, float), np.asarray(tri, int)
    P, cnt, own = tg_pieces(nodes, tri, xs, ys)
    m = P.shape[1]
    valid = np.arange(m)[None, :] < cnt[:, None]
    flat = P.reshape(-1, 2)
    tv = tg_eval(flat, xs, ys, t).reshape(len(P), m)
    integ = np.zeros(len(P))
    for kk in range(1, m - 1):                                            # fan triangles (v0, vk, vk+1)
        use = kk + 1 < cnt
        a, b, c = P[:, 0], P[:, kk], P[:, kk + 1]
        area = 0.5 * np.abs((b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1]) - (b[:, 1] - a[:, 1]) * (c[:, 0] - a[:, 0]))
        integ += np.where(use, area * (tv[:, 0] + tv[:, kk] + tv[:, kk + 1]) / 3, 0.0)
    integral = np.bincount(own, integ, minlength=len(tri))
    # barycentric coordinates of the valid piece vertices in their owner triangle
    o = np.repeat(own, m)[valid.ravel()]
    q = flat[valid.ravel()]
    pt = nodes[tri[o]]
    M = np.stack([pt[:, 1] - pt[:, 0], pt[:, 2] - pt[:, 0]], 2)          # columns e1, e2
    lam12 = np.linalg.solve(M, (q - pt[:, 0])[..., None])[..., 0]
    lam = np.concatenate([1 - lam12.sum(1, keepdims=True), lam12], 1)
    return integral, o, lam, tv.ravel()[valid.ravel()]
