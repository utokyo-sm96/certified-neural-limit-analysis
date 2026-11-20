#!/usr/bin/env python3
"""Exact per-triangle quantities of the bilinear scan interpolant t_b on an arbitrary triangle mesh.

The certified thickness model is t_b(x, y) = bilinear interpolation of the scan grid (xs, ys, t) (cell-wise bilinear,
continuous). A mesh that is not aligned with the scan grid sees t_b as piecewise bilinear inside a triangle, and even an
aligned mesh sees a bilinear (not linear) function. For each triangle T this module computes, exactly up to round-off:
  integral_T  = int_T t_b dA                         (upper bound: exact dissipation for constant strain rate)
  delta_T     = max_T (t_lin - t_b)                  (lower bound: t_b >= t_lin - delta_T on T)
where t_lin is the linear interpolant of the nodal values t_b(vertices). Method: clip T by the scan grid lines into
convex pieces, each inside one scan cell; on a piece t_lin - t_b = alpha + beta x + gamma y - d x y has no interior
extremum (indefinite Hessian unless d = 0), so its maximum is at a piece vertex or at a stationary point of the
quadratic restriction to an edge. Integrals: fan triangulation and the edge-midpoint rule (exact for quadratics).

Requires every triangle to cross at most one vertical and one horizontal grid line (true when the mesh size is below
the scan step, as for the 256 x 80 reference mesh and for meshes built on the scan grid); this is checked.
"""
import numpy as np


def _clip(P, cnt, nx, ny, c):  # noqa: C901
    """Clip convex polygons P (n, m, 2) with cnt vertices against nx*x + ny*y <= c (per polygon); returns (Q, cnt2)."""
    n, m, _ = P.shape
    Q = np.zeros((n, m + 1, 2))
    out = np.zeros(n, int)
    rows = np.arange(n)
    for k in range(m):
        valid = k < cnt
        a = P[:, k]
        b = P[rows, np.where(k + 1 < cnt, k + 1, 0)]
        fa = nx * a[:, 0] + ny * a[:, 1] - c
        fb = nx * b[:, 0] + ny * b[:, 1] - c
        ina, inb = fa <= 0, fb <= 0
        w = valid & ina
        Q[rows[w], out[w]] = a[w]
        out[w] += 1
        cross = valid & (ina != inb)
        s = np.where(cross, fa / np.where(cross, fa - fb, 1.0), 0.0)
        ip = a + s[:, None] * (b - a)
        Q[rows[cross], out[cross]] = ip[cross]
        out[cross] += 1
    return Q, out


def _pieces(nodes, tri, xs, ys):
    """Split triangles by the scan grid lines; returns polygons (n, 7, 2), counts, owner triangle, cell (i, j)."""
    p = nodes[tri]                                                       # (T, 3, 2)
    T = len(tri)
    lo, hi = p.min(1), p.max(1)
    eps = 1e-12 * max(xs[-1], ys[-1])
    ix = np.searchsorted(xs, lo[:, 0] + eps, side="right") - 1          # cell of the lower-left corner
    iy = np.searchsorted(ys, lo[:, 1] + eps, side="right") - 1
    ix = np.clip(ix, 0, len(xs) - 2)
    iy = np.clip(iy, 0, len(ys) - 2)
    crossx = hi[:, 0] > xs[np.minimum(ix + 1, len(xs) - 1)] + eps
    crossy = hi[:, 1] > ys[np.minimum(iy + 1, len(ys) - 1)] + eps
    if np.any(hi[:, 0] > xs[np.minimum(ix + 2, len(xs) - 1)] + eps) or np.any(hi[:, 1] > ys[np.minimum(iy + 2, len(ys) - 1)] + eps):
        raise ValueError("a triangle crosses more than one grid line in x or y; refine the mesh")
    P0 = np.zeros((T, 7, 2))
    P0[:, :3] = p
    c0 = np.full(T, 3)
    polys, cnts, owner, cells = [], [], [], []
    for sx in (0, 1):
        for sy in (0, 1):
            sel = np.ones(T, bool)
            if sx:
                sel &= crossx
            if sy:
                sel &= crossy
            idx = np.nonzero(sel)[0]
            P, c = P0[idx], c0[idx]
            X = xs[ix[idx] + 1]
            Y = ys[iy[idx] + 1]
            # x side: sx = 0 keeps x <= X (only needed if the triangle crosses X), sx = 1 keeps x >= X
            if sx == 0:
                P, c = _clip(P, c, 1.0, 0.0, np.where(crossx[idx], X, np.inf))
            else:
                P, c = _clip(P, c, -1.0, 0.0, -X)
            P = P[:, :7] if P.shape[1] > 7 else np.concatenate([P, np.zeros((len(P), 7 - P.shape[1], 2))], 1)
            if sy == 0:
                P, c = _clip(P, c, 0.0, 1.0, np.where(crossy[idx], Y, np.inf))
            else:
                P, c = _clip(P, c, 0.0, -1.0, -Y)
            P = P[:, :7] if P.shape[1] > 7 else np.concatenate([P, np.zeros((len(P), 7 - P.shape[1], 2))], 1)
            keep = c >= 3
            polys.append(P[keep])
            cnts.append(c[keep])
            owner.append(idx[keep])
            cells.append(np.stack([ix[idx][keep] + sx, iy[idx][keep] + sy], 1))
    return np.concatenate(polys), np.concatenate(cnts), np.concatenate(owner), np.concatenate(cells)


@np.errstate(invalid="ignore", divide="ignore")
def tri_thickness_exact(nodes, tri, xs, ys, t):
    """Exact int_T t_b dA and max_T (t_lin - t_b) for every triangle (see module docstring)."""
    xs, ys, t = np.asarray(xs, float), np.asarray(ys, float), np.asarray(t, float)
    nodes, tri = np.asarray(nodes, float), np.asarray(tri, int)
    P, cnt, own, cell = _pieces(nodes, tri, xs, ys)
    i, j = cell[:, 0], cell[:, 1]
    x0, x1, y0, y1 = xs[i], xs[i + 1], ys[j], ys[j + 1]
    t00, t10, t01, t11 = t[i, j], t[i + 1, j], t[i, j + 1], t[i + 1, j + 1]

    def tb(x, y):                                                         # bilinear in the piece's cell
        u = (x - x0[:, None]) / (x1 - x0)[:, None]
        v = (y - y0[:, None]) / (y1 - y0)[:, None]
        return ((1 - u) * (1 - v) * t00[:, None] + u * (1 - v) * t10[:, None] + (1 - u) * v * t01[:, None]
                + u * v * t11[:, None])

    d_xy = (t00 - t10 - t01 + t11) / ((x1 - x0) * (y1 - y0))           # coefficient of x*y in the cell
    # linear interpolant of the owner triangle: t_lin(x) = a + g . x
    pt = nodes[tri[own]]
    from scipy.interpolate import RegularGridInterpolator
    tv = RegularGridInterpolator((xs, ys), t)(np.clip(nodes, [0, 0], [xs[-1], ys[-1]]))[tri[own]]
    M = np.concatenate([np.ones((len(own), 3, 1)), pt], 2)
    coef = np.linalg.solve(M, tv[..., None])[..., 0]                    # (a, gx, gy)
    tlin = lambda x, y: coef[:, :1] + coef[:, 1:2] * x + coef[:, 2:3] * y
    m = P.shape[1]
    k = np.arange(m)
    valid = k[None, :] < cnt[:, None]
    X, Y = P[..., 0], P[..., 1]
    # maximum of g = t_lin - t_b: vertices and edge stationary points
    g = np.where(valid, tlin(X, Y) - tb(X, Y), -np.inf)
    gmax = g.max(1)
    nxt = np.where(k[None, :] + 1 < cnt[:, None], k[None, :] + 1, 0)
    Xb, Yb = np.take_along_axis(X, nxt, 1), np.take_along_axis(Y, nxt, 1)
    dx, dy = Xb - X, Yb - Y
    q2 = -d_xy[:, None] * dx * dy                                        # quadratic coefficient of g along the edge
    ga, gb_ = g, np.where(valid, tlin(Xb, Yb) - tb(Xb, Yb), -np.inf)
    gm = np.where(valid, tlin(X + dx / 2, Y + dy / 2) - tb(X + dx / 2, Y + dy / 2), -np.inf)
    q1 = 4 * gm - 3 * ga - gb_                                            # g(s) = ga + q1 s + q2 s^2
    with np.errstate(divide="ignore", invalid="ignore"):
        s_star = np.where(q2 < 0, -q1 / (2 * q2), -1.0)
    ok = valid & (s_star > 0) & (s_star < 1)
    gstar = np.where(ok, ga + q1 * s_star + q2 * s_star ** 2, -np.inf)
    gmax = np.maximum(gmax, gstar.max(1))
    # integral of t_b over the piece: fan triangles (v0, vk, vk+1), edge-midpoint rule
    integ = np.zeros(len(P))
    for kk in range(1, m - 1):
        use = kk + 1 < cnt
        a_, b_, c_ = P[:, 0], P[:, kk], P[:, kk + 1]
        area = 0.5 * ((b_[:, 0] - a_[:, 0]) * (c_[:, 1] - a_[:, 1]) - (b_[:, 1] - a_[:, 1]) * (c_[:, 0] - a_[:, 0]))
        mids = np.stack([(a_ + b_) / 2, (b_ + c_) / 2, (c_ + a_) / 2], 1)
        val = tb(mids[..., 0], mids[..., 1]).sum(1) / 3
        integ += np.where(use, np.abs(area) * val, 0.0)
    T = len(tri)
    integral = np.bincount(own, integ, minlength=T)
    delta = np.full(T, -np.inf)
    np.maximum.at(delta, own, gmax)
    return integral, delta


def _selftest():
    rng = np.random.default_rng(0)
    xs, ys = np.array([0, 2, 4, 6, 7.0]), np.array([0, 2, 4, 5.5])
    t = 8 + rng.uniform(-2, 2, (len(xs), len(ys)))
    from scipy.interpolate import RegularGridInterpolator
    f = RegularGridInterpolator((xs, ys), t)
    # random small triangles inside the plate
    c = np.stack([rng.uniform(0.6, 6.4, 400), rng.uniform(0.6, 4.9, 400)], 1)
    off = rng.uniform(-0.5, 0.5, (400, 3, 2))
    nodes = (c[:, None, :] + off).reshape(-1, 2)
    tri = np.arange(len(nodes)).reshape(-1, 3)
    p = nodes[tri]
    sgn = np.sign((p[:, 1, 0] - p[:, 0, 0]) * (p[:, 2, 1] - p[:, 0, 1]) - (p[:, 1, 1] - p[:, 0, 1]) * (p[:, 2, 0] - p[:, 0, 0]))
    tri[sgn < 0] = tri[sgn < 0][:, [0, 2, 1]]
    integral, delta = tri_thickness_exact(nodes, tri, xs, ys, t)
    # brute force: dense barycentric sampling
    n = 120
    a, b = np.meshgrid(np.linspace(0, 1, n), np.linspace(0, 1, n))
    a, b = a.ravel(), b.ravel()
    keep = a + b <= 1
    a, b = a[keep], b[keep]
    err_i, err_d = 0.0, 0.0
    for k in range(len(tri)):
        q = nodes[tri[k]]
        pts = q[0] + a[:, None] * (q[1] - q[0]) + b[:, None] * (q[2] - q[0])
        tv = f(q)
        tl = tv[0] + a * (tv[1] - tv[0]) + b * (tv[2] - tv[0])
        tbv = f(pts)
        area = 0.5 * abs((q[1, 0] - q[0, 0]) * (q[2, 1] - q[0, 1]) - (q[1, 1] - q[0, 1]) * (q[2, 0] - q[0, 0]))
        err_d = max(err_d, (tl - tbv).max() - delta[k])                   # sampled max must not exceed the exact max
        err_i = max(err_i, abs(tbv.mean() * area - integral[k]) / (area * 8))
    print(f"self-test: sampled max exceeds exact max by at most {err_d:.2e} (must be <= ~1e-12); "
          f"integral vs sampling mean rel. diff {err_i:.2e} (sampling error only)")


if __name__ == "__main__":
    _selftest()
