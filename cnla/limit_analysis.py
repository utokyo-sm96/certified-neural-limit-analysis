#!/usr/bin/env python3
"""FE limit analysis of a corroded plate in tension: upper and lower bounds on the collapse load.

Model: plane stress, von Mises, rigid-perfectly plastic; generalised stresses N = t * sigma (N/mm),
thickness t(x, y) linear on each triangle. Plate [0, L] x [0, W]; the left grip is fixed, the right
grip moves with unit velocity in +x.

    grips="rigid":  welded grips, v = (0, 0) at x = 0 and v = (1, 0) at x = L
    grips="smooth": frictionless grips, v_x = 0 at x = 0, v_x = 1 at x = L, v_y free (one node pinned)

Upper bound (kinematic): v is continuous and linear per triangle. P_UB = min over admissible v of
    sum_e  sigma_y * (2/sqrt3) * |M eps(v)|_e * integral_e t dA,
    with |M eps| = sqrt(exx^2 + eyy^2 + exx*eyy + exy^2) and exy the tensor shear.
Lower bound (static): N is linear per triangle and may jump between triangles. It must satisfy
equilibrium in each element, traction continuity across edges and boundary tractions, and
f(N) = sqrt(Nxx^2 - Nxx*Nyy + Nyy^2 + 3 Nxy^2) <= sigma_y * t at every element vertex. That vertex
check is exact for linear N and linear t, because f is convex and positively homogeneous.
P_LB = max of the right-grip x-resultant.
Both bounds are second-order cone programs, solved with Clarabel.

API (used by the frozen evaluator eval_limit_analysis.py):
    solve_bounds(xs, ys, t, sigma_y, grips="rigid") -> dict(
        nodes (N,2) mm, tri (T,3) CCW, thick (N,) mm,
        ub (N), v (N,2), lb (N), stress (T,3,3) [element, vertex, (Nxx, Nyy, Nxy)] N/mm,
        time_ub, time_lb, status_ub, status_lb)
"""
import time

import clarabel
import numpy as np
import scipy.sparse as sp
from scipy.interpolate import RegularGridInterpolator

SQ3 = np.sqrt(3.0)
REFINE_FRACTION = 0.25        # share of grid cells refined at least once (highest UB dissipation first)
LEVEL2_FRACTION = 0.08        # share of grid cells refined twice (1 mm quarter cells, each union-jack)
LEVEL3_FRACTION = 0.03        # share of grid cells refined three times (0.5 mm leaves, each union-jack)
LEVEL4_FRACTION = 0.01        # share of grid cells refined four times (0.25 mm leaves, each union-jack)
UNIFORM_RATIO = 2.0           # no refinement if the largest cell dissipation is below this multiple of the mean


# ----------------------------------------------------------------------------- mesh
def grid_mesh(xs, ys):
    """Structured triangulation of the scan grid, alternating diagonals, counter-clockwise."""
    nx, ny = len(xs), len(ys)
    X, Y = np.meshgrid(xs, ys, indexing="ij")
    nodes = np.stack([X.ravel(), Y.ravel()], axis=1)
    idx = np.arange(nx * ny).reshape(nx, ny)
    a, b = idx[:-1, :-1], idx[1:, :-1]
    c, d = idx[1:, 1:], idx[:-1, 1:]
    even = (np.add.outer(np.arange(nx - 1), np.arange(ny - 1)) % 2 == 0)[..., None]
    t1 = np.where(even, np.stack([a, b, c], -1), np.stack([a, b, d], -1))
    t2 = np.where(even, np.stack([a, c, d], -1), np.stack([b, c, d], -1))
    return nodes, np.concatenate([t1.reshape(-1, 3), t2.reshape(-1, 3)])


def adaptive_mesh(xs, ys, level):
    """Conforming mesh of the scan grid with a refinement level per cell (nx-1, ny-1):
         0  two triangles (alternating diagonals, as grid_mesh)
         L>=1  the cell is split into 2^(L-1) x 2^(L-1) leaf cells, each union-jack (fan around its centre)
    Every leaf is triangulated as a fan through all nodes lying on its boundary, so the edges shared with
    finer neighbours are matched and there are no hanging nodes. Every grid node stays a mesh node."""
    nx, ny = len(xs), len(ys)
    R = 2 ** max(int(level.max()) - 1, 0) * 2               # integer units per grid cell (leaf centres are integer)
    key_id, coords, tris = {}, [], []

    def coord(k, g):
        i, r = divmod(k, R)
        return g[i] if r == 0 else g[i] + r / R * (g[i + 1] - g[i])

    def node(qx, qy):
        k = (qx, qy)
        if k not in key_id:
            key_id[k] = len(coords)
            coords.append((coord(qx, xs), coord(qy, ys)))
        return key_id[k]

    for i in range(nx):                                     # grid nodes first, in grid order
        for j in range(ny):
            node(R * i, R * j)
    leaves = []                                             # (x0, y0, size, level, i, j) in integer units
    for i in range(nx - 1):
        for j in range(ny - 1):
            n = 2 ** max(int(level[i, j]) - 1, 0)
            s = R // n
            for a in range(n):
                for b in range(n):
                    x0, y0 = R * i + a * s, R * j + b * s
                    for cx, cy in ((x0, y0), (x0 + s, y0), (x0 + s, y0 + s), (x0, y0 + s)):
                        node(cx, cy)
                    leaves.append((x0, y0, s, int(level[i, j]), i, j))
    for x0, y0, s, lv, i, j in leaves:
        poly = []
        for (px, py), (dx, dy) in (((x0, y0), (1, 0)), ((x0 + s, y0), (0, 1)),
                                   ((x0 + s, y0 + s), (-1, 0)), ((x0, y0 + s), (0, -1))):
            for k in range(s):                              # walk the edge counter-clockwise, unit steps
                key = (px + k * dx, py + k * dy)
                if key in key_id:
                    poly.append(key_id[key])
        if lv == 0 and len(poly) == 4:
            a, b, c, d = poly
            tris += [(a, b, c), (a, c, d)] if (i + j) % 2 == 0 else [(a, b, d), (b, c, d)]
        else:
            m = node(x0 + s // 2, y0 + s // 2)
            tris += [(poly[k], poly[(k + 1) % len(poly)], m) for k in range(len(poly))]
    return np.array(coords, float), np.array(tris, int)


def geometry(nodes, tri):
    p = nodes[tri]
    x1, y1, x2, y2, x3, y3 = p[:, 0, 0], p[:, 0, 1], p[:, 1, 0], p[:, 1, 1], p[:, 2, 0], p[:, 2, 1]
    area = 0.5 * ((x2 - x1) * (y3 - y1) - (y2 - y1) * (x3 - x1))
    b = np.stack([y2 - y3, y3 - y1, y1 - y2], 1)        # 2A * dN_i/dx
    c = np.stack([x3 - x2, x1 - x3, x2 - x1], 1)        # 2A * dN_i/dy
    return area, b, c


def half_edges(tri):
    """Element edges: (element, local vertex a, local vertex b); paired into interior / boundary edges."""
    T = len(tri)
    la = np.repeat(np.array([0, 1, 2]), T)
    lb = np.repeat(np.array([1, 2, 0]), T)
    el = np.tile(np.arange(T), 3)
    ga, gb = tri[el, la], tri[el, lb]
    key = np.stack([np.minimum(ga, gb), np.maximum(ga, gb)], 1)
    _, inv, cnt = np.unique(key, axis=0, return_inverse=True, return_counts=True)
    order = np.argsort(inv, kind="stable")
    starts = np.concatenate([[0], np.cumsum(cnt)[:-1]])
    inner = starts[cnt == 2]
    h1, h2 = order[inner], order[inner + 1]
    hb = order[starts[cnt == 1]]
    return el, la, lb, ga, gb, (h1, h2), hb


# ----------------------------------------------------------------------------- solver
def _solve(q, A, b, cones):
    st = clarabel.DefaultSettings()
    st.verbose = False
    st.direct_solve_method = "qdldl"             # 2x faster than faer on the refined LB meshes
    t0 = time.time()
    sol = clarabel.DefaultSolver(sp.csc_matrix((len(q), len(q))), q, A.tocsc(), b, cones, st).solve()
    return sol, time.time() - t0


def upper_bound(nodes, tri, thick, t0, L, grips, tol):
    N, T = len(nodes), len(tri)
    area, b, c = geometry(nodes, tri)
    inv = 1.0 / (2.0 * area)
    dx, dy = 2 * tri, 2 * tri + 1
    s_idx = 2 * N + np.arange(T)
    rows, cols, vals = [], [], []

    def add(r, cc, v):
        rows.append(np.broadcast_to(r, cc.shape).ravel())
        cols.append(cc.ravel())
        vals.append(np.broadcast_to(v, cc.shape).ravel())

    r0 = 4 * np.arange(T)[:, None]
    add(r0[:, 0], s_idx, -1.0)                                            # slack_0 = s_e
    add(r0 + 1, dx, -(b * inv[:, None]))                                  # exx + eyy/2
    add(r0 + 1, dy, -(0.5 * c * inv[:, None]))
    add(r0 + 2, dy, -(0.5 * SQ3 * c * inv[:, None]))                      # (sqrt3/2) eyy
    add(r0 + 3, dx, -(0.5 * c * inv[:, None]))                            # exy (tensor)
    add(r0 + 3, dy, -(0.5 * b * inv[:, None]))
    A_soc = sp.coo_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
                          shape=(4 * T, 2 * N + T))

    left = np.where(np.abs(nodes[:, 0]) < tol)[0]
    right = np.where(np.abs(nodes[:, 0] - L) < tol)[0]
    if grips == "rigid":
        fix = np.concatenate([2 * left, 2 * left + 1, 2 * right, 2 * right + 1])
        val = np.concatenate([np.zeros(2 * len(left)), np.ones(len(right)), np.zeros(len(right))])
    else:
        pin = left[np.argmin(nodes[left, 1])]
        fix = np.concatenate([2 * left, 2 * right, [2 * pin + 1]])
        val = np.concatenate([np.zeros(len(left)), np.ones(len(right)), [0.0]])
    A_fix = sp.coo_matrix((np.ones(len(fix)), (np.arange(len(fix)), fix)), shape=(len(fix), 2 * N + T))

    cost = (2.0 / SQ3) * area * thick[tri].mean(1) / t0
    q = np.concatenate([np.zeros(2 * N), cost])
    A = sp.vstack([A_fix, A_soc])
    bb = np.concatenate([val, np.zeros(4 * T)])
    cones = [clarabel.ZeroConeT(len(fix))] + [clarabel.SecondOrderConeT(4)] * T
    sol, dt = _solve(q, A, bb, cones)
    x = np.asarray(sol.x)
    return dict(ub_hat=float(q @ x), v=x[:2 * N].reshape(N, 2), status=str(sol.status), time=dt,
                diss=cost * x[2 * N:])


def lower_bound(nodes, tri, thick, t0, L, W, grips, tol):
    T = len(tri)
    area, b, c = geometry(nodes, tri)
    nvar = 9 * T

    def vid(e, k, comp):                                                 # variable index of N_comp at (e, k)
        return 9 * e + 3 * k + comp

    rows, cols, vals = [], [], []
    nrow = [0]

    def add_rows(cols_list, vals_list):
        """Each entry of cols_list / vals_list is an (m,) array; together they form m new rows."""
        m = len(cols_list[0])
        r = nrow[0] + np.arange(m)
        for cc, vv in zip(cols_list, vals_list):
            rows.append(r)
            cols.append(cc)
            vals.append(np.broadcast_to(vv, (m,)).astype(float))
        nrow[0] += m

    e = np.arange(T)
    # Equilibrium (times 2A): sum_k b_k Nxx_k + c_k Nxy_k = 0 and sum_k b_k Nxy_k + c_k Nyy_k = 0.
    # It is satisfied exactly by eliminating one Nxx and one Nyy per element (matrix Z below), so
    # the cone program has 7 free parameters per element and no equilibrium rows.
    p_el = np.argmax(np.abs(b), axis=1)
    q_el = np.argmax(np.abs(c), axis=1)
    elim_xx, elim_yy = vid(e, p_el, 0), vid(e, q_el, 1)
    kept = np.ones(nvar, bool)
    kept[elim_xx] = kept[elim_yy] = False
    new = -np.ones(nvar, int)
    new[kept] = np.arange(kept.sum())
    zr, zc, zv = [np.nonzero(kept)[0]], [new[kept]], [np.ones(kept.sum())]
    bp, cq = b[e, p_el], c[e, q_el]
    for k in range(3):
        m = p_el != k                                                     # Nxx_p = -(sum_{k!=p} b_k Nxx_k + sum_k c_k Nxy_k) / b_p
        zr += [elim_xx[m], elim_xx]
        zc += [new[vid(e[m], k, 0)], new[vid(e, k, 2)]]
        zv += [-b[m, k] / bp[m], -c[:, k] / bp]
        m = q_el != k                                                     # Nyy_q = -(sum_k b_k Nxy_k + sum_{k!=q} c_k Nyy_k) / c_q
        zr += [elim_yy, elim_yy[m]]
        zc += [new[vid(e, k, 2)], new[vid(e[m], k, 1)]]
        zv += [-b[:, k] / cq, -c[m, k] / cq[m]]
    Z = sp.csr_matrix((np.concatenate(zv), (np.concatenate(zr), np.concatenate(zc))), shape=(nvar, int(kept.sum())))

    el, la, lb, ga, gb, (h1, h2), hb = half_edges(tri)
    d = nodes[gb] - nodes[ga]
    ln = np.linalg.norm(d, axis=1)
    nrm = np.stack([d[:, 1], -d[:, 0]], 1) / ln[:, None]

    def traction_rows(e1, k1, e2, k2, n, comps=("x", "y")):
        """rows: traction of (e1,k1) minus traction of (e2,k2) (e2 = None -> just e1)."""
        for comp in comps:
            if comp == "x":
                cl = [vid(e1, k1, 0), vid(e1, k1, 2)]
                vl = [n[:, 0], n[:, 1]]
                if e2 is not None:
                    cl += [vid(e2, k2, 0), vid(e2, k2, 2)]
                    vl += [-n[:, 0], -n[:, 1]]
            else:
                cl = [vid(e1, k1, 1), vid(e1, k1, 2)]
                vl = [n[:, 1], n[:, 0]]
                if e2 is not None:
                    cl += [vid(e2, k2, 1), vid(e2, k2, 2)]
                    vl += [-n[:, 1], -n[:, 0]]
            add_rows(cl, vl)

    # interior edges: continuity at both end nodes
    n = nrm[h1]
    e1, e2 = el[h1], el[h2]
    for node_of_h1, k1 in ((ga[h1], la[h1]), (gb[h1], lb[h1])):
        k2 = np.where(ga[h2] == node_of_h1, la[h2], lb[h2])
        traction_rows(e1, k1, e2, k2, n)

    # boundary edges
    mid = 0.5 * (nodes[ga[hb]] + nodes[gb[hb]])
    on_left = np.abs(mid[:, 0]) < tol
    on_right = np.abs(mid[:, 0] - L) < tol
    free = ~(on_left | on_right)
    for mask, comps in ((free, ("x", "y")),) + (((on_left | on_right), ("y",)),) * (grips == "smooth"):
        h = hb[mask]
        for k in (la[h], lb[h]):
            traction_rows(el[h], k, None, None, nrm[h], comps)

    A_eq = sp.coo_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
                         shape=(nrow[0], nvar))

    # objective: maximise P = sum over right edges of len/2 * (Nxx(a) + Nxx(b))
    hr = hb[on_right]
    g = np.zeros(nvar)
    np.add.at(g, vid(el[hr], la[hr], 0), 0.5 * ln[hr])
    np.add.at(g, vid(el[hr], lb[hr], 0), 0.5 * ln[hr])

    # yield cones at every element vertex: [t/t0, Nxx - Nyy/2, sqrt3/2 Nyy, sqrt3 Nxy]
    ev = np.repeat(e, 3)
    kv = np.tile(np.arange(3), T)
    r0 = 4 * np.arange(3 * T)
    A_soc = sp.coo_matrix((np.concatenate([-np.ones(3 * T), 0.5 * np.ones(3 * T), -0.5 * SQ3 * np.ones(3 * T),
                                           -SQ3 * np.ones(3 * T)]),
                           (np.concatenate([r0 + 1, r0 + 1, r0 + 2, r0 + 3]),
                            np.concatenate([vid(ev, kv, 0), vid(ev, kv, 1), vid(ev, kv, 1), vid(ev, kv, 2)]))),
                          shape=(12 * T, nvar))
    b_soc = np.zeros(12 * T)
    b_soc[r0] = thick[tri[ev, kv]] / t0

    A = sp.vstack([A_eq, A_soc]).tocsr() @ Z
    bb = np.concatenate([np.zeros(nrow[0]), b_soc])
    cones = [clarabel.ZeroConeT(nrow[0])] + [clarabel.SecondOrderConeT(4)] * (3 * T)
    sol, dt = _solve(-(Z.T @ g), A, bb, cones)
    y = Z @ np.asarray(sol.x)
    return dict(lb_hat=float(g @ y), stress_hat=y.reshape(T, 3, 3), status=str(sol.status), time=dt)


def solve_bounds(xs, ys, t, sigma_y, grips="rigid", thick_model="bilinear"):
    """thick_model: "bilinear" (nodal values of the bilinear scan interpolant; step-1 default, verifier v1) or "tg"
    (nodal values of t_g, the piecewise-linear interpolant on the scan grid with alternating diagonals; every adaptive
    mesh refines that triangulation, so the bounds are certified for t_g exactly; verifier v2)."""
    xs, ys, t = np.asarray(xs, float), np.asarray(ys, float), np.asarray(t, float)
    L, W = xs[-1], ys[-1]
    t0 = float(t.max())                                                  # scaling only
    if thick_model == "tg":
        from thickness_tg import tg_eval
        interp = lambda q: tg_eval(np.asarray(q, float), xs, ys, t)
    else:
        interp = RegularGridInterpolator((xs, ys), t)
    tol = 1e-6 * max(L, W)
    # pass 1: cheap upper bound on the grid mesh, to locate the plastic zone
    nodes, tri = grid_mesh(xs, ys)
    ub0 = upper_bound(nodes, tri, interp(nodes), t0, L, grips, tol)
    ncell = (len(xs) - 1) * (len(ys) - 1)
    cell_diss = ub0["diss"][:ncell] + ub0["diss"][ncell:]
    rank = np.argsort(cell_diss)[::-1]
    level = np.zeros(ncell, int)
    if cell_diss.max() < UNIFORM_RATIO * cell_diss.mean():          # plastic flow ~uniform: refining cannot help
        rank = rank[:0]
    level[rank[:int(REFINE_FRACTION * ncell)]] = 1
    level[rank[:int(LEVEL2_FRACTION * ncell)]] = 2
    level[rank[:int(LEVEL3_FRACTION * ncell)]] = 3
    level[rank[:int(LEVEL4_FRACTION * ncell)]] = 4
    # pass 2: refine where the dissipation is highest, then both bounds on that mesh
    nodes, tri = adaptive_mesh(xs, ys, level.reshape(len(xs) - 1, len(ys) - 1))
    thick = interp(nodes)
    ub = upper_bound(nodes, tri, thick, t0, L, grips, tol)
    lb = lower_bound(nodes, tri, thick, t0, L, W, grips, tol)
    scale = sigma_y * t0
    return dict(nodes=nodes, tri=tri, thick=thick,
                ub=ub["ub_hat"] * scale, v=ub["v"], status_ub=ub["status"], time_ub=ub["time"],
                lb=lb["lb_hat"] * scale, stress=lb["stress_hat"] * scale, status_lb=lb["status"],
                time_lb=lb["time"])


if __name__ == "__main__":
    xs, ys = np.arange(0, 236, 2.0), np.arange(0, 76, 2.0)
    r = solve_bounds(xs, ys, np.full((len(xs), len(ys)), 10.5), 281.6, grips="smooth")
    ex = 281.6 * ys[-1] * 10.5
    print(f"uniform strip, smooth grips: UB/exact={r['ub'] / ex:.6f} LB/exact={r['lb'] / ex:.6f} "
          f"({r['status_ub']}, {r['status_lb']}; {r['time_ub']:.1f}s, {r['time_lb']:.1f}s)")
