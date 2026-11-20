#!/usr/bin/env python3
"""EDITABLE in loops nlo-refine-1107, fvs-hybrid-1107, fvs-net-hybrid-1108: from the frozen neural limit operator
to certified bounds. Works with ZP networks (nlo_train) and FVS networks (nlo_train_fvs, per-vertex stresses).

bounds(model, plate) receives the frozen network and one plate dict(xs, ys, t, sigma_y) and returns
    dict(ub_res=dict(nodes, tri, thick, v, ub), lb_res=dict(nodes, tri, thick, stress, lb))
with stress (T, 3, 3) per vertex (Nxx, Nyy, Nxy) in N/mm and bounds in N. The frozen evaluator
nlo_refine_eval.py certifies both independently and times the call (budget: mean 5 s per plate).

Mirror ensemble: the collapse load is invariant under x- and y-mirroring of the plate (rigid grips at
both ends). The network is run on the 4 mirror images; each field is mapped back to the original
plate (a mirrored admissible field is admissible: v -> (1 - vx, vy) under x-mirror, (vx, -vy) under
y-mirror, Nxy changes sign under a single mirror).

Convex combination: the 4 fields are put on one canonical mesh; the best combination is found by two
small cone programs: LB = max over weights w of the grip resultant of sum w_k N_k subject to yield in
every triangle (any combination of equilibrated fields is equilibrated); UB = min over w (sum w_k = 1,
so the grip conditions hold) of the dissipation of sum w_k v_k.

Local corrections (loop fvs-hybrid-1107):
  * LB: C1-cubic (FVS) Airy windows around the hottest cells (refine_lb_fvs): N = lam N0 + N_fvs(dc), linear stresses,
    yield at the vertices; re-selected rounds on an anytime schedule (no new round after LB_SECONDS); the window cone
    programs are truncated at 14 interior-point iterations (the result is rescaled to yield and certified anyway).
  * UB: one local round over the band carrying 98 % of the dissipation, capped hottest-first at 16000 nodes,
    solve truncated at 15 interior-point iterations (loop fvs-net-hybrid-1108).
"""
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import nlo_train  # noqa: E402  (frozen in this loop)
import nlo_train_fvs  # noqa: E402  (frozen: FVS networks, per-vertex stresses)
import clarabel  # noqa: E402
import scipy.sparse as sp  # noqa: E402
from nlo_core import RefMesh  # noqa: E402

SQ3 = np.sqrt(3.0)
LB_SECONDS = 1.2                 # time after which no new LB round starts
LB_MAX_ROUNDS = 8                # budget knobs (defaults = the fvs-net-hybrid-1108 method; nlo_pareto.py varies them)
LB_MAX_DOF = 6000                # FVS window size
UB_ROUNDS = 1                    # local UB rounds
UB_MAX_NODES = 16000             # local UB window cap
UB_MAX_ITER = 15                 # local UB solve truncation (None: solve to tolerance)
GLOBAL_ARM = "fvs"               # plan C global stage (nlo_global): "zp" (AiryFit solves) or "fvs" (offline factorisation)
GLOBAL_SECONDS = 5.0             # LB ADMM time budget (used when GLOBAL_ITERS is None); 0 = stage off
GLOBAL_ITERS = 30                # LB ADMM iteration budget (deterministic; ~6 s for FVS)
REPAIR_SECONDS = 2.0             # local FVS repair rounds of the scaled ADMM iterate (2.3 before t_g: trimmed for the 15 s gate)
UB_GLOBAL_SECONDS = 0.0          # plan C global UB stage (ADMM over velocities, every iterate certified); 0 = off
UB_GLOBAL_ITERS = 75             # UB ADMM iteration budget (None: use UB_GLOBAL_SECONDS)
ADAPT_G0 = 0.04                  # adaptive budgets: both ADMM iteration counts x clip(hybrid gap / ADAPT_G0, LO, HI)
ADAPT_LO, ADAPT_HI = 0.7, 2.5    # (hard plates get more iterations, easy ones fewer; None: off)
THICK_MODEL = "bilinear"         # "tg": capacities and UB weights for the common thickness model t_g (verifier v2)
_CTX = {}                        # per-plate t_g data while bounds() runs: "tv" (T, 3) vertex capacities, "tub" (T,)
_C = {}


def canon():
    """Reference mesh, node lookup by doubled unit coordinates, sorted triangle keys."""
    if not _C:
        mesh = RefMesh(nlo_train.NX, nlo_train.NY)
        NX, NY = mesh.NX, mesh.NY
        ku = np.rint(mesh.unit[:, 0] * 2 * NX).astype(int)
        kw = np.rint(mesh.unit[:, 1] * 2 * NY).astype(int)
        look = -np.ones((2 * NX + 1, 2 * NY + 1), int)
        look[ku, kw] = np.arange(len(mesh.unit))
        n = len(mesh.unit)
        key = np.sort(mesh.tri, 1) @ np.array([n * n, n, 1])
        order = np.argsort(key)
        _C.update(mesh=mesh, look=look, n=n, key=key[order], order=order)
    return _C


def to_canonical(r, L, W):
    """Node permutation and triangle permutation of a mapped-back result onto the canonical mesh."""
    c = canon()
    NX, NY = c["mesh"].NX, c["mesh"].NY
    nid = c["look"][np.rint(r["nodes"][:, 0] / L * 2 * NX).astype(int), np.rint(r["nodes"][:, 1] / W * 2 * NY).astype(int)]
    tri_c = nid[r["tri"]]
    key = np.sort(tri_c, 1) @ np.array([c["n"] ** 2, c["n"], 1])
    tid = c["order"][np.searchsorted(c["key"], key)]
    v = np.zeros_like(r["v"])
    v[nid] = r["v"]
    ref = c["mesh"].tri[tid]                                             # canonical vertex order of each triangle
    S = np.asarray(r["stress"], float)                                   # (T, 3, 3) per vertex, in r's vertex order
    Nv = np.zeros((len(tid), 3, 3))
    rows = np.arange(len(tid))
    for k in range(3):                                                   # per-vertex stresses follow their vertex
        Nv[tid, k] = S[rows, np.argmax(tri_c == ref[:, k:k + 1], axis=1)]
    return v, Nv


def solve_cone(q, A, b, cones, max_iter=None):
    st = clarabel.DefaultSettings()
    st.verbose = False
    if max_iter:                                                         # truncated interior point: any iterate is
        st.max_iter = max_iter                                           # usable where the result is rescaled
    sol = clarabel.DefaultSolver(sp.csc_matrix((len(q), len(q))), q, sp.csc_matrix(A), b, cones, st).solve()
    return np.asarray(sol.x), str(sol.status)


def capv(th, tri):
    """Vertex capacities (T, 3): t_g-safe values while bounds() runs with THICK_MODEL "tg", else nodal thickness. The
    t_g-safe capacity of vertex k of T is t_g(v_k) - delta_T with delta_T = max_T (t_lin - t_g), so that the vertex yield
    test with these capacities implies f(N) <= sigma_y t_g on the whole triangle (N linear, capacity minorant linear)."""
    tv = _CTX.get("tv")
    return tv if tv is not None and len(tv) == len(tri) else th[tri]


def ubth(th, tri):
    """Per-triangle thickness of the dissipation (T,): int_T t_g / A_T with THICK_MODEL "tg", else mean nodal value."""
    tb = _CTX.get("tub")
    return tb if tb is not None and len(tb) == len(tri) else th[tri].mean(1)


def diss_w(v, nodes, tri, tb):
    """(2/sqrt3) sum_T A_T D_T tb_T: dissipation / sigma_y of a piecewise-linear velocity with per-triangle thickness."""
    p = nodes[tri]
    area = 0.5 * ((p[:, 1, 0] - p[:, 0, 0]) * (p[:, 2, 1] - p[:, 0, 1]) - (p[:, 1, 1] - p[:, 0, 1]) * (p[:, 2, 0] - p[:, 0, 0]))
    bx = np.stack([p[:, 1, 1] - p[:, 2, 1], p[:, 2, 1] - p[:, 0, 1], p[:, 0, 1] - p[:, 1, 1]], 1) / (2 * area[:, None])
    cy = np.stack([p[:, 2, 0] - p[:, 1, 0], p[:, 0, 0] - p[:, 2, 0], p[:, 1, 0] - p[:, 0, 0]], 1) / (2 * area[:, None])
    vx, vy = v[tri, 0], v[tri, 1]
    exx, eyy = (bx * vx).sum(1), (cy * vy).sum(1)
    exy = 0.5 * ((cy * vx).sum(1) + (bx * vy).sum(1))
    return float(((2 / SQ3) * area * tb * np.sqrt(np.maximum(exx ** 2 + eyy ** 2 + exx * eyy + exy ** 2, 0))).sum())


def tg_context(nodes, tri, xs, ys, t):
    """t_g-safe vertex capacities (best linear minorant per triangle, tg_capacity.tg_minorant), exact UB thickness of the
    canonical mesh and the piece-vertex data for the exact final scaling (tg_capacity.tg_ratio), for one plate."""
    from thickness_tg import tg_quantities
    from tg_capacity import tg_minorant
    q = tg_quantities(nodes, tri, xs, ys, t)
    p = nodes[tri]
    area = 0.5 * ((p[:, 1, 0] - p[:, 0, 0]) * (p[:, 2, 1] - p[:, 0, 1]) - (p[:, 1, 1] - p[:, 0, 1]) * (p[:, 2, 0] - p[:, 0, 0]))
    return dict(tv=tg_minorant(nodes, tri, xs, ys, t, quantities=q), tub=q[0] / area, q=q)


def ratios(N, tmin, sy):
    f = np.sqrt(np.maximum(N[:, 0] ** 2 - N[:, 0] * N[:, 1] + N[:, 1] ** 2 + 3 * N[:, 2] ** 2, 0))
    return f / (sy * tmin)


def combine_lb(Nvs, nodes, tri, th, W, sy, start=900, add=900, rounds=12):
    """max_w P(sum w_k N_k) s.t. yield at every triangle vertex, by cutting planes: with K weights at most ~K
    yield constraints are active, so a small cone program over a growing set of (triangle, vertex) pairs suffices.
    Nvs: per-vertex fields (T, 3, 3) (ZP fields are constant per triangle). Returns the combined per-vertex stress
    (T, 3, 3) scaled exactly to the yield surface (certified either way)."""
    tv = capv(th, tri)
    Nvs = [Nv / ratios_v(Nv, tv, sy).max() for Nv in Nvs]                # each field exactly at yield
    K = len(Nvs)
    s0 = sy * th.max()
    a = np.stack([Nv.reshape(-1, 3) for Nv in Nvs], -1) / s0             # (3T, 3, K)
    P = np.array([lb_v(Nv, W) for Nv in Nvs]) / s0
    tmin = tv.reshape(-1)                                                # vertex thickness per (triangle, vertex)
    rho = np.max([ratios_v(Nv, tv, sy).reshape(-1) for Nv in Nvs], 0)
    act = np.argsort(rho)[-start:]
    best, best_lb = None, -np.inf
    for _ in range(rounds):
        m = len(act)
        A = np.zeros((4 * m, K))
        r0 = 4 * np.arange(m)
        A[r0 + 1] = -(a[act, 0] - 0.5 * a[act, 1])
        A[r0 + 2] = -(0.5 * SQ3 * a[act, 1])
        A[r0 + 3] = -(SQ3 * a[act, 2])
        b = np.zeros(4 * m)
        b[r0] = tmin[act] / th.max()
        x, status = solve_cone(-P, A, b, [clarabel.SecondOrderConeT(4)] * m)
        if status not in ("Solved", "AlmostSolved"):
            break
        N = ((a @ x) * s0).reshape(-1, 3, 3)
        r = ratios_v(N, tv, sy).reshape(-1)
        N = N / r.max()
        lb = lb_v(N, W)
        if lb > best_lb:
            best, best_lb = N, lb
        new = np.setdiff1d(np.argsort(r)[-add:], act)
        if r.max() <= 1 + 1e-6 or not len(new):
            break
        act = np.union1d(act, new)
    return best


def combine_ub(vs, nodes, tri, th, sy):
    """min over w (sum w_k = 1) of the dissipation of sum w_k v_k; K is tiny, so the convex sum of norms is
    minimised by iteratively reweighted least squares (a (K-1) x (K-1) solve per pass)."""
    K = len(vs)
    p = nodes[tri]
    area = 0.5 * ((p[:, 1, 0] - p[:, 0, 0]) * (p[:, 2, 1] - p[:, 0, 1]) - (p[:, 1, 1] - p[:, 0, 1]) * (p[:, 2, 0] - p[:, 0, 0]))
    bx = np.stack([p[:, 1, 1] - p[:, 2, 1], p[:, 2, 1] - p[:, 0, 1], p[:, 0, 1] - p[:, 1, 1]], 1) / (2 * area[:, None])
    cy = np.stack([p[:, 2, 0] - p[:, 1, 0], p[:, 0, 0] - p[:, 2, 0], p[:, 1, 0] - p[:, 0, 0]], 1) / (2 * area[:, None])
    comp = []
    for v in vs:
        vx, vy = v[tri, 0], v[tri, 1]
        exx, eyy = (bx * vx).sum(1), (cy * vy).sum(1)
        exy = 0.5 * ((cy * vx).sum(1) + (bx * vy).sum(1))
        comp.append(np.stack([exx + 0.5 * eyy, 0.5 * SQ3 * eyy, exy], 1))
    comp = np.stack(comp, -1)                                            # (T, 3, K)
    cost = (2 / SQ3) * area * ubth(th, tri)

    # minimise sum_e cost_e |C_e w| over w = w0 + Z z (sum w = 1) by iteratively reweighted least squares
    d0 = [float(cost @ np.linalg.norm(comp[:, :, k], axis=1)) for k in range(K)]
    w0 = np.eye(K)[int(np.argmin(d0))]
    Z = np.vstack([np.eye(K - 1), -np.ones((1, K - 1))])                 # basis of {sum w = 0}
    g = comp @ w0                                                        # (T, 3)
    H = comp @ Z                                                         # (T, 3, K-1)
    z = np.zeros(K - 1)
    best_w, best_d = w0, min(d0)
    g2, H2 = g.reshape(-1), H.reshape(-1, K - 1)                         # 2-D BLAS products
    prev = np.inf
    for _ in range(15):                                                  # ~1e-7 relative by 15 passes
        nrm = np.linalg.norm((g2 + H2 @ z).reshape(-1, 3), axis=1)
        d = float(cost @ nrm)
        if d < best_d:
            best_w, best_d = w0 + Z @ z, d
        if prev - d < 1e-8 * d:                                          # IRLS is monotone: stop when it stalls
            break
        prev = d
        om = np.repeat(np.sqrt(cost / np.maximum(nrm, 1e-9 * nrm.max())), 3)
        Hw = H2 * om[:, None]
        M = Hw.T @ Hw
        rhs = -Hw.T @ (g2 * om)
        z = np.linalg.solve(M + 1e-14 * np.trace(M) * np.eye(K - 1), rhs)
    w = best_w / best_w.sum()                                            # exact sum 1 (grip conditions)
    return sum(wk * v for wk, v in zip(w, vs))


def airy_cell_matrix():
    """Sparse map c -> (psi_ww, psi_uu, -psi_uw) per triangle in cell units (cached)."""
    c = canon()
    if "A_cell" not in c:
        from nlo_core import zp_stress_matrix
        c["A_cell"] = zp_stress_matrix(c["mesh"], c["mesh"].NX, c["mesh"].NY).tocsc()
    return c["A_cell"]


def refine_lb_local(N0, th, tri, L, W, sy, theta=0.95, dil=1, max_coef=2500):
    """Local LB correction: N = lam * N0 + A_w dc, with Airy-coefficient corrections dc only around the
    triangles near yield. max lam * P0 + g_w . dc s.t. yield on the affected triangles, lam <= 1 / (worst
    unaffected ratio), traction-free edge rows. Any dc keeps equilibrium; the result is rescaled exactly."""
    mesh = canon()["mesh"]
    NX, NY = mesh.NX, mesh.NY
    hx, hy = L / NX, W / NY
    tmin = th[tri].min(1)
    r0 = ratios(N0, tmin, sy)
    hot = np.nonzero(r0 >= theta * r0.max())[0]
    hot = hot[np.argsort(r0[hot])[::-1]]                                 # hottest first
    cells = np.unique(hot // 4, return_index=True)
    cells = cells[0][np.argsort(cells[1])]                               # cells in order of their hottest triangle
    per = (3 + 2 * dil) ** 2                                             # coefficients touched per cell (upper bound)
    cells = cells[:max(1, max_coef // per * 2)]                          # cap the window, hottest cells kept
    ci, cj = cells // NY, cells % NY
    sel = np.zeros((NX + 2, NY + 2), bool)
    for a in range(-dil, 3 + dil):                                       # coefficients touching the hot cells
        for b in range(-dil, 3 + dil):
            sel[np.clip(ci + a, 0, NX + 1), np.clip(cj + b, 0, NY + 1)] = True
    cols = np.nonzero(sel.ravel())[0]
    s0 = sy * th.max()
    D = np.tile([1 / hy ** 2, 1 / hx ** 2, 1 / (hx * hy)], len(tri))
    Aw = sp.diags(D) @ airy_cell_matrix()[:, cols] / s0                  # (3T, m) physical / s0
    Aw = Aw.tocsr()
    aff = np.unique(Aw.nonzero()[0] // 3)
    out = np.setdiff1d(np.arange(len(tri)), aff)
    lam_max = 1.0 / r0[out].max() if len(out) else np.inf
    N0s = N0 / s0
    P0 = N0s[mesh.right, 0].sum() * W / NY
    g = np.asarray(Aw[3 * mesh.right].sum(0)).ravel() * W / NY
    m = len(cols)
    nv = 1 + m                                                           # [lam, dc]
    Ae = Aw[3 * aff]                                                     # Nxx rows of affected triangles
    Ay = Aw[3 * aff + 1]
    Axy = Aw[3 * aff + 2]
    na = len(aff)
    lamcol = lambda vals: sp.csr_matrix(vals[:, None])
    # cone rows per affected triangle: [tmin/t0; Nxx - Nyy/2; sqrt3/2 Nyy; sqrt3 Nxy] with N = lam N0 + A dc
    r1 = sp.hstack([lamcol(N0s[aff, 0] - 0.5 * N0s[aff, 1]), Ae - 0.5 * Ay])
    r2 = sp.hstack([lamcol(0.5 * SQ3 * N0s[aff, 1]), 0.5 * SQ3 * Ay])
    r3 = sp.hstack([lamcol(SQ3 * N0s[aff, 2]), SQ3 * Axy])
    zero = sp.csr_matrix((na, nv))
    blocks = sp.vstack([zero, -r1, -r2, -r3]).tocsr()
    perm = np.arange(4 * na).reshape(4, na).T.ravel()
    A_soc = blocks[perm]
    b_soc = np.zeros(4 * na)
    b_soc[0::4] = tmin[aff] / th.max()
    edge = np.intersect1d(aff, np.concatenate([mesh.bottom, mesh.top]))  # keep free edges traction-free
    A_eq = sp.vstack([sp.hstack([sp.csr_matrix((len(edge), 1)), Aw[3 * edge + 1]]),
                      sp.hstack([sp.csr_matrix((len(edge), 1)), Aw[3 * edge + 2]])]).tocsr()
    A_lam = sp.csr_matrix(([1.0], ([0], [0])), shape=(1, nv))
    A = sp.vstack([A_eq, A_lam, A_soc]).tocsc()
    b = np.concatenate([np.zeros(A_eq.shape[0]), [min(lam_max, 1e6)], b_soc])
    cones = [clarabel.ZeroConeT(A_eq.shape[0]), clarabel.NonnegativeConeT(1)] + [clarabel.SecondOrderConeT(4)] * na
    q = -np.concatenate([[P0], g])
    x, status = solve_cone(q, A, b, cones)
    if status not in ("Solved", "AlmostSolved"):
        return None
    N = (x[0] * N0s + (Aw @ x[1:]).reshape(-1, 3)) * s0
    return N / ratios(N, tmin, sy).max()


def fvs_cell_matrix():
    """Sparse map free FVS unknowns -> per-vertex (Nxx, Nyy, Nxy) in cell units (hx = hy = 1), rows 9e + 3v + comp,
    and the DOF layout (cached). Physical rows: comp 0 / hy^2, comp 1 / hx^2, comp 2 / (hx hy)."""
    c = canon()
    if "F_cell" not in c:
        from nlo_fvs import FVS, dof_layout, fvs_stress_matrix
        mesh = c["mesh"]
        c["F_cell"] = fvs_stress_matrix(mesh, FVS(mesh), mesh.NX, mesh.NY).tocsc()
        c["F_layout"] = dof_layout(mesh)
    return c["F_cell"], c["F_layout"]


def ratios_v(Nv, tv, sy):
    """Per-vertex yield ratios (T, 3) of per-vertex stresses Nv (T, 3, 3) against vertex thickness tv (T, 3)."""
    f = np.sqrt(np.maximum(Nv[..., 0] ** 2 - Nv[..., 0] * Nv[..., 1] + Nv[..., 1] ** 2 + 3 * Nv[..., 2] ** 2, 0))
    return f / (sy * tv)


def lb_v(Nv, W):
    """Right-grip resultant of per-vertex stresses (right triangles: vertices 0, 1 on x = L)."""
    mesh = canon()["mesh"]
    return 0.5 * (Nv[mesh.right, 0, 0] + Nv[mesh.right, 1, 0]).sum() * W / mesh.NY


def refine_lb_fvs(Nv0, th, tri, L, W, sy, theta=0.9, dil=1, max_dof=6000):
    """Local LB correction in the C1-cubic (FVS) space: N = lam * N0 + N_fvs(dc), with FVS DOF corrections dc on
    the cells around the triangles near yield (zero on the free edges, so dc is self-equilibrated, traction-free
    and carries no load). max lam s.t. yield at the 3 vertices of every affected triangle (exact for linear
    stresses), lam <= 1 / (worst unaffected ratio). Returns per-vertex stresses (T, 3, 3) at the yield surface."""
    mesh = canon()["mesh"]
    NX, NY = mesh.NX, mesh.NY
    hx, hy = L / NX, W / NY
    A_cell, (Vi, Hi, Vti, _, _) = fvs_cell_matrix()
    T = len(tri)
    tv = capv(th, tri)
    r0 = ratios_v(Nv0, tv, sy).max(1)
    hot = np.nonzero(r0 >= theta * r0.max())[0]
    hot = hot[np.argsort(r0[hot])[::-1]]
    cu, first = np.unique(hot // 4, return_index=True)
    cells = cu[np.argsort(first)][:max(1, max_dof // (5 * (1 + 2 * dil) ** 2) * 2)]
    ci, cj = cells // NY, cells % NY
    win = np.zeros((NX, NY), bool)
    for a in range(-dil, dil + 1):
        for b in range(-dil, dil + 1):
            win[np.clip(ci + a, 0, NX - 1), np.clip(cj + b, 0, NY - 1)] = True
    I, J = np.nonzero(win)
    cols = np.concatenate([Vi[I, J].ravel(), Vi[I + 1, J].ravel(), Vi[I + 1, J + 1].ravel(), Vi[I, J + 1].ravel(),
                           Hi[I, J], Hi[I, J + 1], Vti[I, J], Vti[I + 1, J]])
    cols = np.unique(cols[cols >= 0])
    s0 = sy * th.max()
    D = np.tile([1 / hy ** 2, 1 / hx ** 2, 1 / (hx * hy)], 3 * T) / s0
    Aw = (sp.diags(D) @ A_cell[:, cols]).tocsr()                          # (9T, m) physical / s0
    aff = np.unique(Aw.nonzero()[0] // 9)
    out = np.setdiff1d(np.arange(T), aff)
    lam_max = 1.0 / r0[out].max() if len(out) else np.inf
    base = (9 * aff[:, None] + 3 * np.arange(3)[None]).ravel()           # (aff triangle, vertex) rows
    Ae, Ay, Axy = Aw[base], Aw[base + 1], Aw[base + 2]
    n0 = (Nv0[aff] / s0).reshape(-1, 3)
    nc, m = len(base), len(cols)
    nv = 1 + m
    lamcol = lambda vals: sp.csr_matrix(vals[:, None])
    r1 = sp.hstack([lamcol(n0[:, 0] - 0.5 * n0[:, 1]), Ae - 0.5 * Ay])
    r2 = sp.hstack([lamcol(0.5 * SQ3 * n0[:, 1]), 0.5 * SQ3 * Ay])
    r3 = sp.hstack([lamcol(SQ3 * n0[:, 2]), SQ3 * Axy])
    blocks = sp.vstack([sp.csr_matrix((nc, nv)), -r1, -r2, -r3]).tocsr()
    A_soc = blocks[np.arange(4 * nc).reshape(4, nc).T.ravel()]
    b_soc = np.zeros(4 * nc)
    b_soc[0::4] = tv[aff].ravel() / th.max()
    A_lam = sp.csr_matrix(([1.0], ([0], [0])), shape=(1, nv))
    A = sp.vstack([A_lam, A_soc]).tocsc()
    b = np.concatenate([[min(lam_max, 1e6)], b_soc])
    q = np.zeros(nv)
    q[0] = -1.0
    x, status = solve_cone(q, A, b, [clarabel.NonnegativeConeT(1)] + [clarabel.SecondOrderConeT(4)] * nc, max_iter=14)
    if status not in ("Solved", "AlmostSolved", "MaxIterations"):        # rescaled to yield below: certified anyway
        return None
    N = (x[0] * Nv0 / s0 + (Aw @ x[1:]).reshape(T, 3, 3)) * s0
    return N / ratios_v(N, tv, sy).max()


def refine_ub_local(v0, nodes, tri, th, L, frac=0.98, max_nodes=16000):
    """Local UB correction: free the velocities of the nodes of the elements that carry `frac` of the
    dissipation (plus one layer), keep grip nodes and all other nodes fixed, and minimise the exact
    dissipation of the affected elements (one cone program). Any v with the grip values is admissible."""
    p = nodes[tri]
    area = 0.5 * ((p[:, 1, 0] - p[:, 0, 0]) * (p[:, 2, 1] - p[:, 0, 1]) - (p[:, 1, 1] - p[:, 0, 1]) * (p[:, 2, 0] - p[:, 0, 0]))
    bx = np.stack([p[:, 1, 1] - p[:, 2, 1], p[:, 2, 1] - p[:, 0, 1], p[:, 0, 1] - p[:, 1, 1]], 1) / (2 * area[:, None])
    cy = np.stack([p[:, 2, 0] - p[:, 1, 0], p[:, 0, 0] - p[:, 2, 0], p[:, 1, 0] - p[:, 0, 0]], 1) / (2 * area[:, None])
    cost = (2 / SQ3) * area * ubth(th, tri)
    vx, vy = v0[tri, 0], v0[tri, 1]
    exx, eyy = (bx * vx).sum(1), (cy * vy).sum(1)
    exy = 0.5 * ((cy * vx).sum(1) + (bx * vy).sum(1))
    d = cost * np.sqrt(np.maximum(exx ** 2 + eyy ** 2 + exx * eyy + exy ** 2, 0))
    order = np.argsort(d)[::-1]
    grip = (nodes[:, 0] < 1e-9 * L) | (nodes[:, 0] > L * (1 - 1e-12))

    def band(k):
        free = np.zeros(len(nodes), bool)
        free[np.unique(tri[order[:k]])] = True
        free[np.unique(tri[free[tri].any(1)])] = True                   # one layer of neighbours
        return free & ~grip

    k = np.searchsorted(np.cumsum(d[order]), frac * d.sum()) + 1
    free = band(k)
    for _ in range(6):                                                   # too large: cap hottest-first, not skip
        if free.sum() <= max_nodes or k == 1:
            break
        k = max(1, int(k * max_nodes / free.sum() * 0.95))
        free = band(k)
    fn = np.nonzero(free)[0]
    if len(fn) > max_nodes or not len(fn):
        return None
    col = -np.ones(len(nodes), int)
    col[fn] = np.arange(len(fn))
    aff = np.nonzero(free[tri].any(1))[0]
    na, nf = len(aff), len(fn)
    # strain components of affected elements: fixed-node part h, free-node part G (columns: vx_f, vy_f)
    rows, cols, vals = [], [], []
    h = np.zeros((na, 3))
    for k in range(3):
        nd = tri[aff, k]
        isf = free[nd]
        bxk, cyk = bx[aff, k], cy[aff, k]
        # component rows: 0: exx + eyy/2, 1: sqrt3/2 eyy, 2: exy
        for comp, (wx, wy) in enumerate(((bxk, 0.5 * cyk), (0 * bxk, 0.5 * SQ3 * cyk), (0.5 * cyk, 0.5 * bxk))):
            h[~isf, comp] += wx[~isf] * v0[nd[~isf], 0] + wy[~isf] * v0[nd[~isf], 1]
            e = np.nonzero(isf)[0]
            rows += [4 * e + 1 + comp, 4 * e + 1 + comp]
            cols += [col[nd[e]], nf + col[nd[e]]]
            vals += [-wx[e], -wy[e]]
    rows.append(4 * np.arange(na)); cols.append(2 * nf + np.arange(na)); vals.append(-np.ones(na))
    A = sp.coo_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(4 * na, 2 * nf + na))
    b = np.zeros(4 * na)
    for comp in range(3):
        b[4 * np.arange(na) + 1 + comp] = h[:, comp]
    cn = cost[aff] / cost[aff].sum()
    q = np.concatenate([np.zeros(2 * nf), cn])
    x, status = solve_cone(q, A, b, [clarabel.SecondOrderConeT(4)] * na, max_iter=UB_MAX_ITER)
    if status not in ("Solved", "AlmostSolved", "MaxIterations"):        # any v with the grip values is admissible
        return None
    v = v0.copy()
    v[fn, 0] = x[:nf]
    v[fn, 1] = x[nf:2 * nf]
    return v


def global_stage(lb_res, v, nodes, tri, th, L, W, sy, iters=None):
    """Plan C: ADMM with plate-independent exact equilibrium solves (nlo_global), warm-started from the hybrid's LB
    field (primal) and UB velocity (dual); the last iterate is scaled to yield and repaired by local FVS windows.
    Returns the better LB result (the evaluator certifies it independently)."""
    import nlo_global
    c = canon()
    key = "global_" + GLOBAL_ARM
    if key not in c:
        c[key] = (nlo_global.ZPArm if GLOBAL_ARM == "zp" else nlo_global.FVSArm)(c["mesh"])
    arm = c[key]
    if not np.array_equal(np.asarray(lb_res["tri"]), tri):
        return lb_res
    area, e = nlo_global.strains(nodes, tri, v)
    x, y = arm.setup(L, W, th, sy, np.asarray(lb_res["stress"], float), area, e, cap=capv(th, tri))
    Nv, _ = nlo_global.admm(arm, x, y, GLOBAL_SECONDS, iters=GLOBAL_ITERS if iters is None else iters)
    Nv = Nv / ratios_v(Nv, capv(th, tri), sy).max()
    lb = lb_v(Nv, W)
    t0 = time.time()
    for _ in range(LB_MAX_ROUNDS):                                      # local exact repair (same windows as the LB phase)
        if time.time() - t0 > REPAIR_SECONDS:
            break
        Nl = refine_lb_fvs(Nv, th, tri, L, W, sy, theta=0.9, dil=1, max_dof=LB_MAX_DOF)
        if Nl is None or lb_v(Nl, W) <= lb * (1 + 1e-4):
            break
        Nv, lb = Nl, lb_v(Nl, W)
    if lb > lb_res["lb"]:
        return dict(nodes=nodes, tri=tri, thick=th, stress=Nv, lb=lb)
    return lb_res


def ub_global_stage(ub_res, v, lb_res, nodes, tri, th, L, W, sy, iters=None):
    """Plan C, UB: ADMM over velocities with a plate-independent Laplacian solve (nlo_global.UBArm), warm-started from
    the hybrid's UB velocity (primal) and its best LB stress field (dual, exactly feasible). Every iterate satisfies the
    grip conditions, so the best one is a UB as it stands (the evaluator recomputes and certifies it)."""
    import nlo_global
    c = canon()
    if "global_ub" not in c:
        c["global_ub"] = nlo_global.UBArm(c["mesh"])
    arm = c["global_ub"]
    vf, y = arm.setup(L, W, th, sy, v, np.asarray(lb_res["stress"], float).mean(1), tub=ubth(th, tri))
    vb, d, _ = nlo_global.admm_ub(arm, vf, y, UB_GLOBAL_SECONDS, iters=UB_GLOBAL_ITERS if iters is None else iters)
    if d < ub_res["ub"]:
        return dict(nodes=nodes, tri=tri, thick=th, v=vb, ub=d)
    return ub_res


MIRRORS = ((False, False), (True, False), (False, True), (True, True))


def mirror_back(r, fx, fy, L, W):
    """Map a result computed on the mirrored plate back to the original plate."""
    nodes = r["nodes"].copy()
    v = r["v"].copy()
    S = r["stress"].copy()
    tri = r["tri"]
    if fx:
        nodes[:, 0] = L - nodes[:, 0]
        v[:, 0] = 1.0 - v[:, 0]
    if fy:
        nodes[:, 1] = W - nodes[:, 1]
        v[:, 1] = -v[:, 1]
    if fx != fy:                                                         # one mirror: orientation and Nxy flip
        tri = tri[:, [0, 2, 1]]
        S = S[:, [0, 2, 1], :]
        S[:, :, 2] = -S[:, :, 2]
    return dict(nodes=nodes, tri=tri, thick=r["thick"], v=v, ub=r["ub"], stress=S, lb=r["lb"])


def bounds(model, plate):
    try:
        return _bounds(model, plate)
    finally:
        _CTX.clear()


def _bounds(model, plate):
    xs, ys, t = np.asarray(plate["xs"], float), np.asarray(plate["ys"], float), np.asarray(plate["t"], float)
    L, W = xs[-1], ys[-1]
    # exact geometric mirror images: real scans have a shorter last grid step, so the grid is mirrored too
    plates = [dict(xs=L - xs[::-1] if fx else xs, ys=W - ys[::-1] if fy else ys,
                   t=t[::-1 if fx else 1, ::-1 if fy else 1], sigma_y=plate["sigma_y"]) for fx, fy in MIRRORS]
    predict = nlo_train_fvs.predict if hasattr(model, "use_fvs") else nlo_train.predict   # FVS or ZP network
    res = [mirror_back(r, fx, fy, L, W) for r, (fx, fy) in zip(predict(model, plates), MIRRORS)]
    from scipy.interpolate import RegularGridInterpolator
    interp = RegularGridInterpolator((xs, ys), t)
    for r in res:                                                        # thickness from the original grid
        r["thick"] = interp(np.clip(r["nodes"], [0, 0], [L, W]))
    best_ub = min(res, key=lambda r: r["ub"])
    best_lb = max(res, key=lambda r: r["lb"])
    ub_res = {k: best_ub[k] for k in ("nodes", "tri", "thick", "v", "ub")}
    lb_res = {k: best_lb[k] for k in ("nodes", "tri", "thick", "stress", "lb")}
    # convex combination of the 4 mirror fields on the canonical mesh (falls back to the best single field)
    c = canon()
    nodes = c["mesh"].nodes(L, W)
    tri = c["mesh"].tri
    th = res[0]["thick"]                                                 # identical node set and values
    th = nlo_train.plate_on_mesh(c["mesh"], xs, ys, t)["thick"]
    can = [to_canonical(r, L, W) for r in res]
    if THICK_MODEL == "tg":                                             # common thickness model: compare on the canonical mesh
        _CTX.update(tg_context(nodes, tri, xs, ys, t))
        sy_ = plate["sigma_y"]
        cand = [Nv / ratios_v(Nv, _CTX["tv"], sy_).max() for _, Nv in can]
        k = int(np.argmax([lb_v(Nv, W) for Nv in cand]))
        lb_res = dict(nodes=nodes, tri=tri, thick=th, stress=cand[k], lb=lb_v(cand[k], W))
        ubs = [sy_ * diss_w(vv, nodes, tri, _CTX["tub"]) for vv, _ in can]
        k = int(np.argmin(ubs))
        ub_res = dict(nodes=nodes, tri=tri, thick=th, v=can[k][0], ub=ubs[k])
    N = combine_lb([n for _, n in can], nodes, tri, th, W, plate["sigma_y"])
    if N is not None:
        Nv = N                                                           # per-vertex combined field at yield
        lb = lb_v(Nv, W)
        t_lb = time.time()
        for _ in range(LB_MAX_ROUNDS):                                   # F1: FVS window corrections, re-selected,
            if time.time() - t_lb > LB_SECONDS:                          # anytime: no new round after LB_SECONDS
                break
            Nl = refine_lb_fvs(Nv, th, tri, L, W, plate["sigma_y"], theta=0.9, dil=1, max_dof=LB_MAX_DOF)
            if Nl is None or lb_v(Nl, W) <= lb * (1 + 1e-4):
                break
            Nv, lb = Nl, lb_v(Nl, W)
        if lb > lb_res["lb"]:                                            # Nv is already at the yield surface
            lb_res = dict(nodes=nodes, tri=tri, thick=th, stress=Nv, lb=lb)
    v = combine_ub([vv for vv, _ in can], nodes, tri, th, plate["sigma_y"])
    if v is not None:
        from nlo_core import dissipation as _diss
        import torch as _t
        D = lambda vv: float(_diss(_t.tensor(vv[None]), _t.tensor(nodes[None]), _t.tensor(tri), _t.tensor(th[None]))[0])
        if THICK_MODEL == "tg":
            D = lambda vv: diss_w(vv, nodes, tri, _CTX["tub"])
        dv = D(v)
        for _ in range(UB_ROUNDS):                                       # A4: one local UB round (later rounds gain < 0.1 %)
            vl = refine_ub_local(v, nodes, tri, th, L, max_nodes=UB_MAX_NODES)
            if vl is None:
                break
            dl = D(vl)
            if dl >= dv * (1 - 1e-4):
                break
            v, dv = vl, dl
        from nlo_core import dissipation
        import torch
        ub = float(dissipation(torch.tensor(v[None]), torch.tensor(nodes[None]), torch.tensor(tri),
                               torch.tensor(th[None]))[0]) * plate["sigma_y"]
        if THICK_MODEL == "tg":
            ub = diss_w(v, nodes, tri, _CTX["tub"]) * plate["sigma_y"]
        if ub < ub_res["ub"]:
            ub_res = dict(nodes=nodes, tri=tri, thick=th, v=v, ub=ub)
    sc = 1.0                                                             # adaptive budgets from the hybrid gap
    if ADAPT_G0 and GLOBAL_ITERS and UB_GLOBAL_ITERS:
        sc = float(np.clip((1 - lb_res["lb"] / ub_res["ub"]) / ADAPT_G0, ADAPT_LO, ADAPT_HI))
    if GLOBAL_SECONDS > 0 and N is not None and v is not None:          # plan C global stage (off by default)
        lb_res = global_stage(lb_res, v, nodes, tri, th, L, W, plate["sigma_y"],
                              iters=max(1, round(GLOBAL_ITERS * sc)) if GLOBAL_ITERS else None)
    if (UB_GLOBAL_SECONDS > 0 or UB_GLOBAL_ITERS) and v is not None and np.array_equal(np.asarray(lb_res["tri"]), tri):
        ub_res = ub_global_stage(ub_res, v, lb_res, nodes, tri, th, L, W, plate["sigma_y"],
                                 iters=max(1, round(UB_GLOBAL_ITERS * sc)) if UB_GLOBAL_ITERS else None)
    if THICK_MODEL == "tg" and "q" in _CTX and np.array_equal(np.asarray(lb_res["tri"]), tri):
        from tg_capacity import tg_ratio                               # exact scaling to t_g yield (piece vertices)
        r = tg_ratio(np.asarray(lb_res["stress"], float), _CTX["q"], plate["sigma_y"])
        if 0 < r < 1:
            lb_res = dict(lb_res, stress=np.asarray(lb_res["stress"], float) / r, lb=lb_res["lb"] / r)
    return dict(ub_res=ub_res, lb_res=lb_res)
