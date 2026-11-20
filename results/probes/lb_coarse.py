"""Two-level LB correction prototype: N = lam N0 + A (I z), z on a coarse grid of Airy coefficients.

I interpolates coarse values bilinearly onto the interior coefficient rows q = 2..NY-1 (rows 0, 1, NY, NY+1
stay zero), so every correction is exactly equilibrated AND traction-free on y = 0, W (the free-edge
conditions only involve those edge rows). Solved by cutting planes on the yield constraints."""
import numpy as np
import scipy.sparse as sp
import clarabel

SQ3 = np.sqrt(3.0)
_I = {}


def interp_matrix(NX, NY, ncx, ncy):
    key = (NX, NY, ncx, ncy)
    if key not in _I:
        P, Q = NX + 2, NY + 2
        pu = np.linspace(0, ncx - 1, P)                                  # coarse coordinate of each fine column
        qv = np.linspace(0, ncy - 1, NY - 2)                             # interior rows 2..NY-1
        rows, cols, vals = [], [], []
        for qi, q in enumerate(range(2, NY)):
            j0 = min(int(qv[qi]), ncy - 2); fy = qv[qi] - j0
            for p in range(P):
                i0 = min(int(pu[p]), ncx - 2); fx = pu[p] - i0
                for di, wx in ((0, 1 - fx), (1, fx)):
                    for dj, wy in ((0, 1 - fy), (1, fy)):
                        if wx * wy:
                            rows.append(p * Q + q); cols.append((i0 + di) * ncy + j0 + dj); vals.append(wx * wy)
        _I[key] = sp.csr_matrix((vals, (rows, cols)), shape=(P * Q, ncx * ncy))
    return _I[key]


def refine_lb_coarse(N0, th, tri, mesh, A_cell, L, W, sy, ncx=33, ncy=11, start=2000, add=1000, rounds=10):
    NX, NY = mesh.NX, mesh.NY
    hx, hy = L / NX, W / NY
    tmin = th[tri].min(1)
    s0 = sy * th.max()
    D = np.tile([1 / hy ** 2, 1 / hx ** 2, 1 / (hx * hy)], len(tri))
    Az = (sp.diags(D) @ A_cell @ interp_matrix(NX, NY, ncx, ncy)).tocsr() / s0     # (3T, nz)
    nz = Az.shape[1]
    N0s = N0 / s0
    P0 = N0s[mesh.right, 0].sum() * W / NY
    g = np.asarray(Az[3 * mesh.right].sum(0)).ravel() * W / NY
    q = -np.concatenate([[P0], g])
    f0 = np.sqrt(np.maximum(N0[:, 0] ** 2 - N0[:, 0] * N0[:, 1] + N0[:, 1] ** 2 + 3 * N0[:, 2] ** 2, 0)) / (sy * tmin)
    act = np.argsort(f0)[-start:]
    best = N0 / f0.max()
    best_P = P0 / f0.max()
    for _ in range(rounds):
        m = len(act)
        Ax, Ay, Axy = Az[3 * act], Az[3 * act + 1], Az[3 * act + 2]
        col = lambda v: sp.csr_matrix(v[:, None])
        r1 = sp.hstack([col(N0s[act, 0] - 0.5 * N0s[act, 1]), Ax - 0.5 * Ay])
        r2 = sp.hstack([col(0.5 * SQ3 * N0s[act, 1]), 0.5 * SQ3 * Ay])
        r3 = sp.hstack([col(SQ3 * N0s[act, 2]), SQ3 * Axy])
        blocks = sp.vstack([sp.csr_matrix((m, 1 + nz)), -r1, -r2, -r3]).tocsr()
        A = blocks[np.arange(4 * m).reshape(4, m).T.ravel()].tocsc()
        b = np.zeros(4 * m)
        b[0::4] = tmin[act] / th.max()
        st = clarabel.DefaultSettings(); st.verbose = False
        sol = clarabel.DefaultSolver(sp.csc_matrix((1 + nz, 1 + nz)), q, A, b, [clarabel.SecondOrderConeT(4)] * m, st).solve()
        if str(sol.status) not in ("Solved", "AlmostSolved"):
            break
        x = np.asarray(sol.x)
        N = (x[0] * N0s + (Az @ x[1:]).reshape(-1, 3)) * s0
        f = np.sqrt(np.maximum(N[:, 0] ** 2 - N[:, 0] * N[:, 1] + N[:, 1] ** 2 + 3 * N[:, 2] ** 2, 0)) / (sy * tmin)
        Pn = N[mesh.right, 0].sum() * W / NY / f.max() / s0
        if Pn > best_P:
            best, best_P = N / f.max(), Pn
        new = np.setdiff1d(np.argsort(f)[-add:], act)
        if f.max() <= 1 + 1e-6 or not len(new):
            break
        act = np.union1d(act, new)
    return best
