"""Full-space ZP LB re-optimisation from a warm start, by cutting planes + trust region.

N = lam N0 + A dc, dc free on the interior coefficient rows q = 2..NY-1 (edge rows zero -> traction-free),
|dc| <= rho (trust region, enlarged when active), yield cones only on an active set grown on violation."""
import time
import numpy as np
import scipy.sparse as sp
import clarabel

SQ3 = np.sqrt(3.0)


def refine_lb_full(N0, th, tri, mesh, A_cell, L, W, sy, start=3000, add=2000, rounds=8, rho=0.05, budget=20.0, top=None, mu=None):
    t_start = time.time()
    NX, NY = mesh.NX, mesh.NY
    hx, hy = L / NX, W / NY
    tmin = th[tri].min(1)
    s0 = sy * th.max()
    cid = np.arange((NX + 2) * (NY + 2)).reshape(NX + 2, NY + 2)
    cols = cid[:, 2:NY].ravel()
    D = np.tile([1 / hy ** 2, 1 / hx ** 2, 1 / (hx * hy)], len(tri))
    Asel = A_cell[:, cols]
    if top is not None:                                                  # + the 4 traction-free top-edge parameters
        Asel = sp.hstack([Asel, A_cell @ sp.csr_matrix(top)])
    Af = (sp.diags(D) @ Asel).tocsr() / s0
    nz = Af.shape[1]
    # coefficient scale: rho relative to the typical coefficient that produces a unit normalised stress
    cscale = (hy ** 2) * 1.0
    N0s = N0 / s0
    P0 = N0s[mesh.right, 0].sum() * W / NY
    g = np.asarray(Af[3 * mesh.right].sum(0)).ravel() * W / NY
    q = -np.concatenate([[P0], g])
    f0 = np.sqrt(np.maximum(N0[:, 0] ** 2 - N0[:, 0] * N0[:, 1] + N0[:, 1] ** 2 + 3 * N0[:, 2] ** 2, 0)) / (sy * tmin)
    act = np.argsort(f0)[-start:]
    best, best_P = N0 / f0.max(), P0 / f0.max()
    box = rho * cscale * th.max() / th.max()
    log = []
    for it in range(rounds):
        if time.time() - t_start > budget:
            break
        m = len(act)
        Ax, Ay, Axy = Af[3 * act], Af[3 * act + 1], Af[3 * act + 2]
        col = lambda v: sp.csr_matrix(v[:, None])
        r1 = sp.hstack([col(N0s[act, 0] - 0.5 * N0s[act, 1]), Ax - 0.5 * Ay])
        r2 = sp.hstack([col(0.5 * SQ3 * N0s[act, 1]), 0.5 * SQ3 * Ay])
        r3 = sp.hstack([col(SQ3 * N0s[act, 2]), SQ3 * Axy])
        soc = sp.vstack([sp.csr_matrix((m, 1 + nz)), -r1, -r2, -r3]).tocsr()[np.arange(4 * m).reshape(4, m).T.ravel()]
        st = clarabel.DefaultSettings(); st.verbose = False
        t1 = time.time()
        if mu is None:                                                   # box trust region on coefficients
            Ib = sp.hstack([sp.csr_matrix((nz, 1)), sp.identity(nz)])
            A = sp.vstack([Ib, -Ib, soc]).tocsc()
            b = np.concatenate([np.full(2 * nz, box), np.zeros(4 * m)])
            b[2 * nz::4] = tmin[act] / th.max()
            sol = clarabel.DefaultSolver(sp.csc_matrix((1 + nz, 1 + nz)), q, A, b,
                                         [clarabel.NonnegativeConeT(2 * nz)] + [clarabel.SecondOrderConeT(4)] * m, st).solve()
        else:                                                            # proximal step in stress space
            Pm = sp.block_diag([sp.csr_matrix((1, 1)), mu * (Af.T @ Af)]).tocsc()
            Pm = sp.triu(Pm).tocsc()
            b = np.zeros(4 * m)
            b[0::4] = tmin[act] / th.max()
            sol = clarabel.DefaultSolver(Pm, q, soc.tocsc(), b, [clarabel.SecondOrderConeT(4)] * m, st).solve()
            hit = 0.0
        ts = time.time() - t1
        if str(sol.status) not in ("Solved", "AlmostSolved"):
            log.append(f"it{it}:{sol.status}")
            break
        x = np.asarray(sol.x)
        N = (x[0] * N0s + (Af @ x[1:]).reshape(-1, 3)) * s0
        f = np.sqrt(np.maximum(N[:, 0] ** 2 - N[:, 0] * N[:, 1] + N[:, 1] ** 2 + 3 * N[:, 2] ** 2, 0)) / (sy * tmin)
        Pn = N[mesh.right, 0].sum() * W / NY / f.max() / s0
        hit = np.mean(np.abs(x[1:]) > 0.99 * box) if mu is None else 0.0
        log.append(f"it{it}: m={m} P {Pn / P0 * f0.max():.4f}x maxr {f.max():.3f} box-hit {100 * hit:.1f}% {ts:.1f}s")
        if Pn > best_P:
            best, best_P = N / f.max(), Pn
        viol = np.nonzero(f > 1 + 1e-6)[0]
        if not len(viol):
            if mu is not None or hit > 0.01:                                               # trust region active: re-centre and enlarge
                N0s = N / s0
                P0 = N0s[mesh.right, 0].sum() * W / NY
                q = -np.concatenate([[P0], g])
                f0 = f
                box *= 2
                act = np.union1d(act, np.argsort(f)[-add:])
                continue
            break
        act = np.union1d(act, viol[np.argsort(f[viol])[-add:]])
    return best, log
