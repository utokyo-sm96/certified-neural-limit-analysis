#!/usr/bin/env python3
"""Global stage of the certified hybrid (plan C): ADMM with exact, plate-independent equilibrium solves.

Lower-bound problem in cell-unit derivative space (second derivatives of the Airy function per constraint j):
    max P(x)  s.t.  ||G d_j(x)|| <= sy t_j,   G = M S(hx, hy)   (von Mises as a Euclidean norm of the physical stress)
The equilibrium map x -> d(x) does not depend on the plate (cell units), so its least-squares solve is set up once:
    "zp":  ZP (C1-quadratic) Airy space, the existing AiryFit banded solver (fast, ~ms per solve);
    "fvs": full FVS (C1-cubic) space, one offline sparse factorisation of K_cell^T K_cell (~10 s, once per process).
ADMM (scaled form, residual balancing):
    x <- argmin -P(x) + rho/2 ||d(x) - z + u||^2   (one least-squares solve);  z <- Proj_ellipsoids(d(x) + u);  u += d(x) - z
Warm start: primal from the hybrid's LB field, dual from its UB velocity field (exactly dual feasible by discrete virtual
work, checked to 1e-13). ADMM iterates are NOT certified (they violate yield locally); the hybrid scales the last iterate
to yield and repairs it with its local FVS windows (nlo_refine), and keeps the better certified LB. Probe history:
docs/plans/plan_C.md sections 7-8.
"""
import numpy as np
import torch

SQ3 = np.sqrt(3.0)
MV = np.array([[1.0, -0.5, 0.0], [0.0, 0.5 * SQ3, 0.0], [0.0, 0.0, SQ3]])
ALPHA = 1.6                      # ADMM over-relaxation (LB stage)
GPU_PROJECT = True               # ellipsoid projections on the GPU (float32, steering only)


class Ellipsoids:
    """Euclidean projection onto {d: ||G d|| <= rad_j} for the rows d of an array (one G, per-row radius)."""

    def __init__(self, G, rad):
        lam, Q = np.linalg.eigh(G.T @ G)
        self.lam, self.Q, self.rad, self.G = lam, Q, np.asarray(rad, float), G

    def ratio(self, d):
        return np.linalg.norm(d @ self.G.T, axis=1) / self.rad

    def project(self, w, newton=30):
        """Float32 arithmetic: the projection only steers ADMM (bounds are certified in float64 afterwards)."""
        if GPU_PROJECT and torch.cuda.is_available():
            return self._project_gpu(w, newton)
        return self._project_cpu(w, newton)

    def _project_gpu(self, w, newton=30):
        """Same safe-left Newton iteration on the GPU (float32); convergence checked every 5 steps (one sync)."""
        dev = "cuda"
        if not hasattr(self, "Qt"):
            self.Qt = torch.tensor(self.Q, dtype=torch.float32, device=dev)
            self.lamt = torch.tensor(self.lam, dtype=torch.float32, device=dev)
            self.r2t = torch.tensor(self.rad ** 2, dtype=torch.float32, device=dev)
        wt = torch.from_numpy(np.ascontiguousarray(w, dtype=np.float32)).to(dev)
        lam = self.lamt
        wh = wt @ self.Qt
        q = (wh * wh * lam).sum(1)
        idx = torch.nonzero(q > self.r2t).squeeze(1)
        if idx.numel() == 0:
            return w.copy()
        whi = wh[idx]
        a = whi * whi * lam
        rr = self.r2t[idx]
        mu = (torch.sqrt(q[idx] / rr) - 1) / lam.max()
        for k in range(newton):
            den = 1 + mu[:, None] * lam
            step = ((a / den ** 2).sum(1) - rr) / (-2 * a * lam / den ** 3).sum(1)
            mu = mu - step
            if k % 5 == 4 and not bool((step.abs() > 1e-6 * (1 + mu)).any()):
                break
        res = w.copy()
        res[idx.cpu().numpy()] = ((whi / (1 + mu[:, None] * lam)) @ self.Qt.T).cpu().numpy()
        return res

    def _project_cpu(self, w, newton=30):
        if not hasattr(self, "Q32"):
            self.Q32, self.lam32, self.r2_32 = self.Q.astype(np.float32), self.lam.astype(np.float32), (self.rad ** 2).astype(np.float32)
        wh = w.astype(np.float32) @ self.Q32
        lam, r2 = self.lam32, self.r2_32
        q = np.einsum("ij,ij->i", wh * lam, wh)
        out = q > r2
        if not out.any():
            return w.copy()
        a = wh[out] ** 2 * lam
        rr = r2[out]
        # start left of the root: mu0 = (||G w|| / r - 1) / lam_max gives phi(mu0) >= 0 (monotone Newton from the left)
        mu = (np.sqrt(q[out] / rr) - 1) / lam.max()
        act = np.arange(len(mu))
        for _ in range(newton):                                          # per-element convergence
            den = 1 + mu[act, None] * lam
            step = ((a[act] / den ** 2).sum(1) - rr[act]) / (-2 * a[act] * lam / den ** 3).sum(1)
            mu[act] -= step
            keep = np.abs(step) > 1e-6 * (1 + mu[act])
            act = act[keep]
            if not len(act):
                break
        res = w.copy()
        res[out] = (wh[out] / (1 + mu[:, None] * lam)) @ self.Q32.T
        return res


def strains(nodes, tri, v):
    """Triangle areas and (exx, eyy, 2 exy) of a linear velocity field."""
    p = nodes[tri]
    area = 0.5 * ((p[:, 1, 0] - p[:, 0, 0]) * (p[:, 2, 1] - p[:, 0, 1]) - (p[:, 1, 1] - p[:, 0, 1]) * (p[:, 2, 0] - p[:, 0, 0]))
    bx = np.stack([p[:, 1, 1] - p[:, 2, 1], p[:, 2, 1] - p[:, 0, 1], p[:, 0, 1] - p[:, 1, 1]], 1) / (2 * area[:, None])
    cy = np.stack([p[:, 2, 0] - p[:, 1, 0], p[:, 0, 0] - p[:, 2, 0], p[:, 1, 0] - p[:, 0, 0]], 1) / (2 * area[:, None])
    vx, vy = v[tri, 0], v[tri, 1]
    return area, np.stack([(bx * vx).sum(1), (cy * vy).sum(1), (cy * vx).sum(1) + (bx * vy).sum(1)], 1)


class ZPArm:
    """x = ZP coefficients; d = per-triangle (ww, uu, uw); physical N = S d, S = diag(1/hy^2, 1/hx^2, -1/(hx hy))."""

    def __init__(self, mesh):
        from nlo_core import AiryFit
        self.mesh, self.fit = mesh, AiryFit(mesh)

    def setup(self, L, W, th, sy, lb_stress, area, e, cap=None):
        NX, NY = self.mesh.NX, self.mesh.NY
        hx, hy = L / NX, W / NY
        self.S = np.array([1 / hy ** 2, 1 / hx ** 2, -1 / (hx * hy)])
        tv = th[self.mesh.tri] if cap is None else cap                   # vertex capacities (T, 3)
        self.ell = Ellipsoids(MV @ np.diag(self.S), sy * tv.min(1))
        h = np.zeros((NX, NY, 4, 3))
        h[NX - 1, :, 1, 0] = 1 / hy                                      # P = sum over right triangles of ww / hy
        self.h = h.reshape(-1, 3)
        x = self._solve(lb_stress.mean(1) / self.S)
        return x, area[:, None] * (self.S * e)

    def _solve(self, target):
        return self.fit.fit(torch.tensor(target.reshape(1, self.mesh.NX, self.mesh.NY, 4, 3)))

    def d(self, x):
        return self.mesh.airy_cell_derivs(x)[0].numpy().reshape(-1, 3)

    def update(self, z, u, rho):
        x = self._solve(z - u + self.h / rho)
        return x, self.d(x)

    def stress_v(self, dx):
        return np.repeat((dx * self.S)[:, None, :], 3, 1)


def nd_order(mesh, leaf=256):
    """Geometric nested-dissection order of the free FVS unknowns (structured grid: the unknowns on a grid line x = c or
    y = c separate the two sides, since K^T K couples only unknowns of the same cell); separators last, globals last."""
    from nlo_fvs import dof_layout
    NX, NY = mesh.NX, mesh.NY
    Vi, Hi, Vti, n, nfree = dof_layout(mesh)
    X, Y = np.full(nfree, np.nan), np.full(nfree, np.nan)
    ii, jj = np.meshgrid(np.arange(NX + 1), np.arange(NY + 1), indexing="ij")
    for k in range(3):
        m = Vi[..., k] >= 0
        X[Vi[..., k][m]], Y[Vi[..., k][m]] = ii[m], jj[m]
    hi, hj = np.meshgrid(np.arange(NX), np.arange(NY + 1), indexing="ij")
    m = Hi >= 0
    X[Hi[m]], Y[Hi[m]] = hi[m] + 0.5, hj[m]
    vi, vj = np.meshgrid(np.arange(NX + 1), np.arange(NY), indexing="ij")
    X[Vti.ravel()], Y[Vti.ravel()] = vi.ravel(), vj.ravel() + 0.5
    assert not np.isnan(X[:n]).any() and not np.isnan(Y[:n]).any()
    out = []

    def rec(idx):
        x, y = X[idx], Y[idx]
        ex, ey = x.max() - x.min(), y.max() - y.min()
        if len(idx) <= leaf or (ex < 2 and ey < 2):
            out.append(idx)
            return
        c_ax, ax = (x, 0) if ex >= ey else (y, 1)
        lo, hi_ = np.floor(c_ax.min()) + 1, np.ceil(c_ax.max()) - 1
        c = float(np.clip(np.round(np.median(c_ax)), lo, hi_))
        rec(idx[c_ax < c])
        rec(idx[c_ax > c])
        out.append(idx[c_ax == c])

    rec(np.arange(n))
    out.append(np.arange(n, nfree))
    p = np.concatenate(out)
    assert len(p) == nfree and len(np.unique(p)) == nfree
    return p



class FVSArm:
    """x = free FVS unknowns; d = K_cell x per (triangle, vertex), rows (ww, uu, -uw); S = diag(1/hy^2, 1/hx^2, 1/(hx hy))."""

    def __init__(self, mesh):
        import scipy.sparse.linalg as spla
        from nlo_fvs import FVS, fvs_stress_matrix
        self.mesh = mesh
        self.K = fvs_stress_matrix(mesh, FVS(mesh), mesh.NX, mesh.NY).tocsr()
        self.KT = self.K.T.tocsr()
        # offline, plate-independent: geometric nested dissection (fill 72M -> 33M vs COLAMD, solve 88 -> 53 ms)
        self.p = nd_order(mesh)
        A = (self.KT @ self.K).tocsc()[self.p][:, self.p].tocsc()
        self.lu = spla.splu(A, permc_spec="NATURAL", diag_pivot_thresh=0.0, options=dict(SymmetricMode=True))

    def _solve(self, b):
        x = np.empty_like(b)
        x[self.p] = self.lu.solve(b[self.p])
        return x

    def setup(self, L, W, th, sy, lb_stress, area, e, cap=None):
        NX, NY = self.mesh.NX, self.mesh.NY
        hx, hy = L / NX, W / NY
        self.S = np.array([1 / hy ** 2, 1 / hx ** 2, 1 / (hx * hy)])
        tv = th[self.mesh.tri] if cap is None else cap                   # vertex capacities (T, 3)
        self.ell = Ellipsoids(MV @ np.diag(self.S), sy * tv.reshape(-1))
        self.g = np.zeros(self.K.shape[1])
        self.g[-1] = 1 / hy                                              # P = c / hy
        x = self._solve(self.KT @ (lb_stress.reshape(-1, 3) / self.S).reshape(-1))
        y = np.repeat((area / 3)[:, None, None] * (self.S * e)[:, None, :], 3, 1).reshape(-1, 3)
        return x, y

    def d(self, x):
        return (self.K @ x).reshape(-1, 3)

    def update(self, z, u, rho):
        x = self._solve(self.KT @ (z - u).reshape(-1) + self.g / rho)
        return x, self.d(x)

    def stress_v(self, dx):
        return (dx * self.S).reshape(-1, 3, 3)


def admm(arm, x, y, seconds, iters=None):
    """Run ADMM from (x, y) for `iters` iterations (deterministic) or, if None, for `seconds`; returns the last
    iterate's physical per-vertex stresses (not certified) and the iteration count."""
    import time
    t0 = time.time()
    dx = arm.d(x)
    z = dx.copy()
    rho = np.linalg.norm(y) / max(np.linalg.norm(z), 1e-30)
    u = y / rho
    it = 0
    while (it < iters) if iters is not None else (time.time() - t0 < seconds):
        x, dx = arm.update(z, u, rho)
        z_old = z
        dxr = ALPHA * dx + (1 - ALPHA) * z_old                           # over-relaxation
        z = arm.ell.project(dxr + u)
        u = u + dxr - z
        it += 1
        if it % 5 == 0:                                                  # residual balancing (u rescaled with rho)
            pr = np.linalg.norm(dx - z) / max(np.linalg.norm(z), 1e-30)
            dr = np.linalg.norm(z - z_old) / max(np.linalg.norm(u), 1e-30)
            if pr > 10 * dr:
                rho, u = rho * 2, u / 2
            elif dr > 10 * pr:
                rho, u = rho / 2, u * 2
    return arm.stress_v(dx), it


# ----------------------------------------------------------------------------- upper bound (velocities)
class UBArm:
    """Global UB stage: min sum_T c_T ||A g_T||  s.t.  g = B v,  v = grip values on x = 0, L  (every v is admissible).

    g_T = cell-unit velocity gradient (du vx, dw vx, du vy, dw vy) on triangle T: B is plate-independent, so the
    v-update (two decoupled Laplacian-type solves on the free nodes) is factorised once. Physical strain
    e = E g, E = [[1/hx,0,0,0],[0,0,0,1/hy],[0,1/hy,1/hx,0]] (exx, eyy, 2exy); dissipation density sy t ||M^-T e||, so
    A = M^-T E (3x4) and c_T = sy * mean vertex thickness * area_T. The g-update is the prox of c||A g||, computed by the
    Moreau identity with a projection onto the (degenerate) ellipsoid {A^T s: ||s|| <= c}. Dual warm start from any
    statically admissible stress field N: y_T = area_T E^T N_T (exactly dual feasible)."""

    def __init__(self, mesh):
        import scipy.sparse as sp
        import scipy.sparse.linalg as spla
        self.mesh = mesh
        NX, NY = mesh.NX, mesh.NY
        P = mesh.unit * np.array([NX, NY])                              # node coordinates in cell units
        tri = mesh.tri
        p = P[tri]
        area = 0.5 * ((p[:, 1, 0] - p[:, 0, 0]) * (p[:, 2, 1] - p[:, 0, 1]) - (p[:, 1, 1] - p[:, 0, 1]) * (p[:, 2, 0] - p[:, 0, 0]))
        bu = np.stack([p[:, 1, 1] - p[:, 2, 1], p[:, 2, 1] - p[:, 0, 1], p[:, 0, 1] - p[:, 1, 1]], 1) / (2 * area[:, None])
        bw = np.stack([p[:, 2, 0] - p[:, 1, 0], p[:, 0, 0] - p[:, 2, 0], p[:, 1, 0] - p[:, 0, 0]], 1) / (2 * area[:, None])
        T, Nn = len(tri), len(P)
        rows, cols, vals = [], [], []
        for k in range(3):                                               # g = (du vx, dw vx, du vy, dw vy), v = (vx; vy)
            for comp, (coef, off) in enumerate(((bu, 0), (bw, 0), (bu, Nn), (bw, Nn))):
                rows.append(4 * np.arange(T) + comp)
                cols.append(tri[:, k] + off)
                vals.append(coef[:, k])
        self.B = sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(4 * T, 2 * Nn))
        u = mesh.unit[:, 0]
        self.left, self.right = u < 1e-12, u > 1 - 1e-12
        fixed = np.concatenate([self.left | self.right, self.left | self.right])
        self.free = np.nonzero(~fixed)[0]
        self.vb = np.zeros(2 * Nn)
        self.vb[:Nn][self.right] = 1.0                                   # vx = 1 on the right grip, everything else 0
        self.Bf = self.B[:, self.free].tocsc()
        self.BfT = self.Bf.T.tocsr()
        self.gb = self.B @ self.vb
        Nf = (self.BfT @ self.Bf).tocsc()
        h = len(self.free) // 2                                          # vx and vy decouple: two identical Laplacians
        assert abs(Nf[:h, h:]).max() == 0 and abs(Nf[:h, :h] - Nf[h:, h:]).max() < 1e-12
        self.h = h
        self.lu = spla.splu(Nf[:h, :h].tocsc(), permc_spec="COLAMD")
        self.cell_area = area
        self.Nn = Nn

    def setup(self, L, W, th, sy, v, Nlb, tub=None):
        NX, NY = self.mesh.NX, self.mesh.NY
        hx, hy = L / NX, W / NY
        E = np.array([[1 / hx, 0, 0, 0], [0, 0, 0, 1 / hy], [0, 1 / hy, 1 / hx, 0]])
        A = np.linalg.inv(MV).T @ E
        Uu, sig, Vt = np.linalg.svd(A, full_matrices=False)              # A = U diag(sig) Vr^T, Vr: 4x3
        self.Vr, self.sig, self.E = Vt.T, sig, E
        area = self.cell_area * hx * hy
        self.c = sy * (th[self.mesh.tri].mean(1) if tub is None else tub) * area   # tub: exact int t_g / A (t_g mode)
        self.Ein = Ellipsoids(np.diag(1 / sig), self.c)                  # {h: ||diag(1/sig) h|| <= c} in row-space coords
        self.th, self.sy, self.area = th, sy, area
        vv = np.concatenate([v[:, 0], v[:, 1]])
        y = area[:, None] * (Nlb @ E)                                     # y_T = area E^T N_T  (Nlb: per-triangle stress)
        return vv[self.free], y

    def solve(self, rhs):
        return self.lu.solve(rhs.reshape(2, self.h).T).T.reshape(-1)

    def g(self, vf):
        return (self.Bf @ vf + self.gb).reshape(-1, 4)

    def dissipation(self, gg):
        return float((self.c * np.linalg.norm(gg @ (np.linalg.inv(MV).T @ self.E).T, axis=1)).sum())

    def velocity(self, vf):
        vv = self.vb.copy()
        vv[self.free] = vf
        return np.stack([vv[:self.Nn], vv[self.Nn:]], 1)

    def prox(self, q, rho):
        """argmin_g c||A g|| + rho/2 ||g - q||^2 = q - (1/rho) Proj_C(rho q), C = {A^T s : ||s|| <= c}."""
        w = rho * q
        h = w @ self.Vr                                                  # row-space coordinates; C = Vr * {sig s}
        return q - (self.Ein.project(h) @ self.Vr.T) / rho


def admm_ub(arm, vf, y, seconds, iters=None):
    """ADMM for the UB from (v, y), `iters` iterations or `seconds`; every iterate is kinematically admissible.
    Returns the best velocity field, its dissipation and the iteration count."""
    import time
    t0 = time.time()
    gg = arm.g(vf)
    z = gg.copy()
    best_d, best_v = arm.dissipation(gg), vf.copy()
    rho = np.linalg.norm(y) / max(np.linalg.norm(z), 1e-30)
    u = y / rho
    it = 0
    while (it < iters) if iters is not None else (time.time() - t0 < seconds):
        vf = arm.solve(arm.BfT @ (z - u - arm.gb.reshape(-1, 4)).reshape(-1))
        gg = arm.g(vf)
        last = (it + 1 >= iters) if iters is not None else False
        if it % 2 == 1 or last:                                          # certified dissipation every 2nd iterate
            dd = arm.dissipation(gg)
            if dd < best_d:
                best_d, best_v = dd, vf.copy()
        z_old = z
        ggr = ALPHA * gg + (1 - ALPHA) * z_old                           # over-relaxation (as in the LB stage)
        z = arm.prox(ggr + u, rho)
        u = u + ggr - z
        it += 1
        if it % 5 == 0:
            pr = np.linalg.norm(gg - z) / max(np.linalg.norm(z), 1e-30)
            dr = np.linalg.norm(z - z_old) / max(np.linalg.norm(u), 1e-30)
            if pr > 10 * dr:
                rho, u = rho * 2, u / 2
            elif dr > 10 * pr:
                rho, u = rho / 2, u * 2
    return arm.velocity(best_v), best_d, it
