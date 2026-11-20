#!/usr/bin/env python3
"""Certified layer of the neural limit operator (NLO): any network output becomes a rigorous bound.

One reference mesh serves every plate. The unit square is split into NX x NY cells, each cell into
four triangles around its centre (union-jack), and the mesh is stretched affinely onto the plate
[0, L] x [0, W]. Thickness at the nodes is the bilinear interpolation of the scan grid, exactly as
in the step-1 FE solver and its frozen verifier.

Upper bound (kinematic). The velocity v is linear on each triangle and equals (0, 0) at x = 0 and
(1, 0) at x = L (rigid welded grips). Any such v gives
    P_UB = sum_e sigma_y (2/sqrt3) |M eps(v)|_e * area_e * mean_vertex(t)      >= collapse load.

Lower bound (static), via an Airy stress function. psi is a C1 piecewise-quadratic spline on the
union-jack mesh, written in Zwart-Powell box splines psi = sum_pq c_pq ZP(u - p, w - q), where
(u, w) = (x / hx, y / hy) are cell coordinates. Its second derivatives are constant on each
triangle, so the stresses
    Nxx = psi_yy,  Nyy = psi_xx,  Nxy = -psi_xy
are piecewise constant, satisfy equilibrium in every triangle and are traction-continuous across
every edge, for ANY coefficients c (C1 continuity of psi). Translates of ZP span the whole C1
quadratic spline space on this mesh, i.e. every equilibrated piecewise-constant field.
Traction-free edges y = 0 and y = W (Nyy = Nxy = 0 on the triangles that touch them) are imposed
by an exact linear projection of c, which is small because it only involves the rows next to
those edges. Yield: constant N against the smallest vertex thickness of the triangle, which is
exact for linear t. Then
    P_LB = (right-grip resultant of N) / max_e f(N_e) / (sigma_y t_min,e)      <= collapse load.

Everything here is differentiable torch code (float32 for training, float64 for certification),
and to_res() hands a plain numpy result to the frozen step-1 verifier eval_limit_analysis.verify.
"""
import numpy as np
import torch
import torch.nn.functional as F
from scipy.interpolate import RegularGridInterpolator

SQ3 = np.sqrt(3.0)

# ZP[k, comp, a, b]: second derivative `comp` (0: psi_ww, 1: psi_uu, 2: psi_uw, in cell units) on
# triangle k (0 bottom, 1 right, 2 top, 3 left) of cell (i, j), from coefficient c[i + a, j + b].
# Derived numerically (null space of the C1 conditions on a 5 x 5 patch, zp_derive in the notes)
# and checked: exact reproduction of all quadratics, partition of unity.
ZP = np.array(
    [[[[0.5, -0.5, 0.0], [0.0, -1.0, 1.0], [0.5, -0.5, 0.0]],
      [[0.5, 0.5, 0.0], [-1.0, -1.0, 0.0], [0.5, 0.5, 0.0]],
      [[0.5, -0.5, 0.0], [0.0, 0.0, 0.0], [-0.5, 0.5, 0.0]]],
     [[[0.0, 0.0, 0.0], [0.5, -1.0, 0.5], [0.5, -1.0, 0.5]],
      [[0.0, 1.0, 0.0], [-0.5, -1.0, -0.5], [0.5, 0.0, 0.5]],
      [[0.0, 0.0, 0.0], [0.5, 0.0, -0.5], [-0.5, 0.0, 0.5]]],
     [[[0.0, -0.5, 0.5], [1.0, -1.0, 0.0], [0.0, -0.5, 0.5]],
      [[0.0, 0.5, 0.5], [0.0, -1.0, -1.0], [0.0, 0.5, 0.5]],
      [[0.0, 0.5, -0.5], [0.0, 0.0, 0.0], [0.0, -0.5, 0.5]]],
     [[[0.5, -1.0, 0.5], [0.5, -1.0, 0.5], [0.0, 0.0, 0.0]],
      [[0.5, 0.0, 0.5], [-0.5, -1.0, -0.5], [0.0, 1.0, 0.0]],
      [[0.5, 0.0, -0.5], [-0.5, 0.0, 0.5], [0.0, 0.0, 0.0]]]])


class RefMesh:
    """Union-jack mesh of the unit square with NX x NY cells.
    Nodes: grid nodes (i, j) -> i*(NY+1) + j, then cell centres (i, j) -> G + i*NY + j.
    Triangles: e = 4*(i*NY + j) + k, k = 0 bottom, 1 right, 2 top, 3 left, counter-clockwise."""

    def __init__(self, NX=128, NY=40):
        self.NX, self.NY = NX, NY
        G = (NX + 1) * (NY + 1)
        self.G = G
        gi = np.arange(G).reshape(NX + 1, NY + 1)
        I, J = np.meshgrid(np.arange(NX), np.arange(NY), indexing="ij")
        a, b, c, d = gi[I, J], gi[I + 1, J], gi[I + 1, J + 1], gi[I, J + 1]
        m = G + I * NY + J
        self.tri = np.stack([np.stack(t, -1) for t in ((a, b, m), (b, c, m), (c, d, m), (d, a, m))], 2).reshape(-1, 3)
        u, w = np.meshgrid(np.arange(NX + 1) / NX, np.arange(NY + 1) / NY, indexing="ij")
        uc, wc = np.meshgrid((np.arange(NX) + 0.5) / NX, (np.arange(NY) + 0.5) / NY, indexing="ij")
        self.unit = np.concatenate([np.stack([u.ravel(), w.ravel()], 1), np.stack([uc.ravel(), wc.ravel()], 1)])
        self.unit[gi[NX, :], 0] = 1.0                                    # exact edges
        self.unit[gi[:, NY], 1] = 1.0
        cell = np.arange(NX * NY).reshape(NX, NY)
        self.right = 4 * cell[NX - 1, :] + 1                             # triangles on x = L
        self.bottom = 4 * cell[:, 0] + 0                                 # triangles on y = 0
        self.top = 4 * cell[:, NY - 1] + 2                               # triangles on y = W
        self._edge = {}

    @property
    def T(self):
        return len(self.tri)

    def nodes(self, L, W):
        return self.unit * np.array([L, W])

    # ---------------------------------------------------------------- free-edge projection of c
    def edge_basis(self):
        """Orthonormal basis of the free-edge constraints {psi_uu = psi_uw = 0 on the triangles touching y = 0
        and y = W}. The constraints only involve the three coefficient rows next to each edge, so the basis is
        computed per edge on those columns. Returns (cols, Q): column indices into c.ravel() and Q (r, len(cols))."""
        if "basis" not in self._edge:
            NX, NY = self.NX, self.NY
            cid = np.arange((NX + 2) * (NY + 2)).reshape(NX + 2, NY + 2)
            cols, blocks = [], []
            for j, k in ((0, 0), (NY - 1, 2)):
                cc = cid[:, j:j + 3].ravel()
                pos = {c: n for n, c in enumerate(cc)}
                B = np.zeros((2 * NX, len(cc)))
                for comp in (1, 2):
                    for i in range(NX):
                        for a in range(3):
                            for b in range(3):
                                B[(comp - 1) * NX + i, pos[cid[i + a, j + b]]] += ZP[k, comp, a, b]
                _, sv, Vt = np.linalg.svd(B, full_matrices=False)
                blocks.append(Vt[sv > 1e-12 * sv[0]])
                cols.append(cc)
            off = np.cumsum([0] + [len(c) for c in cols])
            Q = np.zeros((sum(len(b) for b in blocks), off[-1]))
            r = 0
            for b, o in zip(blocks, off):
                Q[r:r + len(b), o:o + b.shape[1]] = b
                r += len(b)
            self._edge["basis"] = (np.concatenate(cols), Q)
        return self._edge["basis"]

    def edge_projector(self, device="cpu"):
        key = str(device)
        if key not in self._edge:
            cols, Q = self.edge_basis()
            self._edge[key] = (torch.tensor(cols, device=device), torch.tensor(Q, device=device))
        return self._edge[key]

    def project_edges(self, c):
        """Exact orthogonal projection of c (batch, NX+2, NY+2) onto the traction-free-edge subspace."""
        cols, Q = self.edge_projector(c.device)
        Q = Q.to(c.dtype)
        flat = c.reshape(c.shape[0], -1)
        sub = flat[:, cols]
        return flat.index_copy(1, cols, sub - (sub @ Q.T) @ Q).reshape(c.shape)

    # ---------------------------------------------------------------- Airy stresses
    def airy_cell_derivs(self, c):
        """c: (batch, NX+2, NY+2) -> (batch, NX, NY, 4, 3) second derivatives (ww, uu, uw) in cell units."""
        w = torch.tensor(ZP.reshape(12, 1, 3, 3), dtype=c.dtype, device=c.device)
        out = F.conv2d(c[:, None], w)                                      # (batch, 12, NX, NY)
        return out.permute(0, 2, 3, 1).reshape(c.shape[0], self.NX, self.NY, 4, 3)

    def stress(self, c, L, W):
        """Physical stress resultants per triangle (batch, T, 3) = (Nxx, Nyy, Nxy); L, W: (batch,) tensors."""
        d = self.airy_cell_derivs(c).reshape(c.shape[0], -1, 3)
        hx = (L / self.NX)[:, None]
        hy = (W / self.NY)[:, None]
        return torch.stack([d[..., 0] / hy ** 2, d[..., 1] / hx ** 2, -d[..., 2] / (hx * hy)], -1)

    def base_coeffs(self, dtype=torch.float64, device="cpu"):
        """c for psi = w^2 / 2 (uniform Nxx = 1 / hy^2, Nyy = Nxy = 0), already traction-free."""
        q = torch.arange(self.NY + 2, dtype=dtype, device=device)
        return (0.5 * q ** 2)[None, :].expand(self.NX + 2, self.NY + 2).clone()


# -------------------------------------------------------------------- plate data on the mesh
def plate_on_mesh(mesh, xs, ys, t):
    """Thickness at the mesh nodes for one plate (numpy float64), plus geometry."""
    xs, ys, t = np.asarray(xs, float), np.asarray(ys, float), np.asarray(t, float)
    L, W = float(xs[-1]), float(ys[-1])
    nodes = mesh.nodes(L, W)
    th = RegularGridInterpolator((xs, ys), t)(np.clip(nodes, [0, 0], [L, W]))
    return dict(L=L, W=W, nodes=nodes, thick=th)


def geometry_torch(nodes, tri):
    """nodes (batch, N, 2), tri (T, 3) long -> area (batch, T), bx, cy (batch, T, 3) shape-function gradients."""
    p = nodes[:, tri]                                                    # (batch, T, 3, 2)
    x, y = p[..., 0], p[..., 1]
    area = 0.5 * ((x[..., 1] - x[..., 0]) * (y[..., 2] - y[..., 0]) - (y[..., 1] - y[..., 0]) * (x[..., 2] - x[..., 0]))
    bx = torch.stack([y[..., 1] - y[..., 2], y[..., 2] - y[..., 0], y[..., 0] - y[..., 1]], -1) / (2 * area[..., None])
    cy = torch.stack([x[..., 2] - x[..., 1], x[..., 0] - x[..., 2], x[..., 1] - x[..., 0]], -1) / (2 * area[..., None])
    return area, bx, cy


def dissipation(v, nodes, tri, thick):
    """Upper bound / sigma_y for velocities v (batch, N, 2): sum (2/sqrt3) |M eps| area mean(t).  Units: mm^2."""
    area, bx, cy = geometry_torch(nodes, tri)
    vx, vy = v[:, tri, 0], v[:, tri, 1]
    exx, eyy = (bx * vx).sum(-1), (cy * vy).sum(-1)
    exy = 0.5 * ((cy * vx).sum(-1) + (bx * vy).sum(-1))
    dens = torch.sqrt(exx ** 2 + eyy ** 2 + exx * eyy + exy ** 2 + 1e-30)
    return ((2 / SQ3) * dens * area * thick[:, tri].mean(-1)).sum(-1)


def yield_ratio(N, tmin):
    """f(N) / t_min per triangle (sigma_y = 1); N (batch, T, 3), tmin (batch, T)."""
    Nxx, Nyy, Nxy = N[..., 0], N[..., 1], N[..., 2]
    f = torch.sqrt(torch.clamp(Nxx ** 2 - Nxx * Nyy + Nyy ** 2 + 3 * Nxy ** 2, min=0.0) + 1e-30)
    return f / tmin


def resultant(mesh, N, W):
    """x-resultant on the right grip, (batch,): sum over right-edge triangles of Nxx * hy."""
    return N[:, mesh.right, 0].sum(-1) * (W / mesh.NY)


def to_res(mesh, plate, v, N, sigma_y):
    """numpy float64 result for the frozen verifier: UB from v, LB from N scaled to the yield surface."""
    v = np.asarray(v, float).copy()
    nodes, tri, th = plate["nodes"], mesh.tri, plate["thick"]
    tn = torch.tensor(nodes[None]), torch.tensor(tri), torch.tensor(th[None])
    ub = float(dissipation(torch.tensor(v[None]), tn[0], tn[1], tn[2])[0]) * sigma_y
    Nt = torch.tensor(np.asarray(N, float)[None])
    tmin = torch.tensor(th[tri].min(1)[None])
    ratio = float(yield_ratio(Nt, tmin).max())
    P = float(resultant(mesh, Nt, torch.tensor([plate["W"]], dtype=torch.float64))[0])
    S = np.repeat((np.asarray(N, float) * sigma_y / ratio)[:, None, :], 3, axis=1)   # (T, 3 vertices, 3)
    return dict(nodes=nodes, tri=tri, thick=th, v=v, ub=ub, stress=S, lb=sigma_y * P / ratio,
                status_ub="Solved", status_lb="Solved")


# -------------------------------------------------------------------- mesh-optimal references (Clarabel)
def zp_stress_matrix(mesh, L, W):
    """Sparse map c (flattened (NX+2)*(NY+2)) -> N (T*3) for a plate of size L x W."""
    import scipy.sparse as sp
    NX, NY = mesh.NX, mesh.NY
    hx, hy = L / NX, W / NY
    cid = np.arange((NX + 2) * (NY + 2)).reshape(NX + 2, NY + 2)
    I, J = np.meshgrid(np.arange(NX), np.arange(NY), indexing="ij")
    e0 = 4 * (I * NY + J)
    scale = (1 / hy ** 2, 1 / hx ** 2, -1 / (hx * hy))
    rows, cols, vals = [], [], []
    for k in range(4):
        for comp in range(3):
            for a in range(3):
                for b in range(3):
                    s = ZP[k, comp, a, b]
                    if s != 0:
                        rows.append((3 * (e0 + k) + comp).ravel())
                        cols.append(cid[I + a, J + b].ravel())
                        vals.append(np.full(I.size, s * scale[comp]))
    return sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
                         shape=(3 * mesh.T, (NX + 2) * (NY + 2)))


def optimal_lb_zp(mesh, plate, sigma_y):
    """Best lower bound over all ZP-Airy fields on this mesh (SOCP). Returns (LB, N (T,3), status, seconds)."""
    import time
    import clarabel
    import scipy.sparse as sp
    L, W, th = plate["L"], plate["W"], plate["thick"]
    T = mesh.T
    t0 = th.max()
    A_N = zp_stress_matrix(mesh, L, W)
    cols, Q = mesh.edge_basis()
    Bm = sp.csr_matrix((Q.ravel(), (np.repeat(np.arange(len(Q)), Q.shape[1]), np.tile(cols, len(Q)))),
                       shape=(len(Q), (mesh.NX + 2) * (mesh.NY + 2)))
    nc = A_N.shape[1]
    g = np.asarray(A_N[3 * mesh.right].sum(0)).ravel() * (W / mesh.NY)
    tmin = th[mesh.tri].min(1) / t0
    # cone rows: s - A x in K with s = [tmin, 0, 0, 0]; A rows: [0, -(Nxx - Nyy/2), -(sqrt3/2) Nyy, -sqrt3 Nxy]
    Nxx, Nyy, Nxy = A_N[0::3], A_N[1::3], A_N[2::3]
    blocks = [sp.csr_matrix((T, nc)), -(Nxx - 0.5 * Nyy), -(0.5 * SQ3) * Nyy, -SQ3 * Nxy]
    A_soc = sp.vstack(blocks).tocsr()
    perm = np.arange(4 * T).reshape(4, T).T.ravel()                     # interleave per triangle
    A_soc = A_soc[perm]
    b_soc = np.zeros(4 * T)
    b_soc[0::4] = tmin
    A = sp.vstack([Bm, A_soc]).tocsc()
    b = np.concatenate([np.zeros(Bm.shape[0]), b_soc])
    st = clarabel.DefaultSettings()
    st.verbose = False
    st.direct_solve_method = "qdldl"
    cones = [clarabel.ZeroConeT(Bm.shape[0])] + [clarabel.SecondOrderConeT(4)] * T
    tt = time.time()
    sol = clarabel.DefaultSolver(sp.csc_matrix((nc, nc)), -g, A, b, cones, st).solve()
    x = np.asarray(sol.x)
    N = (A_N @ x).reshape(T, 3)
    return float(g @ x) * sigma_y * t0, N, str(sol.status), time.time() - tt


def optimal_ub(mesh, plate, sigma_y):
    """Best upper bound over all piecewise-linear v on this mesh (step-1 SOCP). Returns (UB, v, status, seconds)."""
    from limit_analysis import upper_bound
    th = plate["thick"]
    t0 = th.max()
    r = upper_bound(plate["nodes"], mesh.tri, th, t0, plate["L"], "rigid", 1e-6 * max(plate["L"], plate["W"]))
    return r["ub_hat"] * sigma_y * t0, r["v"], r["status"], r["time"]


# -------------------------------------------------------------------- least-squares Airy layer
class AiryFit:
    """Maps any target stress field N* to the closest exactly-equilibrated, traction-free ZP-Airy field.

    Work in derivative space D = (psi_ww, psi_uu, psi_uw) per triangle (cell units), with
    D = (Nxx hy^2, Nyy hx^2, -Nxy hx hy). The traction-free conditions on y = 0 and y = W reduce,
    row by row, to: (sum of the two edge coefficient rows) affine in p, (difference) = alpha + beta (-1)^p.
    On the bottom edge those 4 parameters are exactly the null space of the Airy map (affine functions
    plus the Zwart-Powell dependency), so the bottom rows are set to zero (gauge). The top rows keep
    their 4 parameters, so every c produced here satisfies both free edges EXACTLY, without projection.
    Interior rows q = 2..NY-1 are free. The normal equations are banded (bandwidth 2(NY-2)+2) plus
    4 dense columns, solved by a banded Cholesky factor and a 4 x 4 Schur complement (float64, CPU).
    fit() is a fixed linear map, differentiable (custom backward = the same symmetric solve)."""

    def __init__(self, mesh):
        import scipy.sparse as sp
        from scipy.linalg import cholesky_banded
        self.mesh = mesh
        NX, NY = mesh.NX, mesh.NY
        nc = (NX + 2) * (NY + 2)
        cid = np.arange(nc).reshape(NX + 2, NY + 2)
        A = zp_stress_matrix(mesh, NX, NY).tocsc()                       # (ww, uu, -uw); signs cancel in A^T A
        self.inner = cid[:, 2:NY].ravel()                                # interior coefficients, p-major
        p = np.arange(NX + 2, dtype=float)
        top = np.zeros((nc, 4))                                          # d = 1, p ; e = 1, (-1)^p
        for k, (dv, ev) in enumerate(((np.ones_like(p), 0 * p), (p, 0 * p), (0 * p, np.ones_like(p)), (0 * p, (-1) ** p))):
            top[cid[:, NY], k] = 0.5 * (dv + ev)
            top[cid[:, NY + 1], k] = 0.5 * (dv - ev)
        self.top = top
        Ai = A[:, self.inner]
        At = A @ top
        Kii = (Ai.T @ Ai).todia()
        u = 2 * (NY - 2) + 2
        assert Kii.offsets.max() <= u
        ab = np.zeros((u + 1, Kii.shape[0]))
        for off, dat in zip(Kii.offsets, Kii.data):
            if 0 <= off:
                ab[u - off, off:] = dat[off:]
        self.cb = cholesky_banded(ab)
        self.Kit = np.asarray(Ai.T @ At)                                 # (ni, 4)
        self.X = self._band(self.Kit)                                    # Kii^-1 Kit
        Ktt = np.asarray(At.T @ At)
        self.Sinv = np.linalg.inv(Ktt - self.Kit.T @ self.X)
        self._t = {}

    def _band(self, r):
        from scipy.linalg import cho_solve_banded
        return cho_solve_banded((self.cb, False), r)

    def solve_np(self, r):
        """r = E^T A^T D* as (n_inner + 4, B) -> z (same shape)."""
        ri, rt = r[:-4], r[-4:]
        y = self._band(ri)
        zt = self.Sinv @ (rt - self.Kit.T @ y)
        return np.concatenate([y - self.X @ zt, zt])

    def _tensors(self, device, dtype):
        key = (str(device), dtype)
        if key not in self._t:
            self._t[key] = (torch.tensor(self.inner, device=device), torch.tensor(self.top, device=device, dtype=dtype))
        return self._t[key]

    def fit(self, D):
        """D: (B, NX, NY, 4, 3) target derivatives (ww, uu, uw) -> c (B, NX+2, NY+2), differentiable."""
        mesh = self.mesh
        B = D.shape[0]
        w = torch.tensor(ZP.reshape(12, 1, 3, 3), dtype=D.dtype, device=D.device)
        r = F.conv_transpose2d(D.reshape(B, mesh.NX, mesh.NY, 12).permute(0, 3, 1, 2), w)[:, 0]   # A^T D
        inner, top = self._tensors(D.device, D.dtype)
        flat = r.reshape(B, -1)
        rz = torch.cat([flat[:, inner], flat @ top], 1)                 # E^T A^T D
        z = _AirySolve.apply(rz, self)
        c = torch.zeros_like(flat)
        c = c.index_copy(1, inner, z[:, :-4]) + z[:, -4:] @ top.T
        return c.reshape(B, mesh.NX + 2, mesh.NY + 2)


class _AirySolve(torch.autograd.Function):
    @staticmethod
    def forward(ctx, rz, fitter):
        ctx.fitter = fitter
        z = fitter.solve_np(rz.detach().double().cpu().numpy().T).T
        return torch.from_numpy(np.ascontiguousarray(z)).to(rz.device, rz.dtype)

    @staticmethod
    def backward(ctx, g):
        return _AirySolve.apply(g, ctx.fitter), None


def derivs_from_stress(N, L, W, mesh):
    """Physical (B, T, 3) target stresses -> (B, NX, NY, 4, 3) derivative-space target (ww, uu, uw)."""
    hx, hy = (L / mesh.NX)[:, None], (W / mesh.NY)[:, None]
    D = torch.stack([N[..., 0] * hy ** 2, N[..., 1] * hx ** 2, -N[..., 2] * hx * hy], -1)
    return D.reshape(N.shape[0], mesh.NX, mesh.NY, 4, 3)
