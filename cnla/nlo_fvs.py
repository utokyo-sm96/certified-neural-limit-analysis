#!/usr/bin/env python3
"""C1-cubic Airy layer on the union-jack reference mesh: the Fraeijs de Veubeke-Sander (FVS) macro-element.

psi is a C1 piecewise cubic on the criss-cross mesh (cell split into 4 triangles by its diagonals). On this
mesh S_3^1 has dimension 5P^2 + 8P + 3 on a P x P patch, which is exactly the number of FVS degrees of freedom
(probe notes, results/probes/step1-probes-NOTES.md), so the FVS DOFs parametrise the whole space with a local
basis and no dependencies:
    V[i, j]  = (Psi, Psi_u, Psi_w)  at grid vertex (i, j),            shape (B, NX+1, NY+1, 3)
    H[i, j]  = Psi_w                 at horizontal edge mid (i+1/2, j), shape (B, NX,   NY+1)
    Vt[i, j] = Psi_u                 at vertical edge mid   (i, j+1/2), shape (B, NX+1, NY)
all in cell units (u, w) = (x / hx, y / hy). Every DOF vector gives a C1 psi, so the stresses
    Nxx = psi_yy,  Nyy = psi_xx,  Nxy = -psi_xy
are LINEAR on each triangle, equilibrated in every triangle and traction-continuous across every edge.
Free edges (exact, by fixing DOFs):
    y = 0:  all bottom DOFs zero (also removes the affine gauge);
    y = W:  V = (a + b u, b, c), H = c   (a: moment, b: shear, c: axial force);
the axial resultant is P = c / hy on every section.

The per-cell map from the 16 local DOFs to the second derivatives (ww, uu, uw) at the 3 vertices of each of the
4 triangles (36 x 16) is derived numerically below and checked (reproduction of cubics, ZP embedding). Yield
is checked at the triangle vertices, which is exact for linear stresses and linear thickness (f(N) - t is convex).
The frozen verifier eval_limit_analysis.verify already certifies per-vertex stresses.

Every ZP (C1-quadratic) field is a C1 cubic: from_zp() gives its FVS DOFs exactly, so the ZP load path of the
existing AiryFit can be kept and FVS corrections added on top.
"""
import numpy as np
import torch
import torch.nn.functional as F

from nlo_core import ZP, SQ3

# ----------------------------------------------------------------------------- polynomial helpers
MON3 = [(i, d - i) for d in range(4) for i in range(d, -1, -1)]          # x^i y^j, degree <= 3 (10)


def _mon(x, y, dx=0, dy=0, mons=MON3):
    """Row of d^(dx+dy)/dx^dx dy^dy of each monomial at (x, y)."""
    out = []
    for i, j in mons:
        if i < dx or j < dy:
            out.append(0.0)
            continue
        ci = np.prod(np.arange(i - dx + 1, i + 1)) if dx else 1.0
        cj = np.prod(np.arange(j - dy + 1, j + 1)) if dy else 1.0
        out.append(ci * cj * x ** (i - dx) * y ** (j - dy))
    return np.array(out, float)


# unit cell, triangles in RefMesh order: k = 0 bottom (a,b,m), 1 right (b,c,m), 2 top (c,d,m), 3 left (d,a,m)
_A, _B, _C, _D, _M = (0., 0.), (1., 0.), (1., 1.), (0., 1.), (.5, .5)
CELL_TRIS = [(_A, _B, _M), (_B, _C, _M), (_C, _D, _M), (_D, _A, _M)]
# local DOFs: 0-2 a, 3-5 b, 6-8 c, 9-11 d (Psi, Psi_u, Psi_w); 12 bottom mid Psi_w, 13 right mid Psi_u,
# 14 top mid Psi_w, 15 left mid Psi_u. (triangle used to evaluate each functional, point, derivative)
_DOFS = ([(0, _A, d) for d in ((0, 0), (1, 0), (0, 1))] + [(0, _B, d) for d in ((0, 0), (1, 0), (0, 1))]
         + [(2, _C, d) for d in ((0, 0), (1, 0), (0, 1))] + [(2, _D, d) for d in ((0, 0), (1, 0), (0, 1))]
         + [(0, (.5, 0.), (0, 1)), (1, (1., .5), (1, 0)), (2, (.5, 1.), (0, 1)), (3, (0., .5), (1, 0))])


def _null(A, tol=1e-10):
    _, s, Vt = np.linalg.svd(A)
    return Vt[(s > tol * s[0]).sum():]


def derive_cell():
    """Local FVS basis on the unit cell: returns (Bcoef (40, 16): monomial coefficients of the 16 basis functions on
    the 4 triangles, Mder (4, 3, 3, 16): second derivatives [k, vertex, (ww, uu, uw), dof])."""
    rows = []
    for k1, k2, p, q in ((0, 3, _A, _M), (0, 1, _B, _M), (1, 2, _C, _M), (2, 3, _D, _M)):   # interior edges
        p, q = np.array(p), np.array(q)
        for s in (0, 1 / 3, 2 / 3, 1):
            pt = p + s * (q - p)
            r = np.zeros(40)
            r[10 * k1:10 * k1 + 10], r[10 * k2:10 * k2 + 10] = _mon(*pt), -_mon(*pt)
            rows.append(r)
        for s in (0, .5, 1):
            pt = p + s * (q - p)
            for dd in ((1, 0), (0, 1)):
                r = np.zeros(40)
                r[10 * k1:10 * k1 + 10], r[10 * k2:10 * k2 + 10] = _mon(*pt, *dd), -_mon(*pt, *dd)
                rows.append(r)
    N = _null(np.array(rows))
    assert len(N) == 16, len(N)                                          # dim S_3^1 on the FVS split
    Dm = np.zeros((16, 40))
    for n, (k, pt, dd) in enumerate(_DOFS):
        Dm[n, 10 * k:10 * k + 10] = _mon(*pt, *dd)
    G = Dm @ N.T
    assert np.linalg.cond(G) < 1e8, "FVS DOFs not unisolvent"
    Bcoef = N.T @ np.linalg.inv(G)                                       # (40, 16), Dm @ Bcoef = I
    Mder = np.zeros((4, 3, 3, 16))
    for k, tv in enumerate(CELL_TRIS):
        for v, pt in enumerate(tv):
            for comp, dd in enumerate(((0, 2), (2, 0), (1, 1))):           # ww, uu, uw
                Mder[k, v, comp] = _mon(*pt, *dd) @ Bcoef[10 * k:10 * k + 10]
    return Bcoef, Mder


# ----------------------------------------------------------------------------- ZP basis function, piecewise


def _tri_of(s, t):
    """Triangle k of the unit cell containing local point (s, t) (any one if on an edge)."""
    if t <= s and t <= 1 - s:
        return 0
    if t >= s and t >= 1 - s:
        return 2
    return 1 if s > 0.5 else 3


def derive_zp_values():
    """ZP basis function of coefficient (p, q) = (3, 3) on a 5 x 5 cell patch (support: cells 1..3), as quadratic
    pieces in global patch coordinates. Returns a function Z(du, dw, dx, dy) = derivative of ZP(. - p, . - q)."""
    P = 5
    nT = 4 * P * P
    tid = lambda i, j, k: 4 * (i * P + j) + k
    rows, rhs = [], []
    # second derivatives fixed per triangle (coefficients of u^2, uw, w^2); affine parts (1, u, w) unknown
    sec = np.zeros((nT, 3))                                              # (c_uu, c_uw, c_ww) monomial coefficients
    for i in range(P):
        for j in range(P):
            a, b = 3 - i, 3 - j
            for k in range(4):
                if 0 <= a <= 2 and 0 <= b <= 2:
                    dww, duu, duw = ZP[k, 0, a, b], ZP[k, 1, a, b], ZP[k, 2, a, b]
                    sec[tid(i, j, k)] = (duu / 2, duw, dww / 2)

    # unknown x: (nT, 3) affine coefficients. Piece value = x0 + x1 u + x2 w + s0 u^2 + s1 u w + s2 w^2.
    def row(t, pt, dd):
        u, w = pt
        r = np.zeros(3 * nT)
        if dd == (0, 0):
            r[3 * t:3 * t + 3] = (1, u, w)
            c = sec[t] @ (u * u, u * w, w * w)
        elif dd == (1, 0):
            r[3 * t:3 * t + 3] = (0, 1, 0)
            c = sec[t] @ (2 * u, w, 0)
        else:
            r[3 * t:3 * t + 3] = (0, 0, 1)
            c = sec[t] @ (0, u, 2 * w)
        return r, c

    # all edges between triangles of the patch (interior edges of cells and cell-cell edges)
    edges = {}
    for i in range(P):
        for j in range(P):
            for k, tv in enumerate(CELL_TRIS):
                pts = [(i + x, j + y) for x, y in tv]
                for s in range(3):
                    key = tuple(sorted([pts[s], pts[(s + 1) % 3]]))
                    edges.setdefault(key, []).append(tid(i, j, k))
    for (p, q), ts in edges.items():
        p, q = np.array(p), np.array(q)
        if len(ts) == 2:
            pairs = [(ts[0], ts[1])]
        else:
            pairs = [(ts[0], None)]                                      # patch boundary: zero (outside support)
        for t1, t2 in pairs:
            for s in (0, .5, 1):
                pt = p + s * (q - p)
                for dd in ((0, 0), (1, 0), (0, 1)):
                    r1, c1 = row(t1, pt, dd)
                    if t2 is None:
                        rows.append(r1); rhs.append(-c1)
                    else:
                        r2, c2 = row(t2, pt, dd)
                        rows.append(r1 - r2); rhs.append(c2 - c1)
    A, b = np.array(rows), np.array(rhs)
    x, *_ = np.linalg.lstsq(A, b, rcond=None)
    res = np.abs(A @ x - b).max()
    assert res < 1e-10, f"ZP pieces inconsistent: {res}"
    x = x.reshape(nT, 3)

    def Z(du, dw, dd=(0, 0)):
        u, w = 3 + du, 3 + dw
        if not (0 <= u <= P and 0 <= w <= P):
            return 0.0
        i, j = min(int(np.floor(u)), P - 1), min(int(np.floor(w)), P - 1)
        t = tid(i, j, _tri_of(u - i, w - j))
        r, c = row(t, (u, w), dd)
        return float(r[3 * t:3 * t + 3] @ x[t] + c)
    return Z


# ----------------------------------------------------------------------------- the layer
class FVS:
    """FVS C1-cubic Airy layer on a RefMesh (NX x NY cells)."""

    def __init__(self, mesh):
        self.mesh = mesh
        self.NX, self.NY = mesh.NX, mesh.NY
        self.Bcoef, self.Mder = derive_cell()
        Z = derive_zp_values()
        # ZP coefficient c[i + a, j + b] -> DOFs at vertex (i, j), H mid (i+1/2, j), Vt mid (i, j+1/2)
        self.Kv = np.array([[[Z(-a, -b, dd) for b in range(2)] for a in range(2)]
                            for dd in ((0, 0), (1, 0), (0, 1))])            # (3, 2, 2)
        self.Kh = np.array([[Z(.5 - a, -b, (0, 1)) for b in range(2)] for a in range(3)])   # (3, 2)
        self.Kvt = np.array([[Z(-a, .5 - b, (1, 0)) for b in range(3)] for a in range(2)])  # (2, 3)
        self._t = {}

    def _tens(self, dtype, device):
        key = (dtype, str(device))
        if key not in self._t:
            T = lambda a: torch.tensor(a, dtype=dtype, device=device)
            self._t[key] = dict(M=T(self.Mder.reshape(36, 16)), Kv=T(self.Kv[:, None]), Kh=T(self.Kh[None, None]),
                                Kvt=T(self.Kvt[None, None]))
        return self._t[key]

    # ---- DOFs -> per-vertex second derivatives / stresses
    def local(self, V, H, Vt):
        """Gather the 16 local DOFs per cell: (B, NX, NY, 16)."""
        a, b, c, d = V[:, :-1, :-1], V[:, 1:, :-1], V[:, 1:, 1:], V[:, :-1, 1:]
        return torch.cat([a, b, c, d, H[:, :, :-1, None], Vt[:, 1:, :, None], H[:, :, 1:, None], Vt[:, :-1, :, None]], -1)

    def derivs(self, V, H, Vt):
        """(B, NX, NY, 4, 3, 3): (ww, uu, uw) in cell units at vertex v of triangle k of cell (i, j)."""
        t = self._tens(V.dtype, V.device)
        loc = self.local(V, H, Vt)
        return (loc @ t["M"].T).reshape(*loc.shape[:3], 4, 3, 3)

    def stress(self, V, H, Vt, L, W):
        """Physical per-vertex stress resultants (B, T, 3 vertices, 3) = (Nxx, Nyy, Nxy); L, W: (B,) tensors."""
        d = self.derivs(V, H, Vt).reshape(V.shape[0], -1, 3, 3)
        hx = (L / self.NX)[:, None, None]
        hy = (W / self.NY)[:, None, None]
        return torch.stack([d[..., 0] / hy ** 2, d[..., 1] / hx ** 2, -d[..., 2] / (hx * hy)], -1)

    def load(self, V, W):
        """Axial resultant P = c / hy, with c = Psi_w on the top edge (bottom edge Psi_w = 0)."""
        return V[:, 0, -1, 2] / (W / self.NY)

    # ---- ZP embedding (exact: C1 quadratics are C1 cubics)
    def from_zp(self, c):
        """ZP coefficients c (B, NX+2, NY+2) -> (V, H, Vt) of the same psi."""
        t = self._tens(c.dtype, c.device)
        x = c[:, None]
        V = F.conv2d(x, t["Kv"]).permute(0, 2, 3, 1)                       # (B, NX+1, NY+1, 3)
        H = F.conv2d(x, t["Kh"])[:, 0]                                     # (B, NX, NY+1)
        Vt = F.conv2d(x, t["Kvt"])[:, 0]                                   # (B, NX+1, NY)
        return V, H, Vt

    # ---- free edges
    def free_edges(self, V, H, Vt, abc=None):
        """Exact traction-free form: bottom DOFs zero; top V = (a + b u, b, c), H = c. abc (B, 3) gives the top
        parameters; if None they are read off V (least squares fit of the top row, i.e. a projection)."""
        NX = self.NX
        u = torch.arange(NX + 1, dtype=V.dtype, device=V.device)
        if abc is None:
            top = V[:, :, -1]                                              # (B, NX+1, 3)
            # least squares in (a, b) on Psi and Psi_u, c from Psi_w and H
            Am = torch.stack([torch.ones_like(u), u], 1)
            Am = torch.cat([Am, torch.stack([torch.zeros_like(u), torch.ones_like(u)], 1)], 0)   # (2(NX+1), 2)
            rhs = torch.cat([top[..., 0], top[..., 1]], 1)                 # (B, 2(NX+1))
            ab = torch.linalg.lstsq(Am[None].expand(V.shape[0], -1, -1), rhs[..., None]).solution[..., 0]
            cc = torch.cat([top[..., 2], H[:, :, -1]], 1).mean(1)
            abc = torch.stack([ab[:, 0], ab[:, 1], cc], 1)
        a, b, cc = abc[:, 0:1], abc[:, 1:2], abc[:, 2:3]
        Vtop = torch.stack([a + b * u, b.expand(-1, NX + 1), cc.expand(-1, NX + 1)], -1)
        V = torch.cat([torch.zeros_like(V[:, :, :1]), V[:, :, 1:-1], Vtop[:, :, None]], 2)
        H = torch.cat([torch.zeros_like(H[:, :, :1]), H[:, :, 1:-1], cc.expand(-1, NX)[:, :, None]], 2)
        return V, H, Vt


# ----------------------------------------------------------------------------- certification helpers
def to_res_fvs(mesh, plate, v, Nv, sigma_y):
    """numpy result for the frozen verifier: UB from v, LB from per-vertex stresses Nv (T, 3, 3) scaled to yield."""
    from nlo_core import dissipation
    v = np.asarray(v, float).copy()
    nodes, tri, th = plate["nodes"], mesh.tri, plate["thick"]
    ub = float(dissipation(torch.tensor(v[None]), torch.tensor(nodes[None]), torch.tensor(tri),
                           torch.tensor(th[None]))[0]) * sigma_y
    Nv = np.asarray(Nv, float)
    f = np.sqrt(np.maximum(Nv[..., 0] ** 2 - Nv[..., 0] * Nv[..., 1] + Nv[..., 1] ** 2 + 3 * Nv[..., 2] ** 2, 0))
    ratio = float((f / th[tri]).max())
    P = float(np.sum(0.5 * (Nv[mesh.right, 0, 0] + Nv[mesh.right, 1, 0])) * plate["W"] / mesh.NY)  # right edge b-c
    return dict(nodes=nodes, tri=tri, thick=th, v=v, ub=ub, stress=Nv * sigma_y / ratio, lb=sigma_y * P / ratio,
                status_ub="Solved", status_lb="Solved")


# ----------------------------------------------------------------------------- mesh-optimal LB in the FVS space
def dof_layout(mesh):
    """Global numbering of the free FVS unknowns. Returns (cell_dofs (NX, NY, 16) -> column index or -1 (fixed zero),
    top (NX+1 index arrays) and n_free). Unknowns: interior V rows j = 1..NY-1 (3 each), interior H rows,
    all Vt, then the 3 top parameters (a, b, c) as the last 3 columns."""
    NX, NY = mesh.NX, mesh.NY
    n = 0
    Vi = -np.ones((NX + 1, NY + 1, 3), int)
    Vi[:, 1:NY] = np.arange(n, n + (NX + 1) * (NY - 1) * 3).reshape(NX + 1, NY - 1, 3); n += (NX + 1) * (NY - 1) * 3
    Hi = -np.ones((NX, NY + 1), int)
    Hi[:, 1:NY] = np.arange(n, n + NX * (NY - 1)).reshape(NX, NY - 1); n += NX * (NY - 1)
    Vti = np.arange(n, n + (NX + 1) * NY).reshape(NX + 1, NY); n += (NX + 1) * NY
    return Vi, Hi, Vti, n, n + 3


def fvs_stress_matrix(mesh, fvs, L, W):
    """Sparse map from the free unknowns (incl. top a, b, c) to per-vertex stresses, rows 9*e + 3*v + comp."""
    import scipy.sparse as sp
    NX, NY = mesh.NX, mesh.NY
    Vi, Hi, Vti, ia, nfree = dof_layout(mesh)
    hx, hy = L / NX, W / NY
    scale = np.array([1 / hy ** 2, 1 / hx ** 2, -1 / (hx * hy)])
    I, J = np.meshgrid(np.arange(NX), np.arange(NY), indexing="ij")
    # local dof -> (global column, coefficient) incl. the top family: Psi = a + b u, Psi_u = b, Psi_w = c, H = c
    loc_cols = []                                                        # list over 16 local dofs of [(col_array, coef_array)]
    for vi, (di, dj) in enumerate(((0, 0), (1, 0), (1, 1), (0, 1))):
        ii, jj = I + di, J + dj
        for comp in range(3):
            terms = [(Vi[ii, jj, comp], np.ones(I.shape))]
            ontop = jj == NY
            if comp == 0:
                terms += [(np.where(ontop, ia, -1), np.ones(I.shape)), (np.where(ontop, ia + 1, -1), ii.astype(float))]
            elif comp == 1:
                terms += [(np.where(ontop, ia + 1, -1), np.ones(I.shape))]
            else:
                terms += [(np.where(ontop, ia + 2, -1), np.ones(I.shape))]
            loc_cols.append(terms)
    loc_cols.append([(Hi[I, J], np.ones(I.shape))])
    loc_cols.append([(Vti[I + 1, J], np.ones(I.shape))])
    loc_cols.append([(Hi[I, J + 1], np.ones(I.shape)), (np.where(J + 1 == NY, ia + 2, -1), np.ones(I.shape))])
    loc_cols.append([(Vti[I, J], np.ones(I.shape))])
    M = fvs.Mder.reshape(36, 16)
    e0 = 4 * (I * NY + J)
    rows, cols, vals = [], [], []
    for r in range(36):
        k, v, comp = r // 9, (r // 3) % 3, r % 3
        rr = 9 * (e0 + k) + 3 * v + comp
        for d in range(16):
            m = M[r, d]
            if abs(m) < 1e-14:
                continue
            for col, coef in loc_cols[d]:
                ok = col >= 0
                rows.append(rr[ok]); cols.append(col[ok]); vals.append(m * coef[ok] * scale[comp])
    return sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
                         shape=(9 * mesh.T, nfree))


def optimal_lb_fvs(mesh, plate, sigma_y, fvs=None, cap=None):
    """Best lower bound over all FVS (C1-cubic Airy) fields on this mesh (SOCP, per-vertex yield).
    cap: optional per-(triangle, vertex) capacities (T, 3) instead of the nodal thickness (e.g. t_g-safe values).
    Returns (LB, Nv (T, 3, 3), status, seconds)."""
    import time
    import clarabel
    import scipy.sparse as sp
    fvs = fvs or FVS(mesh)
    L, W, th = plate["L"], plate["W"], plate["thick"]
    T = mesh.T
    t0 = th.max()
    A_N = fvs_stress_matrix(mesh, fvs, L, W)
    nf = A_N.shape[1]
    g = np.zeros(nf)
    g[-1] = 1 / (W / mesh.NY)                                            # P = c / hy
    tv = ((th[mesh.tri] if cap is None else cap) / t0).ravel()          # per (triangle, vertex)
    Nxx, Nyy, Nxy = A_N[0::3], A_N[1::3], A_N[2::3]
    blocks = [sp.csr_matrix((3 * T, nf)), -(Nxx - 0.5 * Nyy), -(0.5 * SQ3) * Nyy, -SQ3 * Nxy]
    A = sp.vstack(blocks).tocsr()
    perm = np.arange(4 * 3 * T).reshape(4, 3 * T).T.ravel()
    A = A[perm].tocsc()
    b = np.zeros(4 * 3 * T)
    b[0::4] = tv
    st = clarabel.DefaultSettings()
    st.verbose = False
    st.direct_solve_method = "qdldl"
    tt = time.time()
    sol = clarabel.DefaultSolver(sp.csc_matrix((nf, nf)), -g, A, b, [clarabel.SecondOrderConeT(4)] * (3 * T), st).solve()
    x = np.asarray(sol.x)
    Nv = (A_N @ x).reshape(T, 3, 3)
    return float(g @ x) * sigma_y * t0, Nv * t0, str(sol.status), time.time() - tt
