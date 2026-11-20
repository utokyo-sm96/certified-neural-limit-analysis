"""GPU PDHG for the ZP lower bound, warm-started; every iterate is certifiable (any coefficients are admissible).

max_z  P(z)  s.t.  || M N_e(z) || <= sy * tmin_e   for every triangle e,
with c = E z: interior coefficient rows q = 2..NY-1 free, bottom rows 0, top rows = 4-parameter traction-free
family (AiryFit.top). N = physical ZP stresses (conv stencils). Diagonally preconditioned PDHG (Pock-Chambolle).
"""
import time
import numpy as np
import torch
import torch.nn.functional as F

SQ3 = np.sqrt(3.0)


class LBOperator:
    def __init__(self, mesh, top, L, W, dev, dtype=torch.float64):
        from nlo_core import ZP
        self.mesh, self.NX, self.NY = mesh, mesh.NX, mesh.NY
        self.w = torch.tensor(ZP.reshape(12, 1, 3, 3), dtype=dtype, device=dev)
        self.top = torch.tensor(top.reshape(mesh.NX + 2, mesh.NY + 2, 4), dtype=dtype, device=dev)   # (P, Q, 4)
        hx, hy = L / mesh.NX, W / mesh.NY
        # per-component scaling from cell-unit derivatives (ww, uu, uw) to M N = [Nxx - Nyy/2, sqrt3/2 Nyy, sqrt3 Nxy]
        a, b, cxy = 1 / hy ** 2, 1 / hx ** 2, -1 / (hx * hy)
        self.Mfull = torch.tensor([[a, -0.5 * b, 0.0], [0.0, 0.5 * SQ3 * b, 0.0], [0.0, 0.0, SQ3 * cxy]], dtype=dtype, device=dev)
        self.W, self.dev, self.dtype = W, dev, dtype

    def coeffs(self, zi, zt):
        """zi (P, NY-2) interior rows, zt (4,) top params -> c (P, Q)."""
        P, Q = self.NX + 2, self.NY + 2
        c = torch.zeros(P, Q, dtype=self.dtype, device=self.dev)
        c[:, 2:self.NY] = zi
        return c + self.top @ zt

    def coeffs_T(self, gc):
        return gc[:, 2:self.NY], torch.einsum("pqk,pq->k", self.top, gc)

    def K(self, zi, zt):
        d = F.conv2d(self.coeffs(zi, zt)[None, None], self.w)[0]          # (12, NX, NY)
        d = d.reshape(4, 3, self.NX, self.NY)                            # (k, comp, i, j)
        return torch.einsum("rc,kcij->krij", self.Mfull, d)              # M N per triangle: (4, 3, NX, NY)

    def KT(self, y):
        d = torch.einsum("rc,krij->kcij", self.Mfull, y).reshape(1, 12, self.NX, self.NY)
        gc = F.conv_transpose2d(d, self.w)[0, 0]
        return self.coeffs_T(gc)

    def stress(self, zi, zt):
        """physical N (T, 3) in mesh triangle order."""
        d = F.conv2d(self.coeffs(zi, zt)[None, None], self.w)[0].reshape(4, 3, self.NX, self.NY)
        hx, hy = self.mesh_hx, self.mesh_hy
        N = torch.stack([d[:, 0] / hy ** 2, d[:, 1] / hx ** 2, -d[:, 2] / (hx * hy)], -1)   # (4, NX, NY, 3)
        return N.permute(1, 2, 0, 3).reshape(-1, 3)


def pdhg_lb(N0, th, tri, mesh, top, A_cell, L, W, sy, iters=4000, check=200, dev="cuda:0", fit=None, v=None, nodes=None, ysc=1.0):
    import scipy.sparse as sp
    NX, NY = mesh.NX, mesh.NY
    hx, hy = L / NX, W / NY
    op = LBOperator(mesh, top, L, W, dev)
    op.mesh_hx, op.mesh_hy = hx, hy
    tmin = th[tri].min(1)
    s0 = sy * th.max()
    rad = torch.tensor((sy * tmin / s0).reshape(NX, NY, 4).transpose(2, 0, 1), dtype=torch.float64, device=dev)   # (4, NX, NY)
    # warm start: exact Airy coefficients of N0 (AiryFit reproduces any admissible ZP field)
    D = derivs(N0 / s0, hx, hy, mesh)
    c0 = fit.fit(torch.tensor(D[None]))[0].numpy()
    zi = torch.tensor(c0[:, 2:NY], device=dev)
    # top params: least-squares from the top rows (exact for an AiryFit output)
    T4 = top.reshape(NX + 2, NY + 2, 4)[:, NY:NY + 2].reshape(-1, 4)
    zt = torch.tensor(np.linalg.lstsq(T4, c0[:, NY:NY + 2].reshape(-1), rcond=None)[0], device=dev)
    # objective gradient: P = W/NY * sum over right-edge triangles of Nxx  (linear in c)
    gN = torch.zeros(4, 3, NX, NY, dtype=torch.float64, device=dev)
    gN[1, 0, NX - 1, :] = W / NY / hy ** 2                                # dP/d(ww) on right triangles (k=1)
    gc = F.conv_transpose2d(gN.reshape(1, 12, NX, NY), op.w)[0, 0]
    gi, gt = op.coeffs_T(gc)
    # diagonal preconditioners from absolute row / column sums of K
    absop = LBOperator(mesh, np.abs(top), L, W, dev)
    absop.w = op.w.abs()
    absop.Mfull = op.Mfull.abs()
    ones_i, ones_t = torch.ones_like(zi), torch.ones_like(zt)
    rowsum = absop.K(ones_i, ones_t)                                     # sum_j |K_ij|
    colsum_i, colsum_t = absop.KT(torch.ones(4, 3, NX, NY, dtype=torch.float64, device=dev))
    tau_i, tau_t = 1.0 / torch.clamp(colsum_i, min=1e-12), 1.0 / torch.clamp(colsum_t, min=1e-12)
    sig = 1.0 / torch.clamp(rowsum, min=1e-12)
    y = torch.zeros(4, 3, NX, NY, dtype=torch.float64, device=dev)
    if v is not None:                                                    # dual warm start from the UB velocity field
        p = nodes[tri]
        area = 0.5 * ((p[:, 1, 0] - p[:, 0, 0]) * (p[:, 2, 1] - p[:, 0, 1]) - (p[:, 1, 1] - p[:, 0, 1]) * (p[:, 2, 0] - p[:, 0, 0]))
        bx = np.stack([p[:, 1, 1] - p[:, 2, 1], p[:, 2, 1] - p[:, 0, 1], p[:, 0, 1] - p[:, 1, 1]], 1) / (2 * area[:, None])
        cy = np.stack([p[:, 2, 0] - p[:, 1, 0], p[:, 0, 0] - p[:, 2, 0], p[:, 1, 0] - p[:, 0, 0]], 1) / (2 * area[:, None])
        vx, vy = v[tri, 0], v[tri, 1]
        exx, eyy = (bx * vx).sum(1), (cy * vy).sum(1)
        gxy = (cy * vx).sum(1) + (bx * vy).sum(1)                        # engineering shear 2 exy
        Mm = np.array([[1, -0.5, 0], [0, 0.5 * SQ3, 0], [0, 0, SQ3]])
        Ye = (np.stack([exx, eyy, gxy], 1) * area[:, None]) @ np.linalg.inv(Mm)   # rows: (M^-T e)^T = e^T M^-1
        y = ysc * torch.tensor(Ye.reshape(NX, NY, 4, 3).transpose(2, 3, 0, 1).copy(), device=dev)
        gzi, gzt = op.KT(y)
        print(f"    dual check: |K^T y - g| / |g| = {torch.sqrt(((gzi - gi) ** 2).sum() + ((gzt - gt) ** 2).sum()) / torch.sqrt((gi ** 2).sum() + (gt ** 2).sum()):.2e}")
    best, best_lb, hist = None, -np.inf, []
    t0 = time.time()
    for it in range(iters + 1):
        if it % check == 0:
            N = op.stress(zi, zt)
            Nc = N.cpu().numpy() * s0
            f = np.sqrt(np.maximum(Nc[:, 0] ** 2 - Nc[:, 0] * Nc[:, 1] + Nc[:, 1] ** 2 + 3 * Nc[:, 2] ** 2, 0)) / (sy * tmin)
            lb = Nc[mesh.right, 0].sum() * W / NY / f.max()
            hist.append((it, time.time() - t0, lb))
            if lb > best_lb:
                best, best_lb = Nc / f.max(), lb
        gzi, gzt = op.KT(y)
        zi_n = zi - tau_i * (gzi - gi)                                   # minimise -P
        zt_n = zt - tau_t * (gzt - gt)
        ybar = y + sig * op.K(2 * zi_n - zi, 2 * zt_n - zt)
        # y <- ybar - sig * Proj_ball(ybar / sig): Moreau, ball radius rad per triangle (columns of comps)
        u = ybar / sig
        nrm = torch.sqrt((u ** 2).sum(1, keepdim=True))
        proj = u * torch.clamp(rad[:, None] / torch.clamp(nrm, min=1e-30), max=1.0)
        y = ybar - sig * proj
        zi, zt = zi_n, zt_n
    torch.cuda.synchronize()
    return best, hist


def derivs(Ns, hx, hy, mesh):
    D = np.stack([Ns[:, 0] * hy ** 2, Ns[:, 1] * hx ** 2, -Ns[:, 2] * hx * hy], -1)
    return D.reshape(mesh.NX, mesh.NY, 4, 3)
