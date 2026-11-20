"""Plan C, probe C0: PDLP-style restarted PDHG for the FVS lower bound on the GPU (float32), warm-started from the
certified hybrid (primal: its LB field; dual: its UB velocity field, exactly dual feasible by virtual work).

LB problem (normalised by sy*t0):  max g.x  s.t. ||(K x)_j|| <= r_j  for every (triangle, vertex) j,
K = blockdiag(M) A / (sy t0), A = FVS DOFs -> per-vertex stresses, M: von Mises as a Euclidean norm, r_j = t_j / t0.
Any x is equilibrated, so every iterate gives a certified LB = sy t0 (g.x) / max_j ||(Kx)_j|| / r_j.
Dual: min sum_j r_j ||y_j|| s.t. K^T y = g. From a velocity field v: y_(T,v) = (area_T/3) M^-T e_T(v) is exactly
feasible and its objective is the dissipation (= the hybrid's UB).

Usage: python c0_pdlp.py --plates syn0903,syn0769 --seconds 10 [--extra 20] --out <csv>
"""
import argparse
import csv
import os
import sys
import time

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import torch

PF = '/home/oldlinux/Research_Projects/Corrosion/PaperX/cnla'
sys.path.insert(0, PF)
import nlo_refine as R  # noqa: E402
import nlo_train_fvs  # noqa: E402
from nlo_core import plate_on_mesh  # noqa: E402
from nlo_fvs import FVS, fvs_stress_matrix, to_res_fvs  # noqa: E402
from nlo_refine_eval import certify_lb  # noqa: E402

SQ3 = np.sqrt(3.0)
MV = np.array([[1.0, -0.5, 0.0], [0.0, 0.5 * SQ3, 0.0], [0.0, 0.0, SQ3]])
DEV = "cuda:0"


def ruiz_pc(K, iters=10):
    """Ruiz equilibration (inf norms, one factor per 3-row cone block) + Pock-Chambolle (alpha = 1)."""
    m = K.shape[0] // 3
    d1, d2 = np.ones(m), np.ones(K.shape[1])
    Kc = K.tocsr()
    for it in range(iters + 1):
        Ka = abs(Kc)
        if it < iters:
            rb = np.asarray(Ka.max(axis=1).todense()).ravel().reshape(m, 3).max(1)
            cb = np.asarray(Ka.max(axis=0).todense()).ravel()
        else:                                                            # Pock-Chambolle, alpha = 1
            rb = np.asarray(Ka.sum(axis=1)).ravel().reshape(m, 3).max(1)
            cb = np.asarray(Ka.sum(axis=0)).ravel()
        s1, s2 = 1 / np.sqrt(np.maximum(rb, 1e-30)), 1 / np.sqrt(np.maximum(cb, 1e-30))
        Kc = (sp.diags(np.repeat(s1, 3)) @ Kc @ sp.diags(s2)).tocsr()
        d1, d2 = d1 * s1, d2 * s2
    return Kc, d1, d2


def to_torch(A, dtype=torch.float32):
    A = A.tocsr()
    return torch.sparse_csr_tensor(torch.tensor(A.indptr), torch.tensor(A.indices), torch.tensor(A.data),
                                   size=A.shape, dtype=dtype, device=DEV)


def setup(plate, out, mesh, fvs):
    """Matrices, scaling and the warm start from the hybrid result `out` (lb_res, ub_res on the canonical mesh)."""
    xs, ys, t, sy = plate["xs"], plate["ys"], plate["t"], plate["sigma_y"]
    pl = plate_on_mesh(mesh, xs, ys, t)
    L, W, th, tri = pl["L"], pl["W"], pl["thick"], mesh.tri
    t0 = th.max()
    A = fvs_stress_matrix(mesh, fvs, L, W).tocsr()                       # (9T, n) physical per-vertex stresses
    T = len(tri)
    K = (sp.kron(sp.identity(3 * T, format="csr"), sp.csr_matrix(MV)) @ A / (sy * t0)).tocsr()
    n = A.shape[1]
    g = np.zeros(n)
    g[-1] = 1.0 / (W / mesh.NY) / (sy * t0)                              # P = c / hy
    r = (th[tri] / t0).ravel()                                           # ball radius per (triangle, vertex)
    # primal warm start: DOFs of the hybrid's LB field (least squares, exact up to round-off)
    lb, ub = out["lb_res"], out["ub_res"]
    assert np.array_equal(lb["tri"], tri) and np.array_equal(ub["tri"], tri), "hybrid result not on the canonical mesh"
    s = np.asarray(lb["stress"], float).reshape(-1)
    ts = time.time()
    lu = spla.splu((A.T @ A).tocsc(), permc_spec="COLAMD")
    x0 = lu.solve(A.T @ s)
    t_rec = time.time() - ts
    rec_err = np.linalg.norm(A @ x0 - s) / np.linalg.norm(s)
    # dual warm start from the UB velocity field
    nodes, v = ub["nodes"], ub["v"]
    p = nodes[tri]
    area = 0.5 * ((p[:, 1, 0] - p[:, 0, 0]) * (p[:, 2, 1] - p[:, 0, 1]) - (p[:, 1, 1] - p[:, 0, 1]) * (p[:, 2, 0] - p[:, 0, 0]))
    bx = np.stack([p[:, 1, 1] - p[:, 2, 1], p[:, 2, 1] - p[:, 0, 1], p[:, 0, 1] - p[:, 1, 1]], 1) / (2 * area[:, None])
    cy = np.stack([p[:, 2, 0] - p[:, 1, 0], p[:, 0, 0] - p[:, 2, 0], p[:, 1, 0] - p[:, 0, 0]], 1) / (2 * area[:, None])
    vx, vy = v[tri, 0], v[tri, 1]
    e = np.stack([(bx * vx).sum(1), (cy * vy).sum(1), (cy * vx).sum(1) + (bx * vy).sum(1)], 1)   # (exx, eyy, 2 exy)
    yT = (area / 3)[:, None] * (e @ np.linalg.inv(MV))                   # (area/3) M^-T e  (row-vector form)
    y0 = np.repeat(yT[:, None, :], 3, 1).reshape(-1)
    dres = np.linalg.norm(K.T @ y0 - g) / np.linalg.norm(g)
    ub_dual = sy * t0 * float(np.sum(r * np.linalg.norm(y0.reshape(-1, 3), axis=1)))
    Ks, d1, d2 = ruiz_pc(K)
    return dict(sy=sy, t0=t0, A=A, K=K, Ks=Ks, d1=d1, d2=d2, g=g, r=r, x0=x0, y0=y0, pl=pl, t_rec=t_rec,
                rec_err=rec_err, dres=dres, ub_dual=ub_dual, ub=ub["ub"], lb=lb["lb"])


def pdlp(S, seconds, check=64, extra=0.0):
    """Restarted PDHG (KKT-based restarts to the average, primal-weight updates). Returns a trace of
    (time, certified LB of the best iterate so far, KKT) and the best x (unscaled, float64)."""
    f32 = torch.float32
    Ks, KsT = to_torch(S["Ks"]), to_torch(S["Ks"].T)
    d1, d2 = S["d1"], S["d2"]
    gt = torch.tensor(S["g"] * d2, dtype=f32, device=DEV)
    rt = torch.tensor(S["r"] * d1, dtype=f32, device=DEV)
    m = len(S["r"])
    x = torch.tensor(S["x0"] / d2, dtype=f32, device=DEV)
    y = torch.tensor(S["y0"] / np.repeat(d1, 3), dtype=f32, device=DEV)
    # step size from ||Ks||_2 (power iteration)
    u = torch.randn(Ks.shape[1], dtype=f32, device=DEV)
    for _ in range(40):
        u = KsT @ (Ks @ u)
        nrm = u.norm()
        u = u / nrm
    eta = 0.95 / float(torch.sqrt(nrm))
    omega = float(gt.norm() / rt.norm())

    def proj(w):                                                         # projection onto the product of balls
        w3 = w.view(m, 3)
        nn = w3.norm(dim=1).clamp_min(1e-30)
        return (w3 * torch.clamp(rt / nn, max=1.0)[:, None]).reshape(-1)

    def kkt(xx, yy):
        Kx = Ks @ xx
        pr = (Kx - proj(Kx)).norm()
        dr = (KsT @ yy - gt).norm()
        gap = (rt * yy.view(m, 3).norm(dim=1)).sum() - gt @ xx
        return float(torch.sqrt(pr ** 2 + dr ** 2 + gap ** 2)), Kx

    def cert(xx, Kx):                                                    # certified LB (N), any x
        ratio = float((Kx.view(m, 3).norm(dim=1) / rt).max())
        return S["sy"] * S["t0"] * float(gt @ xx) / ratio

    k0, Kx0 = kkt(x, y)
    best = cert(x, Kx0)
    best_x = x.clone()
    trace = [(0.0, best, k0, 0)]
    xr, yr, k_last, k_prev = x.clone(), y.clone(), k0, np.inf
    xs_, ys_, nav = torch.zeros_like(x), torch.zeros_like(y), 0
    it, it_restart = 0, 0
    torch.cuda.synchronize()
    tstart = time.time()
    while True:
        for _ in range(check):
            tau, sig = eta / omega, eta * omega
            xn = x - tau * (KsT @ y - gt)
            w = y + sig * (Ks @ (2 * xn - x))
            y = w - sig * proj(w / sig)
            x = xn
            xs_ += x
            ys_ += y
            nav += 1
        it += check
        xa, ya = xs_ / nav, ys_ / nav
        kc, Kxc = kkt(x, y)
        ka, Kxa = kkt(xa, ya)
        for xx, KK in ((x, Kxc), (xa, Kxa)):
            lbc = cert(xx, KK)
            if lbc > best:
                best, best_x = lbc, xx.clone()
        cand = (xa, ya, ka) if ka < kc else (x, y, kc)
        torch.cuda.synchronize()
        el = time.time() - tstart
        trace.append((el, best, cand[2], it))
        restart = (cand[2] <= 0.2 * k_last or (cand[2] <= 0.8 * k_last and cand[2] > k_prev)
                   or (it - it_restart) >= 0.36 * max(it, 1))
        k_prev = cand[2]
        if restart:
            x, y = cand[0].clone(), cand[1].clone()
            dx, dy = float((x - xr).norm()), float((y - yr).norm())
            if dx > 1e-10 and dy > 1e-10:
                omega = float(np.exp(0.5 * np.log(dy / dx) + 0.5 * np.log(omega)))
            xr, yr, k_last, k_prev = x.clone(), y.clone(), cand[2], np.inf
            xs_.zero_(), ys_.zero_()
            nav, it_restart = 0, it
        if el > seconds + extra:
            break
    return trace, best_x.double().cpu().numpy() * d2, eta, it


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plates", default="syn0903,syn0899,syn0769,syn0736,syn0161,syn0295")
    ap.add_argument("--seconds", type=float, default=10.0)
    ap.add_argument("--extra", type=float, default=20.0)
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    D = PF + '/data/'
    d = np.load(D + 'nlo_val.npz', allow_pickle=True)
    names = [str(n) for n in d['name']]
    refs = {r["name"]: r for r in csv.DictReader(open(D + 'nlo_val_ref.csv'))}
    floor = {r["name"]: r for r in csv.DictReader(open(PF + '/../results/runs/pareto/fvs_floor_val.csv'))}
    model = nlo_train_fvs.load(D + 'nlo_fvs_ft30k.pt')
    mesh = R.canon()["mesh"]
    fvs = FVS(mesh)
    rows = []
    for nm in a.plates.split(","):
        i = names.index(nm)
        plate = dict(xs=d['x_mm'][i], ys=d['y_mm'][i], t=d['thickness_mm'][i], sigma_y=float(d['sigma_y_MPa'][i]))
        R.bounds(model, plate)                                           # warm-up
        th = time.time()
        out = R.bounds(model, plate)
        t_hyb = time.time() - th
        ts = time.time()
        S = setup(plate, out, mesh, fvs)
        t_setup = time.time() - ts
        trace, xb, eta, its = pdlp(S, a.seconds, extra=a.extra)
        # final certification in float64 by the frozen verifier
        Nv = (S["A"] @ xb).reshape(-1, 3, 3) / plate["sigma_y"]
        res = to_res_fvs(mesh, S["pl"], np.zeros((len(S["pl"]["nodes"]), 2)), Nv, plate["sigma_y"])
        lb_v, errs = certify_lb(res, plate["xs"], plate["ys"], plate["t"], plate["sigma_y"])
        lb_ref, lb_fl = float(refs[nm]["LB_ref_N"]), float(floor[nm]["LB_fvs_N"])
        at = lambda T: max(b for tt, b, _, _ in trace if tt <= T)
        dfc = lambda lbv: 100 * (1 - lbv / lb_ref)
        row = dict(name=nm, hybrid_lb=S["lb"], hybrid_s=t_hyb, setup_s=t_setup, recover_s=S["t_rec"], rec_err=S["rec_err"],
                   dual_res=S["dres"], ub=S["ub"], ub_dual=S["ub_dual"], lb_2s=at(2), lb_5s=at(5), lb_10s=at(a.seconds),
                   lb_end=trace[-1][1], lb_end_verified=lb_v, iters=its, ms_per_iter=1000 * trace[-1][0] / max(its, 1),
                   deficit_hybrid=dfc(S["lb"]), deficit_10s=dfc(at(a.seconds)), deficit_end=dfc(lb_v),
                   deficit_floor=dfc(lb_fl), certified="yes" if not errs else ";".join(errs))
        rows.append(row)
        print(f"{nm}: hybrid LB {S['lb']/1e3:.3f} kN (deficit {row['deficit_hybrid']:.2f}%) | PDLP LB 2s {row['lb_2s']/1e3:.3f} "
              f"5s {row['lb_5s']/1e3:.3f} 10s {row['lb_10s']/1e3:.3f} (deficit {row['deficit_10s']:.2f}%) end "
              f"{lb_v/1e3:.3f} verified ({row['deficit_end']:.2f}%, {its} it, {row['ms_per_iter']:.2f} ms/it) | floor "
              f"{row['deficit_floor']:.2f}% | checks: rec err {S['rec_err']:.1e}, dual res {S['dres']:.1e}, dual obj "
              f"{S['ub_dual']/1e3:.3f} vs UB {S['ub']/1e3:.3f} | setup {t_setup:.1f}s | {row['certified']}", flush=True)
        print("   trace (s, LB kN, KKT): " + "  ".join(f"{tt:.1f}:{b/1e3:.2f}/{k:.1e}" for tt, b, k, _ in trace[::max(1, len(trace) // 12)]), flush=True)
    if a.out and rows:
        with open(a.out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)


if __name__ == "__main__":
    main()
