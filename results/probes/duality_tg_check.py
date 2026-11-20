"""Numerical check of the extended duality identity (Appendix A, Proposition 5 for a quadrature over piece vertices):
for the lower-bound problem certified for t_g on the reference mesh (yield at every piece vertex p, capacity t_g(p)),
weights w_p = fan quadrature of the pieces (each fan triangle gives a third of its area to each of its vertices),
  (a) virtual work: sum_p w_p N(p) : eps_T(p) = P(N) for any equilibrated, traction-free cubic Airy field N and any
      kinematically admissible v (here: the network's fields and random Airy coefficient vectors);
  (b) the dual objective sum_p w_p sigma_y t_g(p) D(eps_T(p)) equals the exact t_g dissipation (the reported UB).
Usage: python results/probes/duality_tg_check.py [n_plates]"""
import os
import sys

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "cnla"))
sys.path.insert(0, os.path.join(ROOT, "paper", "figures"))
SQ3 = np.sqrt(3.0)


def fan_quadrature(nodes, tri, xs, ys):
    """Piece vertices (owner triangle, barycentric coordinates, point) with fan-quadrature weights."""
    from thickness_tg import tg_pieces
    P, cnt, own = tg_pieces(nodes, tri, xs, ys)
    pts, ws, ow = [], [], []
    for k in range(1, P.shape[1] - 1):
        use = k + 1 < cnt
        a, b, c = P[use, 0], P[use, k], P[use, k + 1]
        area = 0.5 * np.abs((b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1]) - (b[:, 1] - a[:, 1]) * (c[:, 0] - a[:, 0]))
        for q in (a, b, c):
            pts.append(q)
            ws.append(area / 3)
            ow.append(own[use])
    pts, ws, ow = np.concatenate(pts), np.concatenate(ws), np.concatenate(ow)
    pt = nodes[tri[ow]]
    M = np.stack([pt[:, 1] - pt[:, 0], pt[:, 2] - pt[:, 0]], 2)
    l12 = np.linalg.solve(M, (pts - pt[:, 0])[..., None])[..., 0]
    lam = np.concatenate([1 - l12.sum(1, keepdims=True), l12], 1)
    return ow, lam, ws, pts


def strain_rates(nodes, tri, v):
    p = nodes[tri]
    area = 0.5 * ((p[:, 1, 0] - p[:, 0, 0]) * (p[:, 2, 1] - p[:, 0, 1]) - (p[:, 1, 1] - p[:, 0, 1]) * (p[:, 2, 0] - p[:, 0, 0]))
    bx = np.stack([p[:, 1, 1] - p[:, 2, 1], p[:, 2, 1] - p[:, 0, 1], p[:, 0, 1] - p[:, 1, 1]], 1) / (2 * area[:, None])
    cy = np.stack([p[:, 2, 0] - p[:, 1, 0], p[:, 0, 0] - p[:, 2, 0], p[:, 1, 0] - p[:, 0, 0]], 1) / (2 * area[:, None])
    vx, vy = v[tri, 0], v[tri, 1]
    exx, eyy = (bx * vx).sum(1), (cy * vy).sum(1)
    exy = 0.5 * ((cy * vx).sum(1) + (bx * vy).sum(1))                     # tensor component
    return exx, eyy, exy


def main():
    import nlo_global as G
    import nlo_refine as R
    import nlo_train_fvs
    from thickness_tg import tg_eval, tg_quantities
    from val_histories import load_set
    from verify_v2 import verify_v2
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    model = nlo_train_fvs.load(os.path.join(ROOT, "cnla", "data", "nlo_fvs_ft30k.pt"))
    mesh = R.canon()["mesh"]
    arm = G.FVSArm(mesh)
    rng = np.random.default_rng(0)
    for p in load_set("val")[:n]:
        xs, ys, t, sy = p["xs"], p["ys"], np.asarray(p["t"], float), p["sigma_y"]
        L, W = xs[-1], ys[-1]
        nodes, tri = mesh.nodes(L, W), mesh.tri
        r0 = R.mirror_back(nlo_train_fvs.predict(model, [dict(xs=xs, ys=ys, t=t, sigma_y=sy)])[0], False, False, L, W)
        v, Nnet = R.to_canonical(r0, L, W)
        ow, lam, ws, pts = fan_quadrature(nodes, tri, xs, ys)
        exx, eyy, exy = strain_rates(nodes, tri, v)
        hx, hy = L / mesh.NX, W / mesh.NY
        S = np.array([1 / hy ** 2, 1 / hx ** 2, 1 / (hx * hy)])

        def vw(Nv):
            Np = (lam[:, :, None] * Nv[ow]).sum(1)
            return float(np.sum(ws * (Np[:, 0] * exx[ow] + Np[:, 1] * eyy[ow] + 2 * Np[:, 2] * exy[ow])))
        errs = [abs(vw(Nnet) / R.lb_v(Nnet, W) - 1)]
        for _ in range(3):                                               # random cubic Airy fields
            x = rng.standard_normal(arm.K.shape[1])
            Nv = ((arm.K @ x).reshape(-1, 3) * S).reshape(-1, 3, 3)
            errs.append(abs(vw(Nv) - x[-1] / hy) / max(abs(x[-1] / hy), np.abs(Nv).max() * W))
        # (b) dual objective vs exact t_g dissipation (verifier v2)
        D = (2 / SQ3) * np.sqrt(np.maximum(exx ** 2 + eyy ** 2 + exx * eyy + exy ** 2, 0))
        dual = float(np.sum(ws * sy * tg_eval(pts, xs, ys, t) * D[ow]))
        Sd = np.zeros((len(tri), 3, 3))
        Sd[:, :, 0] = 1.0
        ub2, _, _, _ = verify_v2(dict(nodes=nodes, tri=tri, thick=np.ones(len(nodes)), v=v, ub=0.0, stress=Sd, lb=0.0,
                                      status_ub="Solved", status_lb="Solved"), xs, ys, t, sy)
        integ = tg_quantities(nodes, tri, xs, ys, t)[0]
        qerr = np.abs(np.bincount(ow, ws * tg_eval(pts, xs, ys, t), minlength=len(tri)) - integ).max() / integ.max()
        print(f"{p['name']}: virtual work rel. error: network field {errs[0]:.1e}, random Airy fields max {max(errs[1:]):.1e}; "
              f"dual objective / exact t_g dissipation - 1 = {dual / ub2 - 1:.1e}; quadrature vs exact int t_g {qerr:.1e}; "
              f"{len(ws)} weighted piece vertices", flush=True)


if __name__ == "__main__":
    main()
