"""Plan C, C1: plate-independent global ADMM (two arms) + local exact repair, on validation plates.

Arm "zp":  global step in the ZP (C1-quadratic) Airy space with the existing plate-independent AiryFit banded solver.
Arm "fvs": global step in the full FVS (C1-cubic) space with ONE offline factorisation of K_cell^T K_cell (cell units).
Both: ADMM in derivative space with per-plate yield ellipsoids (c1_core), warm start from the hybrid (primal: its LB
field, dual: its UB velocity), snapshots repaired by the hybrid's local FVS windows; every bound float64-verified.

Usage: python c1_admm.py --arms zp,fvs --plates ... --out results/runs/plan-c/c1_admm.csv
"""
import argparse
import csv
import sys
import time

import numpy as np
import scipy.sparse.linalg as spla
import torch

sys.path.insert(0, '/home/oldlinux/Research_Projects/Corrosion/PaperX/loops/scratch')
PF = '/home/oldlinux/Research_Projects/Corrosion/PaperX/cnla'
sys.path.insert(0, PF)
import nlo_refine as R  # noqa: E402
import nlo_train_fvs  # noqa: E402
from c1_core import MV, Ellipsoids  # noqa: E402
from nlo_core import AiryFit, plate_on_mesh  # noqa: E402
from nlo_fvs import FVS, fvs_stress_matrix, to_res_fvs  # noqa: E402
from nlo_refine_eval import certify_lb  # noqa: E402

SNAPS = [0.5, 1.0, 2.0, 3.0, 5.0, 10.0]


def strains(nodes, tri, v):
    p = nodes[tri]
    area = 0.5 * ((p[:, 1, 0] - p[:, 0, 0]) * (p[:, 2, 1] - p[:, 0, 1]) - (p[:, 1, 1] - p[:, 0, 1]) * (p[:, 2, 0] - p[:, 0, 0]))
    bx = np.stack([p[:, 1, 1] - p[:, 2, 1], p[:, 2, 1] - p[:, 0, 1], p[:, 0, 1] - p[:, 1, 1]], 1) / (2 * area[:, None])
    cy = np.stack([p[:, 2, 0] - p[:, 1, 0], p[:, 0, 0] - p[:, 2, 0], p[:, 1, 0] - p[:, 0, 0]], 1) / (2 * area[:, None])
    vx, vy = v[tri, 0], v[tri, 1]
    return area, np.stack([(bx * vx).sum(1), (cy * vy).sum(1), (cy * vx).sum(1) + (bx * vy).sum(1)], 1)  # (exx, eyy, 2exy)


class ZPArm:
    """x = ZP coefficients c; D(x) = per-triangle (ww, uu, uw) in cell units; S = diag(1/hy^2, 1/hx^2, -1/(hx hy))."""
    name = "zp"

    def __init__(self, mesh):
        self.mesh, self.fit = mesh, AiryFit(mesh)

    def setup(self, pl, sy, lb_stress, area, e):
        NX, NY = self.mesh.NX, self.mesh.NY
        hx, hy = pl["L"] / NX, pl["W"] / NY
        self.S = np.array([1 / hy ** 2, 1 / hx ** 2, -1 / (hx * hy)])
        th = pl["thick"]
        self.ell = Ellipsoids(MV @ np.diag(self.S), sy * th[self.mesh.tri].min(1))
        self.h = np.zeros((NX, NY, 4, 3))
        self.h[NX - 1, :, 1, 0] = 1 / hy                                 # P = sum over right triangles of ww / hy
        self.h = self.h.reshape(-1, 3)
        x = self.solve(lb_stress.mean(1) / self.S)                       # LSQ fit of the hybrid's field (per-triangle mean)
        y = area[:, None] * (self.S * e)                                 # dual from the UB velocity
        return x, y

    def solve(self, target):
        t = torch.tensor(target.reshape(1, self.mesh.NX, self.mesh.NY, 4, 3))
        return self.fit.fit(t)

    def D(self, c):
        return self.mesh.airy_cell_derivs(c)[0].numpy().reshape(-1, 3)

    def update(self, d, u, rho):
        c = self.solve(d - u + self.h / rho)
        return c, self.D(c)

    def P(self, c, Dx):
        return float((self.h * Dx).sum())

    def stress_v(self, Dx):
        return np.repeat((Dx * self.S)[:, None, :], 3, 1)                # physical per-vertex (constant per triangle)


class FVSArm:
    """x = free FVS unknowns; D(x) = K_cell x per (triangle, vertex), rows (ww, uu, -uw); S = diag(1/hy^2, 1/hx^2, 1/(hx hy))."""
    name = "fvs"

    def __init__(self, mesh):
        self.mesh = mesh
        t0 = time.time()
        self.K = fvs_stress_matrix(mesh, FVS(mesh), mesh.NX, mesh.NY).tocsr()
        self.KT = self.K.T.tocsr()
        self.lu = spla.splu((self.KT @ self.K).tocsc(), permc_spec="COLAMD")
        self.offline_s = time.time() - t0

    def setup(self, pl, sy, lb_stress, area, e):
        NX, NY = self.mesh.NX, self.mesh.NY
        hx, hy = pl["L"] / NX, pl["W"] / NY
        self.S = np.array([1 / hy ** 2, 1 / hx ** 2, 1 / (hx * hy)])
        th = pl["thick"]
        self.ell = Ellipsoids(MV @ np.diag(self.S), sy * th[self.mesh.tri].reshape(-1))
        self.g = np.zeros(self.K.shape[1])
        self.g[-1] = 1 / hy                                              # P = c / hy
        x = self.lu.solve(self.KT @ (lb_stress.reshape(-1, 3) / self.S).reshape(-1))   # exact: the field is FVS
        y = np.repeat((area / 3)[:, None, None] * (self.S * e)[:, None, :], 3, 1).reshape(-1, 3)
        return x, y

    def D(self, x):
        return (self.K @ x).reshape(-1, 3)

    def update(self, d, u, rho):
        x = self.lu.solve(self.KT @ (d - u).reshape(-1) + self.g / rho)
        return x, self.D(x)

    def P(self, x, Dx):
        return float(self.g @ x)

    def stress_v(self, Dx):
        return (Dx * self.S).reshape(-1, 3, 3)


def admm(arm, x, y, seconds):
    Dx = arm.D(x)
    d = Dx.copy()
    rho = np.linalg.norm(y) / max(np.linalg.norm(d), 1e-30)
    u = y / rho
    snaps, marks = {}, list(SNAPS)
    it, t0 = 0, time.time()
    while marks:
        el = time.time() - t0
        while marks and el >= marks[0]:
            snaps[marks.pop(0)] = Dx.copy()
        if not marks:
            break
        x, Dx = arm.update(d, u, rho)
        d_old = d
        d = arm.ell.project(Dx + u)
        u = u + Dx - d
        it += 1
        if it % 5 == 0:                                                  # residual balancing
            pr = np.linalg.norm(Dx - d) / max(np.linalg.norm(d), 1e-30)
            dr = np.linalg.norm(d - d_old) / max(np.linalg.norm(u), 1e-30)
            if pr > 10 * dr:
                rho, u = rho * 2, u / 2
            elif dr > 10 * pr:
                rho, u = rho / 2, u * 2
    return snaps, it, time.time() - t0


def repair(mesh, pl, plate, Nv, secs=2.5):
    th = pl["thick"]
    sy = plate["sigma_y"]
    Nv = Nv / R.ratios_v(Nv, th[mesh.tri], sy).max()
    t0 = time.time()
    for _ in range(12):
        if time.time() - t0 > secs:
            break
        Nl = R.refine_lb_fvs(Nv, th, mesh.tri, pl["L"], pl["W"], sy, theta=0.9, dil=1, max_dof=6000)
        if Nl is None or R.lb_v(Nl, pl["W"]) <= R.lb_v(Nv, pl["W"]) * (1 + 1e-4):
            break
        Nv = Nl
    res = to_res_fvs(mesh, pl, np.zeros((len(pl["nodes"]), 2)), Nv / sy, sy)
    lb, errs = certify_lb(res, plate["xs"], plate["ys"], plate["t"], sy)
    return lb, time.time() - t0, errs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", default="zp,fvs")
    ap.add_argument("--plates", default="syn0903,syn0899,syn0769,syn0736,syn0161,syn0295")
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    D = PF + '/data/'
    d = np.load(D + 'nlo_val.npz', allow_pickle=True)
    names = [str(n) for n in d['name']]
    refs = {r["name"]: r for r in csv.DictReader(open(D + 'nlo_val_ref.csv'))}
    model = nlo_train_fvs.load(D + 'nlo_fvs_ft30k.pt')
    mesh = R.canon()["mesh"]
    t_off = time.time()
    arms = {"zp": ZPArm, "fvs": FVSArm}
    arms = [arms[k](mesh) for k in a.arms.split(",")]
    print(f"offline setup {time.time() - t_off:.1f}s (" + ", ".join(f"{ar.name}" for ar in arms) + ")", flush=True)
    rows = []
    for nm in a.plates.split(","):
        i = names.index(nm)
        plate = dict(xs=d['x_mm'][i], ys=d['y_mm'][i], t=d['thickness_mm'][i], sigma_y=float(d['sigma_y_MPa'][i]))
        R.bounds(model, plate)                                           # warm-up
        th0 = time.time()
        out = R.bounds(model, plate)
        t_hyb = time.time() - th0
        pl = plate_on_mesh(mesh, plate["xs"], plate["ys"], plate["t"])
        lb_ref = float(refs[nm]["LB_ref_N"])
        dfc = lambda lb: 100 * (1 - lb / lb_ref)
        area, e = strains(out["ub_res"]["nodes"], mesh.tri, out["ub_res"]["v"])
        lb_stress = np.asarray(out["lb_res"]["stress"], float)
        for arm in arms:
            ts = time.time()
            x, y = arm.setup(pl, plate["sigma_y"], lb_stress, area, e)
            t_set = time.time() - ts
            snaps, its, t_admm = admm(arm, x, y, max(SNAPS))
            reps = {k: repair(mesh, pl, plate, arm.stress_v(v)) for k, v in snaps.items()}
            row = dict(name=nm, arm=arm.name, hybrid_s=t_hyb, setup_s=t_set, iters=its, ms_per_it=1000 * t_admm / max(its, 1),
                       deficit_hybrid=dfc(out["lb_res"]["lb"]))
            for k, (lb, tr, errs) in reps.items():
                row[f"def_{k:g}s"] = dfc(max(lb, out["lb_res"]["lb"]))   # keep the better certified LB
                row[f"rep_{k:g}s_time"] = tr
                row[f"cert_{k:g}s"] = "yes" if not errs else ";".join(errs)
            rows.append(row)
            print(f"{nm} [{arm.name}] hybrid {row['deficit_hybrid']:.2f}% | " + "  ".join(
                f"{k:g}s: {row[f'def_{k:g}s']:.2f}%" for k in SNAPS) + f" | {its} it ({row['ms_per_it']:.1f} ms/it), setup "
                f"{t_set:.2f}s, repair ~{np.mean([v[1] for v in reps.values()]):.1f}s | certified "
                f"{all(not v[2] for v in reps.values())}", flush=True)
    if a.out and rows:
        with open(a.out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)


if __name__ == "__main__":
    main()
