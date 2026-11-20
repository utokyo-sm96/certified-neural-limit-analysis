"""Plan C, probe C0b: ADMM with an exact equilibrium (normal-equation) solve, so the Airy operator's conditioning
(kappa ~ 4e4 after Ruiz) no longer matters. Same LB problem and warm start as c0_pdlp.py.

    min -g.x  s.t.  K x = z,  z in B (product of balls)      (scaled variables, Ruiz + PC)
    x <- (K^T K)^-1 (K^T (z - u) + g / rho);  z <- Proj_B(K x + u);  u <- u + K x - z
Residual balancing for rho (the factorisation does not depend on rho). Every x gives a certified LB.
CPU, float64, one sparse LU solve per iteration (factorisation per plate is setup, timed separately).
"""
import argparse
import csv
import sys
import time

import numpy as np
import scipy.sparse.linalg as spla

sys.path.insert(0, '/home/oldlinux/Research_Projects/Corrosion/PaperX/loops/scratch')
import c0_pdlp as C  # noqa: E402
import nlo_refine as R  # noqa: E402
import nlo_train_fvs  # noqa: E402
from nlo_fvs import FVS, to_res_fvs  # noqa: E402
from nlo_refine_eval import certify_lb  # noqa: E402


def admm(S, seconds):
    Ks, d1, d2 = S["Ks"].tocsr(), S["d1"], S["d2"]
    KsT = Ks.T.tocsr()
    g, r = S["g"] * d2, S["r"] * d1
    m = len(r)
    tf = time.time()
    lu = spla.splu((KsT @ Ks).tocsc(), permc_spec="COLAMD")
    t_fact = time.time() - tf

    def proj(w):
        w3 = w.reshape(m, 3)
        nn = np.maximum(np.linalg.norm(w3, axis=1), 1e-300)
        return (w3 * np.minimum(1.0, r / nn)[:, None]).reshape(-1)

    def cert(Kx, x):
        return S["sy"] * S["t0"] * float(g @ x) / float((np.linalg.norm(Kx.reshape(m, 3), axis=1) / r).max())

    x = S["x0"] / d2
    z = Ks @ x
    y = S["y0"] / np.repeat(d1, 3)
    rho = np.linalg.norm(y) / max(np.linalg.norm(z), 1e-30)
    u = y / rho
    best, best_x = cert(z, x), x.copy()
    trace = [(0.0, best, 0.0, 0.0, rho)]
    raw = [(0.0, S["sy"] * S["t0"] * float(g @ x), 1.0)]
    t0 = time.time()
    it = 0
    snaps, marks = {}, [2.0, 5.0, 10.0, 20.0, 30.0]
    while time.time() - t0 < seconds:
        while marks and time.time() - t0 >= marks[0]:
            snaps[marks.pop(0)] = x * d2
        x = lu.solve(KsT @ (z - u) + g / rho)
        Kx = Ks @ x
        z_old = z
        z = proj(Kx + u)
        u = u + Kx - z
        it += 1
        lbc = cert(Kx, x)
        if lbc > best:
            best, best_x = lbc, x.copy()
        pr = np.linalg.norm(Kx - z) / max(np.linalg.norm(z), 1e-30)
        dr = rho * np.linalg.norm(KsT @ (z - z_old)) / max(np.linalg.norm(KsT @ (rho * u)), 1e-30)
        if it % 5 == 0:                                                  # residual balancing (u rescaled)
            if pr > 10 * dr:
                rho, u = rho * 2, u / 2
            elif dr > 10 * pr:
                rho, u = rho / 2, u * 2
        trace.append((time.time() - t0, best, pr, dr, rho))
        raw.append((time.time() - t0, S["sy"] * S["t0"] * float(g @ x), float((np.linalg.norm(Kx.reshape(m, 3), axis=1) / r).max())))
    for mk in marks:
        snaps[mk] = x * d2
    return trace, best_x * d2, it, t_fact, raw, x * d2, snaps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plates", default="syn0903,syn0899,syn0769,syn0736,syn0161,syn0295")
    ap.add_argument("--seconds", type=float, default=30.0)
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    D = C.PF + '/data/'
    d = np.load(D + 'nlo_val.npz', allow_pickle=True)
    names = [str(n) for n in d['name']]
    refs = {r["name"]: r for r in csv.DictReader(open(D + 'nlo_val_ref.csv'))}
    floor = {r["name"]: r for r in csv.DictReader(open(C.PF + '/../results/runs/pareto/fvs_floor_val.csv'))}
    model = nlo_train_fvs.load(D + 'nlo_fvs_ft30k.pt')
    mesh = R.canon()["mesh"]
    fvs = FVS(mesh)
    rows = []
    for nm in a.plates.split(","):
        i = names.index(nm)
        plate = dict(xs=d['x_mm'][i], ys=d['y_mm'][i], t=d['thickness_mm'][i], sigma_y=float(d['sigma_y_MPa'][i]))
        R.bounds(model, plate)                                           # warm-up (as in the evaluators)
        out = R.bounds(model, plate)
        S = C.setup(plate, out, mesh, fvs)
        trace, xb, its, t_fact, raw, x_last, snaps = admm(S, a.seconds)
        # local exact repair of ADMM snapshots (uncertified, slightly infeasible) with the hybrid's FVS windows
        th = S["pl"]["thick"]
        lb_ref = float(refs[nm]["LB_ref_N"])

        def repair(xx, secs=2.3):
            Nv = (S["A"] @ xx).reshape(-1, 3, 3)
            Nv = Nv / R.ratios_v(Nv, th[mesh.tri], plate["sigma_y"]).max()
            tr = time.time()
            for _ in range(12):
                if time.time() - tr > secs:
                    break
                Nl = R.refine_lb_fvs(Nv, th, mesh.tri, S["pl"]["L"], S["pl"]["W"], plate["sigma_y"], theta=0.9, dil=1, max_dof=6000)
                if Nl is None or R.lb_v(Nl, S["pl"]["W"]) <= R.lb_v(Nv, S["pl"]["W"]) * (1 + 1e-4):
                    break
                Nv = Nl
            res_ = to_res_fvs(mesh, S["pl"], np.zeros((len(S["pl"]["nodes"]), 2)), Nv / plate["sigma_y"], plate["sigma_y"])
            lbv, er = certify_lb(res_, plate["xs"], plate["ys"], plate["t"], plate["sigma_y"])
            return 100 * (1 - lbv / lb_ref), time.time() - tr, er
        snap_rep = {k: repair(v) for k, v in sorted(snaps.items())}
        print(f"   snapshots (ADMM s -> repaired deficit %, repair s): " + "  ".join(
            f"{k:.0f}s: {v[0]:.2f}% ({v[1]:.1f}s){'' if not v[2] else ' NOT CERTIFIED'}" for k, v in snap_rep.items()), flush=True)
        Nv = (S["A"] @ x_last).reshape(-1, 3, 3)
        Nv = Nv / R.ratios_v(Nv, th[mesh.tri], plate["sigma_y"]).max()
        lb_rep0 = R.lb_v(Nv, S["pl"]["W"])
        tr = time.time()
        for _ in range(12):
            if time.time() - tr > 2.3:
                break
            Nl = R.refine_lb_fvs(Nv, th, mesh.tri, S["pl"]["L"], S["pl"]["W"], plate["sigma_y"], theta=0.9, dil=1, max_dof=6000)
            if Nl is None or R.lb_v(Nl, S["pl"]["W"]) <= R.lb_v(Nv, S["pl"]["W"]) * (1 + 1e-4):
                break
            Nv = Nl
        res_r = to_res_fvs(mesh, S["pl"], np.zeros((len(S["pl"]["nodes"]), 2)), Nv / plate["sigma_y"], plate["sigma_y"])
        lb_rep, errs_r = certify_lb(res_r, plate["xs"], plate["ys"], plate["t"], plate["sigma_y"])
        Nv = (S["A"] @ xb).reshape(-1, 3, 3) / plate["sigma_y"]
        res = to_res_fvs(mesh, S["pl"], np.zeros((len(S["pl"]["nodes"]), 2)), Nv, plate["sigma_y"])
        lb_v, errs = certify_lb(res, plate["xs"], plate["ys"], plate["t"], plate["sigma_y"])
        lb_ref, lb_fl = float(refs[nm]["LB_ref_N"]), float(floor[nm]["LB_fvs_N"])
        dfc = lambda lbv: 100 * (1 - lbv / lb_ref)
        at = lambda T: max(b for tt, b, *_ in trace if tt <= T)
        row = dict(name=nm, deficit_hybrid=dfc(S["lb"]), deficit_5s=dfc(at(5)), deficit_10s=dfc(at(10)),
                   deficit_30s=dfc(at(a.seconds)), deficit_end_verified=dfc(lb_v), deficit_floor=dfc(lb_fl), iters=its,
                   s_per_iter=trace[-1][0] / max(its, 1), factor_s=t_fact, certified="yes" if not errs else ";".join(errs))
        row.update(**{f"rep_{int(k)}s": v[0] for k, v in snap_rep.items()}, **{f"rep_{int(k)}s_time": v[1] for k, v in snap_rep.items()})
        row.update(raw_obj_deficit_end=dfc(raw[-1][1]), raw_max_ratio_end=raw[-1][2], deficit_scaled_last=dfc(lb_rep0),
                   deficit_repaired=dfc(lb_rep), repair_certified="yes" if not errs_r else ";".join(errs_r))
        rows.append(row)
        print(f"   ADMM last iterate: objective deficit {row['raw_obj_deficit_end']:.2f}% at max yield ratio {row['raw_max_ratio_end']:.3f} "
              f"-> scaled {row['deficit_scaled_last']:.2f}% -> after 2.3 s local repair {row['deficit_repaired']:.2f}% ({row['repair_certified']})", flush=True)
        print(f"{nm}: LB deficit vs FE  hybrid {row['deficit_hybrid']:.2f}% -> ADMM 5s {row['deficit_5s']:.2f}  10s "
              f"{row['deficit_10s']:.2f}  {a.seconds:.0f}s {row['deficit_30s']:.2f} (verified {row['deficit_end_verified']:.2f}%) | "
              f"floor {row['deficit_floor']:.2f}% | {its} it, {row['s_per_iter']:.3f} s/it, factorisation {t_fact:.1f}s | "
              f"{row['certified']}", flush=True)
        print("   trace (s, deficit %, primal res, dual res, rho): " + "  ".join(
            f"{tt:.0f}:{dfc(b):.2f}/{p:.0e}/{q:.0e}/{rh:.0e}" for tt, b, p, q, rh in trace[::max(1, len(trace) // 8)]), flush=True)
    if a.out and rows:
        with open(a.out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)


if __name__ == "__main__":
    main()
