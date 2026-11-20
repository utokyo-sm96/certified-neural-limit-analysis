"""Probe: floor of a linear-stress (C1-cubic Airy) LB and the UB on the 256x80 reference mesh, vs ZP constant stress."""
import os, sys, time
os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np
sys.path.insert(0, '/home/oldlinux/Research_Projects/Corrosion/PaperX/cnla')
from multiprocessing import Pool

def job(i):
    from nlo_core import RefMesh, plate_on_mesh, optimal_lb_zp
    from limit_analysis import upper_bound, lower_bound
    from eval_limit_analysis import verify
    d = np.load('/home/oldlinux/Research_Projects/Corrosion/PaperX/cnla/data/la_validation_plates.npz', allow_pickle=True)
    xs, ys, t = d['x_mm'][i], d['y_mm'][i], d['thickness_mm'][i]; sy = float(d['sigma_y_MPa'][i])
    mesh = RefMesh(256, 80); pl = plate_on_mesh(mesh, xs, ys, t)
    L, W, th = pl['L'], pl['W'], pl['thick']; t0 = th.max(); tol = 1e-6 * max(L, W)
    t1 = time.time(); ub = upper_bound(pl['nodes'], mesh.tri, th, t0, L, 'rigid', tol); tu = time.time() - t1
    t1 = time.time(); lb = lower_bound(pl['nodes'], mesh.tri, th, t0, L, W, 'rigid', tol); tl = time.time() - t1
    res = dict(nodes=pl['nodes'], tri=mesh.tri, thick=th, ub=ub['ub_hat'] * sy * t0, v=ub['v'], lb=lb['lb_hat'] * sy * t0,
               stress=lb['stress_hat'] * sy * t0, status_ub=ub['status'], status_lb=lb['status'])
    UB, LB, errs, _ = verify(res, xs, ys, t, sy, 'rigid')
    LBzp, _, s2, tz = optimal_lb_zp(mesh, pl, sy)
    return str(d['name'][i]), UB, LB, LBzp, errs, tu, tl, tz

fe = {'AN-syn0': (132.11, 132.91), 'AN-syn1': (157.62, 158.36), 'AN-syn2': (198.53, 199.40), 'AM-syn3': (180.42, 181.13), 'AM-syn4': (264.96, 266.10), 'AM-syn5': (295.30, 296.19)}
with Pool(3) as p:
    out = p.map(job, range(6))
g_lin, g_zp = [], []
for name, UB, LB, LBzp, errs, tu, tl, tz in out:
    lbf, ubf = fe[name]
    g_lin.append((UB - LB) / UB); g_zp.append((UB - LBzp) / UB)
    print(f"{name}: UB {UB/1e3:.2f} (+{100*(UB/1e3/ubf-1):.2f}%)  LB linear {LB/1e3:.2f} (-{100*(1-LB/1e3/lbf):.2f}%)  LB ZP-const {LBzp/1e3:.2f} (-{100*(1-LBzp/1e3/lbf):.2f}%)  "
          f"gap linear {100*g_lin[-1]:.2f}% vs ZP {100*g_zp[-1]:.2f}%  {errs or 'certified'}  ({tu:.0f}s UB, {tl:.0f}s LB-lin, {tz:.0f}s LB-ZP)", flush=True)
print(f"mean floor: linear-stress {100*np.mean(g_lin):.2f}%  vs ZP constant-stress {100*np.mean(g_zp):.2f}%  (step-1 adaptive FE 0.44%)")
