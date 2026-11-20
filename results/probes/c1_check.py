import sys, time, numpy as np, torch
sys.path.insert(0, '/home/oldlinux/Research_Projects/Corrosion/PaperX/loops/scratch')
import c1_admm as C
D = C.PF + '/data/'
d = np.load(D + 'nlo_val.npz', allow_pickle=True); names = [str(n) for n in d['name']]
model = C.nlo_train_fvs.load(D + 'nlo_fvs_ft30k.pt'); mesh = C.R.canon()["mesh"]
i = names.index('syn0903'); plate = dict(xs=d['x_mm'][i], ys=d['y_mm'][i], t=d['thickness_mm'][i], sigma_y=float(d['sigma_y_MPa'][i]))
out = C.R.bounds(model, plate); pl = C.plate_on_mesh(mesh, plate["xs"], plate["ys"], plate["t"])
area, e = C.strains(out["ub_res"]["nodes"], mesh.tri, out["ub_res"]["v"]); lbs = np.asarray(out["lb_res"]["stress"], float)
rng = np.random.default_rng(0)
for Arm in (C.ZPArm, C.FVSArm):
    t = time.time(); arm = Arm(mesh); toff = time.time() - t
    x, y = arm.setup(pl, plate["sigma_y"], lbs, area, e)
    if arm.name == "zp":
        xr = arm.solve(rng.standard_normal((mesh.T, 3)))
    else:
        xr = rng.standard_normal(arm.K.shape[1])
    Dr = arm.D(xr)
    print(f"[{arm.name}] offline {toff:.1f}s | virtual work: P(x) {arm.P(xr, Dr):.6e} vs y.D(x) {float((y * Dr).sum()):.6e} "
          f"(rel {abs(arm.P(xr, Dr) - (y * Dr).sum()) / abs(arm.P(xr, Dr)):.1e})", flush=True)
    Dx = arm.D(x); sy = plate["sigma_y"]
    lb0 = arm.P(x, Dx) / arm.ell.ratio(Dx).max()
    dual = float((arm.ell.rad * np.linalg.norm(y @ np.linalg.inv(arm.ell.G), axis=1)).sum())
    print(f"   warm start: certified LB {lb0/1e3:.3f} kN (hybrid {out['lb_res']['lb']/1e3:.3f}); dual objective {dual/1e3:.3f} kN (hybrid UB {out['ub_res']['ub']/1e3:.3f})", flush=True)
    t = time.time(); n = 20; dd = Dx.copy(); u = np.zeros_like(dd)
    for _ in range(n):
        x2, D2 = arm.update(dd, u, 1.0); dd = arm.ell.project(D2 + u)
    print(f"   {1000 * (time.time() - t) / n:.1f} ms per ADMM iteration", flush=True)
