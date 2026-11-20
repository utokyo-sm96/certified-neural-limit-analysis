import sys, time, numpy as np
sys.path.insert(0, '/home/oldlinux/Research_Projects/Corrosion/PaperX/cnla')
import nlo_train_fvs, nlo_refine as R
D = '/home/oldlinux/Research_Projects/Corrosion/PaperX/cnla/data/'
d = np.load(D + 'nlo_val.npz', allow_pickle=True); names = [str(n) for n in d['name']]
model = nlo_train_fvs.load(D + 'nlo_fvs_ft30k.pt')
orig_solve = R.solve_cone; orig = R.refine_lb_fvs; log = []
def s2(q, A, b, cones, max_iter=None):
    x, st = orig_solve(q, A, b, cones, max_iter=max_iter); log.append(('solve', A.shape, st, float(x[0]) if len(x) else None)); return x, st
def f(Nv0, th, tri, L, W, sy, **k):
    r0 = R.ratios_v(Nv0, th[tri], sy); t = time.time(); out = orig(Nv0, th, tri, L, W, sy, **k)
    log.append(('round', f"{time.time()-t:.2f}s", None if out is None else f"lb {R.lb_v(out, W)/1e3:.3f} (from {R.lb_v(Nv0, W)/1e3:.3f})", f"rmax {r0.max():.4f}"))
    return out
R.solve_cone = s2; R.refine_lb_fvs = f
for nm in sys.argv[1:]:
    i = names.index(nm); p = dict(xs=d['x_mm'][i], ys=d['y_mm'][i], t=d['thickness_mm'][i], sigma_y=float(d['sigma_y_MPa'][i]))
    log.clear(); R.bounds(model, p); print(nm); [print("  ", e) for e in log]
