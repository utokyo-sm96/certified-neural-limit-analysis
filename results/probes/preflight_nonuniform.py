"""Pre-flight (no held-out data): the full method on validation plates resampled to non-uniform grids like real scans
(shorter last grid step in x and y). Must certify and give a gap similar to the uniform version."""
import sys, time, numpy as np
from scipy.interpolate import RegularGridInterpolator
sys.path.insert(0, '/home/oldlinux/Research_Projects/Corrosion/PaperX/cnla')
import nlo_train_fvs, nlo_refine as R
from nlo_refine_eval import certify_lb, certify_ub
D = '/home/oldlinux/Research_Projects/Corrosion/PaperX/cnla/data/'
d = np.load(D + 'nlo_val.npz', allow_pickle=True); names = [str(n) for n in d['name']]
model = nlo_train_fvs.load(D + 'nlo_fvs_ft30k.pt')
print(f"defaults: GLOBAL_ARM {R.GLOBAL_ARM} ITERS {R.GLOBAL_ITERS} UB_ITERS {R.UB_GLOBAL_ITERS} LB_SECONDS {R.LB_SECONDS}", flush=True)
def run(p):
    t = time.time(); out = R.bounds(model, p); dt = time.time() - t
    ub, e1 = certify_ub(out['ub_res'], p['xs'], p['ys'], p['t'], p['sigma_y']); lb, e2 = certify_lb(out['lb_res'], p['xs'], p['ys'], p['t'], p['sigma_y'])
    return 100 * (ub - lb) / ub, dt, e1 + e2
first = True
for nm in sys.argv[1:]:
    i = names.index(nm); xs, ys, t = d['x_mm'][i], d['y_mm'][i], d['thickness_mm'][i]; sy = float(d['sigma_y_MPa'][i])
    f = RegularGridInterpolator((xs, ys), t)
    xs2 = np.append(xs[:-1], xs[-2] + 0.8); ys2 = np.append(ys[:-1], ys[-2] + 1.2)       # shorter last steps
    X, Y = np.meshgrid(xs2, ys2, indexing='ij'); t2 = f(np.stack([X.ravel(), Y.ravel()], 1)).reshape(X.shape)
    p1 = dict(xs=xs, ys=ys, t=t, sigma_y=sy); p2 = dict(xs=xs2, ys=ys2, t=t2, sigma_y=sy)
    if first:
        R.bounds(model, p1); first = False                               # warm-up (offline factorisations)
    g1, t1, e1 = run(p1); g2, t2_, e2 = run(p2)
    print(f"{nm}: uniform grid gap {g1:.2f}% ({t1:.1f}s, {e1 or 'certified'}) | non-uniform grid (L {xs[-1]:.0f}->{xs2[-1]:.1f}, "
          f"W {ys[-1]:.0f}->{ys2[-1]:.1f}) gap {g2:.2f}% ({t2_:.1f}s, {e2 or 'certified'})", flush=True)
