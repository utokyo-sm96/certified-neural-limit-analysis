import sys, time, numpy as np
sys.path.insert(0, '/home/oldlinux/Research_Projects/Corrosion/PaperX/cnla')
import nlo_train_fvs, nlo_refine as R
from nlo_refine_eval import certify_lb, certify_ub
D = '/home/oldlinux/Research_Projects/Corrosion/PaperX/cnla/data/'
d = np.load(D + 'nlo_val.npz', allow_pickle=True); names = [str(n) for n in d['name']]
model = nlo_train_fvs.load(D + 'nlo_fvs_ft30k.pt')
for nm in sys.argv[1:]:
    i = names.index(nm); p = dict(xs=d['x_mm'][i], ys=d['y_mm'][i], t=d['thickness_mm'][i], sigma_y=float(d['sigma_y_MPa'][i]))
    for gs in (0.0, 2.0):
        R.GLOBAL_SECONDS = gs; R.bounds(model, p)
        t = time.time(); out = R.bounds(model, p); dt = time.time() - t
        ub, e1 = certify_ub(out['ub_res'], p['xs'], p['ys'], p['t'], p['sigma_y']); lb, e2 = certify_lb(out['lb_res'], p['xs'], p['ys'], p['t'], p['sigma_y'])
        print(f"{nm} global {gs:.0f}s: gap {100*(ub-lb)/ub:.2f}%  {dt:.1f}s  {e1+e2 or 'certified'}", flush=True)
