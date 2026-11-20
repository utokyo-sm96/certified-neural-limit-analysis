import os, sys, time
import numpy as np
sys.path.insert(0, '/home/oldlinux/Research_Projects/Corrosion/PaperX/loops/scratch')
sys.path.insert(0, '/home/oldlinux/Research_Projects/Corrosion/PaperX/cnla')
import nlo_train
import nlo_refine_f1 as R
D = '/home/oldlinux/Research_Projects/Corrosion/PaperX/cnla/data/'
d = np.load(D + 'nlo_val.npz', allow_pickle=True); names = [str(n) for n in d['name']]
model = nlo_train.load(D + 'nlo_ckpt_100k.pt')
orig = R.refine_lb_fvs
for cfg in [dict(max_dof=6000, dil=2), dict(max_dof=3000, dil=2), dict(max_dof=6000, dil=1)]:
    log = []
    def f(*a, **k):
        k.update(cfg); t = time.time(); r = orig(*a, **k)
        log.append((time.time() - t, None if r is None else R.lb_v(r, a[4])))
        return r
    R.refine_lb_fvs = f
    for nm in sys.argv[1:]:
        i = names.index(nm); p = dict(xs=d['x_mm'][i], ys=d['y_mm'][i], t=d['thickness_mm'][i], sigma_y=float(d['sigma_y_MPa'][i]))
        log.clear(); R.bounds(model, p)
        print(cfg, nm, "  ".join(f"r{j}: {t:.2f}s LB {lb/1e3:.2f}" if lb else f"r{j}: {t:.2f}s none" for j, (t, lb) in enumerate(log)), flush=True)
