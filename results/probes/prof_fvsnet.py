import sys, time, numpy as np
sys.path.insert(0, '/home/oldlinux/Research_Projects/Corrosion/PaperX/cnla')
import nlo_train_fvs, nlo_refine as R
D = '/home/oldlinux/Research_Projects/Corrosion/PaperX/cnla/data/'
d = np.load(D + 'nlo_val.npz', allow_pickle=True); names = [str(n) for n in d['name']]
model = nlo_train_fvs.load(D + 'nlo_fvs_ft30k.pt')
T = {}
def wrap(mod, name):
    f = getattr(mod, name)
    def g(*a, **k):
        t = time.time(); r = f(*a, **k); T[name] = T.get(name, 0) + time.time() - t; return r
    setattr(mod, name, g)
for n in ('to_canonical', 'combine_lb', 'combine_ub', 'refine_lb_fvs', 'refine_ub_local', 'mirror_back'):
    wrap(R, n)
wrap(nlo_train_fvs, 'predict')
import nlo_fvs; wrap(nlo_fvs, 'to_res_fvs'); nlo_train_fvs.to_res_fvs = nlo_fvs.to_res_fvs
for nm in sys.argv[1:]:
    i = names.index(nm); p = dict(xs=d['x_mm'][i], ys=d['y_mm'][i], t=d['thickness_mm'][i], sigma_y=float(d['sigma_y_MPa'][i]))
    R.bounds(model, p); T.clear()
    t = time.time(); R.bounds(model, p); dt = time.time() - t
    print(nm, f"total {dt:.2f}s  " + "  ".join(f"{k} {v:.2f}" for k, v in T.items()), flush=True)
