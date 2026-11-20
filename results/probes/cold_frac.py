import sys, numpy as np
sys.path.insert(0, '/home/oldlinux/Research_Projects/Corrosion/PaperX/cnla')
import nlo_train, nlo_refine as R
D = '/home/oldlinux/Research_Projects/Corrosion/PaperX/cnla/data/'
d = np.load(D + 'nlo_val.npz', allow_pickle=True); names = [str(n) for n in d['name']]
model = nlo_train.load(D + 'nlo_ckpt_100k.pt')
orig = R.refine_lb_fvs
def f(Nv0, th, tri, L, W, sy, **k):
    r = R.ratios_v(Nv0, th[tri], sy); rmax = r.max()
    out = orig(Nv0, th, tri, L, W, sy, **k)
    # affected set as in refine: recompute cheaply via the same function internals is complex; use ratio histogram of all vertices near hot
    hot = r.max(1) >= 0.5 * rmax
    print(f"  vertices with ratio >= 0.5/0.7/0.8/0.9 of max: {(r>=0.5*rmax).sum()}/{(r>=0.7*rmax).sum()}/{(r>=0.8*rmax).sum()}/{(r>=0.9*rmax).sum()} of {r.size}", flush=True)
    return out
R.refine_lb_fvs = f
for nm in sys.argv[1:]:
    i = names.index(nm); p = dict(xs=d['x_mm'][i], ys=d['y_mm'][i], t=d['thickness_mm'][i], sigma_y=float(d['sigma_y_MPa'][i]))
    print(nm); R.bounds(model, p)
