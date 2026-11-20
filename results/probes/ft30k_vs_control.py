import os, sys, numpy as np
sys.path.insert(0, '/home/oldlinux/Research_Projects/Corrosion/PaperX/cnla')
import nlo_train_fvs as nf
from nlo_refine_eval import certify_lb, certify_ub
D = '/home/oldlinux/Research_Projects/Corrosion/PaperX/cnla/data/'
d = np.load(D + 'nlo_val.npz', allow_pickle=True); names = [str(n) for n in d['name']]
plates = [dict(xs=d["x_mm"][i], ys=d["y_mm"][i], t=d["thickness_mm"][i], sigma_y=float(d["sigma_y_MPa"][i])) for i in range(len(names))]
G = {}
for k in ('nlo_zp_ft30k', 'nlo_fvs_ft30k'):
    g = []
    for p, r in zip(plates, nf.predict(nf.load(D + k + '.pt'), plates)):
        ub, _ = certify_ub(r, p['xs'], p['ys'], p['t'], p['sigma_y']); lb, _ = certify_lb(r, p['xs'], p['ys'], p['t'], p['sigma_y'])
        g.append(100 * (ub - lb) / ub)
    G[k] = np.array(g)
a, b = G['nlo_zp_ft30k'], G['nlo_fvs_ft30k']; dd = b - a; o = np.argsort(-a)
print(f"mean {a.mean():.2f} -> {b.mean():.2f}; median {np.median(a):.2f} -> {np.median(b):.2f}; improved {np.sum(dd < 0)}/32; "
      f"hardest-5 mean diff {dd[o[:5]].mean():+.2f}, other 27 {dd[o[5:]].mean():+.2f}")
for i in o[:6]:
    print(f"  {names[i]}: {a[i]:.2f} -> {b[i]:.2f}")
