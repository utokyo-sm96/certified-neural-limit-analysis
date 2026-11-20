import sys, numpy as np, torch
sys.path.insert(0, '/home/oldlinux/Research_Projects/Corrosion/PaperX/cnla')
import nlo_train, nlo_refine as R
from nlo_core import dissipation
D = '/home/oldlinux/Research_Projects/Corrosion/PaperX/cnla/data/'
d = np.load(D + 'nlo_val.npz', allow_pickle=True); names = [str(n) for n in d['name']]
model = nlo_train.load(D + 'nlo_ckpt_100k.pt')
cap = {}
def glb(Nv0, th, tri, L, W, sy, **k):
    cap['lb'] = R.lb_v(Nv0, W) * 1.0; return None
def gub(v0, nodes, tri, th, L, **k):
    cap['ub'] = float(dissipation(torch.tensor(v0[None]), torch.tensor(nodes[None]), torch.tensor(tri), torch.tensor(th[None]))[0]); return None
R.refine_lb_fvs = glb; R.refine_ub_local = gub
g = []
for i, nm in enumerate(names):
    p = dict(xs=d['x_mm'][i], ys=d['y_mm'][i], t=d['thickness_mm'][i], sigma_y=float(d['sigma_y_MPa'][i]))
    R.bounds(model, p); ub = cap['ub'] * p['sigma_y']; g.append((ub - cap['lb']) / ub)
g = np.array(g)
print("gap after combination: mean %.3f median %.3f; quantiles 10/25/75/90: %s" % (g.mean(), np.median(g), np.round(np.quantile(g, [.1, .25, .75, .9]), 3)))
for gref in (0.05, 0.06, 0.07, 0.08):
    f = np.clip(g / gref, 0.5, 2.0); print(f"g_ref {gref}: mean factor {f.mean():.3f}  min {f.min():.2f} max {f.max():.2f}")
