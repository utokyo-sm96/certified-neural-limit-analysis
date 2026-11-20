import sys, time, numpy as np, torch
sys.path.insert(0, '/home/oldlinux/Research_Projects/Corrosion/PaperX/cnla')
import nlo_train, nlo_refine as R
from nlo_core import dissipation
D = '/home/oldlinux/Research_Projects/Corrosion/PaperX/cnla/data/'
d = np.load(D + 'nlo_val.npz', allow_pickle=True); names = [str(n) for n in d['name']]
model = nlo_train.load(D + 'nlo_ckpt_100k.pt')
cap = {}; orig = R.refine_ub_local
def grab(v0, nodes, tri, th, L, **k):
    cap['a'] = (v0.copy(), nodes, tri, th, L); return None
R.refine_ub_local = grab
R.refine_lb_fvs = lambda *a, **k: None
tot = {}
for nm in sys.argv[1:]:
    i = names.index(nm); p = dict(xs=d['x_mm'][i], ys=d['y_mm'][i], t=d['thickness_mm'][i], sigma_y=float(d['sigma_y_MPa'][i]))
    R.bounds(model, p); v0, nodes, tri, th, L = cap['a']
    Dd = lambda vv: float(dissipation(torch.tensor(vv[None]), torch.tensor(nodes[None]), torch.tensor(tri), torch.tensor(th[None]))[0])
    d0 = Dd(v0); line = []
    for frac in (0.8, 0.9, 0.95, 0.98):
        t = time.time(); v = orig(v0, nodes, tri, th, L, frac=frac); dt = time.time() - t
        g = 0 if v is None else 100 * (1 - Dd(v) / d0)
        tot.setdefault(frac, []).append((dt, g)); line.append(f"f{frac}: {dt:.2f}s -{g:.3f}%")
    print(nm, "  ".join(line), flush=True)
for f, v in tot.items():
    a = np.array(v); print(f"frac {f}: mean time {a[:,0].mean():.2f}s  mean UB gain {a[:,1].mean():.3f}%")
