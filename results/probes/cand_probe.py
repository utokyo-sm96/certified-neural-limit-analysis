"""A8 probe: extra candidates from perturbed thickness inputs (fields stay admissible; bounds use the true thickness)."""
import sys, time, numpy as np, torch
from scipy.ndimage import gaussian_filter
from scipy.interpolate import RegularGridInterpolator
sys.path.insert(0, '/home/oldlinux/Research_Projects/Corrosion/PaperX/cnla')
import nlo_train_fvs, nlo_refine as R
from nlo_core import dissipation
D = '/home/oldlinux/Research_Projects/Corrosion/PaperX/cnla/data/'
d = np.load(D + 'nlo_val.npz', allow_pickle=True); names = [str(n) for n in d['name']]
model = nlo_train_fvs.load(D + 'nlo_fvs_ft30k.pt')
rng = np.random.default_rng(0)
def cands(xs, ys, t, sy, tin):
    L, W = xs[-1], ys[-1]
    pl = [dict(xs=L - xs[::-1] if fx else xs, ys=W - ys[::-1] if fy else ys, t=tin[::-1 if fx else 1, ::-1 if fy else 1], sigma_y=sy) for fx, fy in R.MIRRORS]
    res = [R.mirror_back(r, fx, fy, L, W) for r, (fx, fy) in zip(nlo_train_fvs.predict(model, pl), R.MIRRORS)]
    return [R.to_canonical(r, L, W) for r in res]
summ = {}
for nm in sys.argv[1:]:
    i = names.index(nm); xs, ys, t, sy = d['x_mm'][i], d['y_mm'][i], d['thickness_mm'][i], float(d['sigma_y_MPa'][i])
    L, W = xs[-1], ys[-1]; mesh = R.canon()['mesh']; nodes = mesh.nodes(L, W); tri = mesh.tri
    th = nlo_train_fvs.plate_on_mesh(mesh, xs, ys, t)['thick']
    Dd = lambda v: float(dissipation(torch.tensor(v[None]), torch.tensor(nodes[None]), torch.tensor(tri), torch.tensor(th[None]))[0]) * sy
    base = cands(xs, ys, t, sy, t)
    sets = {'K4': base,
            '+smooth1': base + cands(xs, ys, t, sy, gaussian_filter(t, 1.0, mode='nearest')),
            '+smooth2': base + cands(xs, ys, t, sy, gaussian_filter(t, 2.0, mode='nearest')),
            '+noise3%': base + cands(xs, ys, t, sy, t * (1 + 0.03 * rng.standard_normal(t.shape))),
            '+sm1+sm2': base + cands(xs, ys, t, sy, gaussian_filter(t, 1.0, mode='nearest')) + cands(xs, ys, t, sy, gaussian_filter(t, 2.0, mode='nearest'))}
    line = []
    for k, c in sets.items():
        t0 = time.time(); N = R.combine_lb([n for _, n in c], nodes, tri, th, W, sy); v = R.combine_ub([vv for vv, _ in c], nodes, tri, th, sy); dt = time.time() - t0
        lb, ub = R.lb_v(N, W), Dd(v); g = 100 * (ub - lb) / ub; summ.setdefault(k, []).append(g); line.append(f"{k} {g:.2f}% ({dt:.1f}s)")
    print(nm, "  ".join(line), flush=True)
for k, v in summ.items():
    print(f"{k}: mean combined gap {np.mean(v):.3f}%")
