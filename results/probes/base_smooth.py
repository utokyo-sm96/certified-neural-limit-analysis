import sys, numpy as np, torch
sys.path.insert(0, '.')
import nlo_train as nt
from nlo_core import RefMesh, plate_on_mesh, to_res, yield_ratio, resultant
mesh = RefMesh(nt.NX, nt.NY)
d = np.load('data/la_validation_plates.npz', allow_pickle=True)
ref = [132.11, 157.62, 198.53, 180.42, 264.96, 295.30]
def smooth_x(tc, sig, power=1.0):
    if sig == 0: return tc
    k = torch.arange(-int(3*sig), int(3*sig)+1, dtype=tc.dtype); w = torch.exp(-0.5*(k/sig)**2); w /= w.sum()
    x = torch.nn.functional.pad(tc.permute(0, 2, 1).reshape(-1, 1, tc.shape[1]), (len(k)//2, len(k)//2), mode='replicate')
    return torch.nn.functional.conv1d(x, w[None, None]).reshape(tc.shape[0], tc.shape[2], tc.shape[1]).permute(0, 2, 1)
orig = nt.base_airy
for sig in (0, 5, 10, 20, 40, 80, 1e9):
    out = []
    for i in range(6):
        xs, ys, t = d['x_mm'][i], d['y_mm'][i], d['thickness_mm'][i]; sy = float(d['sigma_y_MPa'][i])
        pl = plate_on_mesh(mesh, xs, ys, t)
        th = torch.tensor(pl['thick'][None]); L = torch.tensor([pl['L']], dtype=torch.float64); W = torch.tensor([pl['W']], dtype=torch.float64)
        G = mesh.G
        tc = th[:, G:].reshape(1, nt.NX, nt.NY)
        if sig >= 1e8: tcs = torch.ones_like(tc)
        else: tcs = smooth_x(tc, sig)
        th2 = th.clone(); th2[:, G:] = tcs.reshape(1, -1)
        N = nt.stress(mesh, torch.zeros(1, nt.NX+2, nt.NY+2, dtype=torch.float64), L, W, th2)
        r = yield_ratio(N, th[:, mesh.tri].min(-1).values).max()
        lb = float(resultant(mesh, N, W)[0] / r) * sy
        out.append(lb/1e3/ref[i])
    print(f'sigma_x {sig:>6} cells: base LB / FE LB =', ' '.join(f'{o:.3f}' for o in out), f' mean {np.mean(out):.3f}')
