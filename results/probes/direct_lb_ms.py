import sys, time, numpy as np, torch, torch.nn.functional as F
sys.path.insert(0, '.')
import nlo_train as nt
from nlo_core import RefMesh, plate_on_mesh, yield_ratio, resultant
dev = 'cuda:0'
mesh = RefMesh(nt.NX, nt.NY)
d = np.load('data/la_validation_plates.npz', allow_pickle=True)
i = int(sys.argv[1]); ref = [132.11, 157.62, 198.53, 180.42, 264.96, 295.30][i]
xs, ys, t = d['x_mm'][i], d['y_mm'][i], d['thickness_mm'][i]; sy = float(d['sigma_y_MPa'][i])
pl = plate_on_mesh(mesh, xs, ys, t)
th = torch.tensor(pl['thick'][None], device=dev, dtype=torch.float32)
L = torch.tensor([pl['L']], device=dev); W = torch.tensor([pl['W']], device=dev)
tmin = th[:, torch.tensor(mesh.tri, device=dev)].min(-1).values
size = (nt.NX + 2, nt.NY + 2)
for levels, scale_pow, tau, lr in [(6, 2.0, 0.05, 0.01), (6, 2.0, 0.2, 0.01), (6, 1.0, 0.05, 0.01), (6, 2.0, 0.05, 0.03)]:
    zs = [torch.zeros(1, 1, max(2, size[0] // 2**l), max(2, size[1] // 2**l), device=dev, requires_grad=True) for l in range(levels)]
    opt = torch.optim.Adam(zs, lr=lr)
    t0 = time.time(); log = []
    for s in range(3001):
        dc = sum((2.0 ** (scale_pow * l)) * F.interpolate(z, size=size, mode='bicubic', align_corners=True) for l, z in enumerate(zs))[:, 0]
        N = nt.stress(mesh, dc, L, W, th)
        r = yield_ratio(N, tmin); m = r.max(-1).values.detach()
        rs = m * tau * torch.logsumexp(r / (m[:, None] * tau), -1)
        lb = resultant(mesh, N, W) / rs
        opt.zero_grad(); (-lb.mean()).backward(); opt.step()
        if s % 750 == 0:
            with torch.no_grad():
                N = nt.stress(mesh, dc, L, W, th); hard = float(resultant(mesh, N, W)[0] / yield_ratio(N, tmin).max()) * sy / 1e3
            log.append(f'{s}:{hard/ref:.3f}')
    print(f'levels {levels} pow {scale_pow} tau {tau} lr {lr}: LB/FE ' + '  '.join(log), f'({time.time()-t0:.0f}s)', flush=True)
