import sys, time, numpy as np, torch
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
for tau, lr, mode in [(0.01, 0.01, "lse"), (0.05, 0.01, "lse"), (0.05, 0.003, "lse"), (0.05, 0.01, "anneal"), (0.2, 0.01, "anneal")]:
    dc = torch.zeros(1, nt.NX+2, nt.NY+2, device=dev, requires_grad=True)
    opt = torch.optim.Adam([dc], lr=lr)
    t0 = time.time(); log = []
    for s in range(3001):
        tt = tau * (0.02 ** (s / 3000)) if mode == 'anneal' else tau
        N = nt.stress(mesh, dc, L, W, th)
        r = yield_ratio(N, tmin)
        m = r.max(-1).values.detach()
        rs = m * tt * torch.logsumexp(r / (m[:, None] * tt), -1)
        lb = resultant(mesh, N, W) / rs
        loss = -lb.mean()
        opt.zero_grad(); loss.backward(); opt.step()
        if s % 750 == 0:
            with torch.no_grad():
                N = nt.stress(mesh, dc, L, W, th); hard = float(resultant(mesh, N, W)[0] / yield_ratio(N, tmin).max()) * sy / 1e3
            log.append(f'{s}:{hard/ref:.3f}')
    print(f'tau {tau} {mode} lr {lr}: certified LB / FE LB  ' + '  '.join(log), f'({time.time()-t0:.0f}s)', flush=True)
