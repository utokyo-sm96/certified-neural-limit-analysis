import sys, time, numpy as np, torch
sys.path.insert(0, '.')
from nlo_core import *
dev = 'cuda:0'
mesh = RefMesh(256, 80); fit = AiryFit(mesh)
d = np.load('data/la_validation_plates.npz', allow_pickle=True)
i = int(sys.argv[1]); ref = [132.11, 157.62, 198.53, 180.42, 264.96, 295.30][i]
xs, ys, t = d['x_mm'][i], d['y_mm'][i], d['thickness_mm'][i]; sy = float(d['sigma_y_MPa'][i])
pl = plate_on_mesh(mesh, xs, ys, t)
th = torch.tensor(pl['thick'][None], device=dev, dtype=torch.float32)
L = torch.tensor([pl['L']], device=dev); W = torch.tensor([pl['W']], device=dev)
tri = torch.tensor(mesh.tri, device=dev); tmin = th[:, tri].min(-1).values; tm = th[:, tri].mean(-1)
def lb(sig, tau=None):
    c = fit.fit(derivs_from_stress(tm[..., None] * sig, L, W, mesh))
    N = mesh.stress(c, L, W); r = yield_ratio(N, tmin)
    if tau is None: return float(resultant(mesh, N, W)[0] / r.max()) * sy / 1e3
    m = r.max(-1).values.detach(); return resultant(mesh, N, W) / (m * tau * torch.logsumexp(r / (m[:, None] * tau), -1))
s0 = torch.zeros(1, mesh.T, 3, device=dev); s0[..., 0] = 1
for tau, lr in [(0.05, 0.01), (0.02, 0.01), (0.05, 0.03), (0.01, 0.003)]:
    ds = torch.zeros_like(s0, requires_grad=True); opt = torch.optim.Adam([ds], lr=lr); log = []; t0 = time.time()
    for s in range(2001):
        loss = -lb(s0 + ds, tau).mean(); opt.zero_grad(); loss.backward(); opt.step()
        if s % 400 == 0:
            with torch.no_grad(): log.append(f'{s}:{lb(s0 + ds)/ref:.3f}')
    print(f'tau {tau} lr {lr}: LB/FE ' + '  '.join(log), f'({time.time()-t0:.0f}s, {1000*(time.time()-t0)/2001:.0f} ms/step)', flush=True)
