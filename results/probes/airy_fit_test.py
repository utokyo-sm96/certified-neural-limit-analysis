import sys, time, numpy as np, torch
sys.path.insert(0, '.')
from nlo_core import *
from eval_limit_analysis import verify
dev = 'cuda:0'
mesh = RefMesh(256, 80)
t0 = time.time(); fit = AiryFit(mesh); print(f'AiryFit setup {time.time()-t0:.1f}s', flush=True)
d = np.load('data/la_validation_plates.npz', allow_pickle=True)
refs = [132.11, 157.62, 198.53, 180.42, 264.96, 295.30]
for i in range(6):
    xs, ys, t = d['x_mm'][i], d['y_mm'][i], d['thickness_mm'][i]; sy = float(d['sigma_y_MPa'][i])
    pl = plate_on_mesh(mesh, xs, ys, t)
    th = torch.tensor(pl['thick'][None], device=dev, dtype=torch.float64)
    L = torch.tensor([pl['L']], device=dev, dtype=torch.float64); W = torch.tensor([pl['W']], device=dev, dtype=torch.float64)
    tri = torch.tensor(mesh.tri, device=dev)
    Ns = torch.zeros(1, mesh.T, 3, device=dev, dtype=torch.float64); Ns[..., 0] = th[:, tri].mean(-1)
    t1 = time.time(); c = fit.fit(derivs_from_stress(Ns, L, W, mesh)); dt = time.time() - t1
    N = mesh.stress(c, L, W)[0].cpu().numpy()
    v = np.zeros((len(pl['nodes']), 2)); v[:, 0] = pl['nodes'][:, 0] / pl['L']
    r = to_res(mesh, pl, v, N, sy); r['status_ub'] = r['status_lb'] = 'Solved'
    ub, lb, errs, info = verify(r, xs, ys, t, sy, 'rigid')
    print(f"{d['name'][i]} start N*=t(1,0,0): certified LB/FE = {lb/1e3/refs[i]:.3f}  {errs}  bnd {info['r_bnd']:.1e} cont {info['r_cont']:.1e}  fit {1000*dt:.0f} ms", flush=True)
