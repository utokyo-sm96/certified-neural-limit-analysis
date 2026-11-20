import sys, time, numpy as np, torch
sys.path.insert(0, '.')
import nlo_train as nt
from nlo_core import *
dev = 'cuda:0'
mesh = RefMesh(nt.NX, nt.NY)
d = np.load('data/la_validation_plates.npz', allow_pickle=True)
i = int(sys.argv[1]); fe = [132.91, 158.36, 199.40, 181.13, 266.10, 296.19][i]; mesh_opt = [133.50, 159.09, 199.64, 181.80, 266.62, 296.05][i]
xs, ys, t = d['x_mm'][i], d['y_mm'][i], d['thickness_mm'][i]; sy = float(d['sigma_y_MPa'][i])
pl = plate_on_mesh(mesh, xs, ys, t)
th = torch.tensor(pl['thick'][None], device=dev, dtype=torch.float32)
L = torch.tensor([pl['L']], device=dev); W = torch.tensor([pl['W']], device=dev)
tri = torch.tensor(mesh.tri, device=dev)
nodes = torch.tensor(mesh.unit, device=dev, dtype=torch.float32)[None] * torch.stack([L, W], -1)[:, None, :]
for vys, lr in [(0.1, 0.05), (1.0, 0.05), (1.0, 0.2)]:
    nt.VY_SCALE = vys
    out = torch.zeros(1, 4, nt.NX, nt.NY, device=dev, requires_grad=True); opt = torch.optim.Adam([out], lr=lr); log = []; t0 = time.time()
    for s in range(3001):
        ub = dissipation(nt.velocity(mesh, out), nodes, tri, th) * sy / 1e3
        opt.zero_grad(); ub.sum().backward(); opt.step()
        if s % 600 == 0: log.append(f'{s}:{ub.item()/mesh_opt:.3f}')
    print(f'VY_SCALE {vys} lr {lr}: UB / mesh-optimal UB ' + '  '.join(log), f'({time.time()-t0:.0f}s)  [mesh-opt/FE {mesh_opt/fe:.3f}]', flush=True)
