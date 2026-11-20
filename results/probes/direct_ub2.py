import sys, time, numpy as np, torch, torch.nn.functional as F
sys.path.insert(0, '.')
import nlo_train as nt
from nlo_core import *
dev = 'cuda:0'
mesh = RefMesh(nt.NX, nt.NY); NX, NY = nt.NX, nt.NY
d = np.load('data/la_validation_plates.npz', allow_pickle=True)
i = int(sys.argv[1]); mesh_opt = [133.50, 159.09, 199.64, 181.80, 266.62, 296.05][i]
xs, ys, t = d['x_mm'][i], d['y_mm'][i], d['thickness_mm'][i]; sy = float(d['sigma_y_MPa'][i])
pl = plate_on_mesh(mesh, xs, ys, t)
th = torch.tensor(pl['thick'][None], device=dev, dtype=torch.float32)
L = torch.tensor([pl['L']], device=dev); W = torch.tensor([pl['W']], device=dev)
tri = torch.tensor(mesh.tri, device=dev)
nodes = torch.tensor(mesh.unit, device=dev, dtype=torch.float32)[None] * torch.stack([L, W], -1)[:, None, :]
xi = (torch.arange(NX + 1, device=dev) / NX)[None, :, None]
def v_plain(g, sx, sy_):
    gn = F.avg_pool2d(F.pad(g[:, :2], (1, 1, 1, 1), mode='replicate'), 2, stride=1)    # (1, 2, NX+1, NY+1)
    b = 4 * xi * (1 - xi)
    vx = xi.expand(1, NX+1, NY+1) + b * sx * gn[:, 0]; vy = b * sy_ * gn[:, 1]
    grid = torch.stack([vx, vy], -1)
    cen = 0.25 * (grid[:, :-1, :-1] + grid[:, 1:, :-1] + grid[:, 1:, 1:] + grid[:, :-1, 1:]) + 0.1 * g[:, 2:4].permute(0, 2, 3, 1)
    return torch.cat([grid.reshape(1, -1, 2), cen.reshape(1, -1, 2)], 1)
for name, fn, lr in [('plain sx=1 sy=1', lambda g: v_plain(g, 1, 1), 0.01), ('plain sx=1 sy=1', lambda g: v_plain(g, 1, 1), 0.003),
                     ('plain sx=.1 sy=.1', lambda g: v_plain(g, 0.1, 0.1), 0.01), ('cumsum (baseline)', lambda g: nt.velocity(mesh, g), 0.05)]:
    g = torch.zeros(1, 4, NX, NY, device=dev, requires_grad=True); opt = torch.optim.Adam([g], lr=lr); log = []
    for s in range(3001):
        ub = dissipation(fn(g), nodes, tri, th) * sy / 1e3
        opt.zero_grad(); ub.sum().backward(); opt.step()
        if s % 750 == 0: log.append(f'{s}:{ub.item()/mesh_opt:.3f}')
    print(f'{name:18s} lr {lr}: UB/mesh-opt ' + '  '.join(log), flush=True)
