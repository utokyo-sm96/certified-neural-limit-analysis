import sys, numpy as np, torch
sys.path.insert(0, '.')
import nlo_train as nt
from nlo_core import *
m = nt.load()
mesh = RefMesh(nt.NX, nt.NY); fit = AiryFit(mesh)
d = np.load('data/la_validation_plates.npz', allow_pickle=True)
refs = [132.11, 157.62, 198.53, 180.42, 264.96, 295.30]
for i in range(6):
    xs, ys, t = d['x_mm'][i], d['y_mm'][i], d['thickness_mm'][i]; sy = float(d['sigma_y_MPa'][i])
    pl = plate_on_mesh(mesh, xs, ys, t)
    th = torch.tensor(pl['thick'][None]); L = torch.tensor([pl['L']], dtype=torch.float64); W = torch.tensor([pl['W']], dtype=torch.float64)
    with torch.no_grad():
        ov, os_ = m(nt.features(mesh, th.cuda(), L.cuda(), W.cuda()))
    tri = torch.tensor(mesh.tri)
    N = nt.stress(mesh, fit, tri, os_.double().cpu(), th, L, W)
    r = yield_ratio(N, th[:, tri].min(-1).values)[0]
    P = resultant(mesh, N, W)[0]
    lb = float(P / r.max()) * sy / 1e3
    e = int(r.argmax()); cell, k = divmod(e, 4); ci, cj = divmod(cell, nt.NY)
    rs = r / r.max()
    # section of the max: fraction of that column's triangles above 0.9
    col = rs.reshape(nt.NX, nt.NY, 4)
    near = [(col[ii] > 0.9).float().mean().item() for ii in range(nt.NX)]
    best_col = int(np.argmax(near))
    tc = th[0, mesh.G:].reshape(nt.NX, nt.NY)
    weakest = int(tc.mean(1).argmin())
    print(f"{d['name'][i]} LB/FE {lb/refs[i]:.3f}  max ratio at cell ({ci},{cj}) k={k}; tri>0.95: {(rs>0.95).sum().item()}, >0.8: {(rs>0.8).sum().item()}; "
          f"most-plastic column {best_col} ({100*max(near):.0f}% >0.9); weakest section column {weakest}; t at max {th[0, tri[e]].min().item():.2f} (plate min {th.min().item():.2f})")
