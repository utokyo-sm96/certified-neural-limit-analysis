import sys, time, numpy as np
sys.path.insert(0, '/home/oldlinux/Research_Projects/Corrosion/PaperX/cnla')
import nlo_train_fvs, nlo_refine as R, nlo_global as G
from nlo_core import plate_on_mesh
D = '/home/oldlinux/Research_Projects/Corrosion/PaperX/cnla/data/'
d = np.load(D + 'nlo_val.npz', allow_pickle=True); names = [str(n) for n in d['name']]
model = nlo_train_fvs.load(D + 'nlo_fvs_ft30k.pt'); mesh = R.canon()['mesh']
arm = G.FVSArm(mesh)
i = names.index('syn0899'); p = dict(xs=d['x_mm'][i], ys=d['y_mm'][i], t=d['thickness_mm'][i], sigma_y=float(d['sigma_y_MPa'][i]))
R.GLOBAL_SECONDS = 0; out = R.bounds(model, p); pl = plate_on_mesh(mesh, p['xs'], p['ys'], p['t'])
area, e = G.strains(out['ub_res']['nodes'], mesh.tri, out['ub_res']['v'])
x, y = arm.setup(pl['L'], pl['W'], pl['thick'], p['sigma_y'], np.asarray(out['lb_res']['stress'], float), area, e)
dx = arm.d(x); z = dx.copy(); rho = np.linalg.norm(y) / np.linalg.norm(z); u = y / rho
T = dict(rhs=0, solve=0, Kx=0, proj=0, rest=0)
for it in range(15):
    t = time.time(); rhs = arm.KT @ (z - u).reshape(-1) + arm.g / rho; T['rhs'] += time.time() - t
    t = time.time(); xx = arm.lu.solve(rhs); T['solve'] += time.time() - t
    t = time.time(); dx = arm.d(xx); T['Kx'] += time.time() - t
    t = time.time(); zo = z; dxr = 1.6 * dx + (1 - 1.6) * zo; z = arm.ell.project(dxr + u); T['proj'] += time.time() - t
    t = time.time(); u = u + dxr - z; T['rest'] += time.time() - t
print("ms per iteration: " + "  ".join(f"{k} {1000*v/15:.1f}" for k, v in T.items()), f" total {1000*sum(T.values())/15:.1f}")
L = arm.lu; print(f"LU fill nnz L {L.L.nnz/1e6:.1f} M  U {L.U.nnz/1e6:.1f} M; K nnz {arm.K.nnz/1e6:.1f} M")
