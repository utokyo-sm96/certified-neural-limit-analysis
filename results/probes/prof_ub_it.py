import sys, time, numpy as np
sys.path.insert(0, '/home/oldlinux/Research_Projects/Corrosion/PaperX/cnla')
import nlo_train_fvs, nlo_refine as R, nlo_global as G
from nlo_core import plate_on_mesh
D = '/home/oldlinux/Research_Projects/Corrosion/PaperX/cnla/data/'
d = np.load(D + 'nlo_val.npz', allow_pickle=True); names = [str(n) for n in d['name']]
model = nlo_train_fvs.load(D + 'nlo_fvs_ft30k.pt'); mesh = R.canon()['mesh']; arm = G.UBArm(mesh)
i = names.index('syn0899'); p = dict(xs=d['x_mm'][i], ys=d['y_mm'][i], t=d['thickness_mm'][i], sigma_y=float(d['sigma_y_MPa'][i]))
R.GLOBAL_SECONDS = 0; R.UB_GLOBAL_ITERS = None; R.UB_GLOBAL_SECONDS = 0; out = R.bounds(model, p); pl = plate_on_mesh(mesh, p['xs'], p['ys'], p['t'])
vf, y = arm.setup(pl['L'], pl['W'], pl['thick'], p['sigma_y'], out['ub_res']['v'], np.asarray(out['lb_res']['stress'], float).mean(1))
gg = arm.g(vf); z = gg.copy(); rho = np.linalg.norm(y) / np.linalg.norm(z); u = y / rho
T = dict(rhs=0, solve=0, g=0, diss=0, prox=0, rest=0)
for it in range(20):
    t = time.time(); rhs = arm.BfT @ (z - u - arm.gb.reshape(-1, 4)).reshape(-1); T['rhs'] += time.time() - t
    t = time.time(); vf = arm.solve(rhs); T['solve'] += time.time() - t
    t = time.time(); gg = arm.g(vf); T['g'] += time.time() - t
    t = time.time(); dd = arm.dissipation(gg); T['diss'] += time.time() - t
    t = time.time(); zo = z; ggr = 1.6 * gg - 0.6 * zo; z = arm.prox(ggr + u, rho); T['prox'] += time.time() - t
    t = time.time(); u = u + ggr - z; T['rest'] += time.time() - t
print("ms per UB iteration: " + "  ".join(f"{k} {1000*v/20:.1f}" for k, v in T.items()), f" total {1000*sum(T.values())/20:.1f}")
w = (1.6 * gg - 0.6 * zo + u) * rho; h = w @ arm.Vr
print(f"fraction of triangles outside the cone set in prox: {np.mean(np.einsum('ij,ij->i', (h @ arm.Ein.Q) * arm.Ein.lam, h @ arm.Ein.Q) > arm.Ein.rad**2):.2f}")
