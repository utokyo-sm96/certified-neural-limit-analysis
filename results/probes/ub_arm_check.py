import sys, time, numpy as np, torch
sys.path.insert(0, '/home/oldlinux/Research_Projects/Corrosion/PaperX/cnla')
import nlo_train_fvs, nlo_refine as R, nlo_global as G
from nlo_core import dissipation, plate_on_mesh
from nlo_refine_eval import certify_ub
D = '/home/oldlinux/Research_Projects/Corrosion/PaperX/cnla/data/'
d = np.load(D + 'nlo_val.npz', allow_pickle=True); names = [str(n) for n in d['name']]
model = nlo_train_fvs.load(D + 'nlo_fvs_ft30k.pt'); mesh = R.canon()['mesh']
t = time.time(); arm = G.UBArm(mesh); print(f"UB arm offline factorisation {time.time()-t:.1f}s, free dofs {len(arm.free)}", flush=True)
for nm in sys.argv[1:]:
    i = names.index(nm); p = dict(xs=d['x_mm'][i], ys=d['y_mm'][i], t=d['thickness_mm'][i], sigma_y=float(d['sigma_y_MPa'][i]))
    R.bounds(model, p); out = R.bounds(model, p)
    pl = plate_on_mesh(mesh, p['xs'], p['ys'], p['t']); th, sy = pl['thick'], p['sigma_y']
    v0 = out['ub_res']['v']; Nlb = np.asarray(out['lb_res']['stress'], float).mean(1)
    vf, y = arm.setup(pl['L'], pl['W'], th, sy, v0, Nlb)
    d_arm = arm.dissipation(arm.g(vf)); d_ref = float(dissipation(torch.tensor(v0[None]), torch.tensor(pl['nodes'][None]), torch.tensor(mesh.tri), torch.tensor(th[None]))[0]) * sy
    eq = np.abs(arm.BfT @ y.reshape(-1)).max() / np.abs(y).max()
    print(f"{nm}: dissipation arm {d_arm/1e3:.4f} vs frozen formula {d_ref/1e3:.4f} kN (hybrid UB {out['ub_res']['ub']/1e3:.4f}); dual equilibrium residual {eq:.1e}", flush=True)
    for secs in (1, 2, 4, 8):
        v, dd, it = G.admm_ub(arm, vf, y, secs)
        res = dict(nodes=pl['nodes'], tri=mesh.tri, thick=th, v=v, ub=dd)
        ub, errs = certify_ub(res, p['xs'], p['ys'], p['t'], sy)
        print(f"   ADMM {secs}s ({it} it): UB {ub/1e3:.3f} kN ({100*(1-ub/out['ub_res']['ub']):+.3f} % lower)  {errs or 'certified'}", flush=True)
