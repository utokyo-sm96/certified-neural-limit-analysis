"""(1) per-plate network gap: FVS fine-tune vs control; (2) ZP vs FVS mesh floor on the hardest val plate."""
import os, sys, csv
os.environ.setdefault("OMP_NUM_THREADS", "4")
import numpy as np
sys.path.insert(0, '/home/oldlinux/Research_Projects/Corrosion/PaperX/cnla')
import nlo_train_fvs as nf
from nlo_refine_eval import certify_lb, certify_ub
from nlo_core import RefMesh, plate_on_mesh, optimal_lb_zp, optimal_ub
from nlo_fvs import FVS, optimal_lb_fvs
D = '/home/oldlinux/Research_Projects/Corrosion/PaperX/cnla/data/'
d = np.load(D + 'nlo_val.npz', allow_pickle=True)
names = [str(n) for n in d['name']]
plates = [dict(xs=d["x_mm"][i], ys=d["y_mm"][i], t=d["thickness_mm"][i], sigma_y=float(d["sigma_y_MPa"][i])) for i in range(len(names))]
gaps = {}
for k in ('fvs0', 'fvs1', 'fvs0_s1', 'fvs1_s1'):
    res = nf.predict(nf.load(D + f'nlo_ft6k_{k}.pt'), plates)
    g = []
    for p, r in zip(plates, res):
        ub, _ = certify_ub(r, p['xs'], p['ys'], p['t'], p['sigma_y']); lb, _ = certify_lb(r, p['xs'], p['ys'], p['t'], p['sigma_y'])
        g.append((ub - lb) / ub)
    gaps[k] = np.array(g)
ctl, fv = (gaps['fvs0'] + gaps['fvs0_s1']) / 2, (gaps['fvs1'] + gaps['fvs1_s1']) / 2
o = np.argsort(-ctl)
print("plate     control  FVS    diff (points)")
for i in o[:8]:
    print(f"{names[i]}  {100*ctl[i]:6.2f}  {100*fv[i]:6.2f}  {100*(fv[i]-ctl[i]):+.2f}")
dd = 100 * (fv - ctl)
print(f"all 32: mean diff {dd.mean():+.3f}, median {np.median(dd):+.3f}, improved {np.sum(dd < 0)}/32; "
      f"top-5 hardest mean diff {dd[o[:5]].mean():+.2f}, rest {dd[o[5:]].mean():+.3f}", flush=True)
i = names.index('syn0769'); p = plates[i]
mesh = RefMesh(256, 80); pl = plate_on_mesh(mesh, p['xs'], p['ys'], p['t'])
UB, _, su, tu = optimal_ub(mesh, pl, p['sigma_y'])
LBz, _, sz, tz = optimal_lb_zp(mesh, pl, p['sigma_y'])
LBf, _, sf, tf = optimal_lb_fvs(mesh, pl, p['sigma_y'], FVS(mesh))
print(f"syn0769 mesh floors: UB_opt {UB/1e3:.2f}  LB ZP {LBz/1e3:.2f} (floor {100*(UB-LBz)/UB:.2f}%)  "
      f"LB FVS {LBf/1e3:.2f} (floor {100*(UB-LBf)/UB:.2f}%)   FE ref [101.56, 102.41]  ({sz}, {sf})")
