"""FVS-optimal LB on the 256x80 mesh vs the step-1 linear-stress LB (same space: should coincide)."""
import os, sys, time
os.environ.setdefault("OMP_NUM_THREADS", "2")
import numpy as np
sys.path.insert(0, '/home/oldlinux/Research_Projects/Corrosion/PaperX/cnla')
from nlo_core import RefMesh, plate_on_mesh
from nlo_fvs import FVS, optimal_lb_fvs, to_res_fvs
from eval_limit_analysis import verify
REF = {'AN-syn0': 132.24, 'AN-syn1': 157.87, 'AN-syn2': 198.56, 'AM-syn3': 180.56, 'AM-syn4': 264.95, 'AM-syn5': 295.35}
d = np.load('/home/oldlinux/Research_Projects/Corrosion/PaperX/cnla/data/la_validation_plates.npz', allow_pickle=True)
mesh = RefMesh(256, 80); fvs = FVS(mesh)
for i in [int(a) for a in sys.argv[1:]]:
    xs, ys, t = d['x_mm'][i], d['y_mm'][i], d['thickness_mm'][i]; sy = float(d['sigma_y_MPa'][i]); name = str(d['name'][i])
    pl = plate_on_mesh(mesh, xs, ys, t)
    LB, Nv, st, sec = optimal_lb_fvs(mesh, pl, sy, fvs)
    v = np.zeros((len(pl['nodes']), 2)); v[:, 0] = pl['nodes'][:, 0] / pl['L']
    res = to_res_fvs(mesh, pl, v, Nv, sy)
    _, LBc, errs, info = verify(res, xs, ys, t, sy, 'rigid')
    print(f"{name}: FVS-optimal LB {LB/1e3:.3f} kN ({st}, {sec:.0f}s), certified {LBc/1e3:.3f} kN, step-1 linear {REF[name]:.2f} kN "
          f"({100*(LBc/1e3/REF[name]-1):+.3f}%), errs {errs}, r_eq {info['r_eq']:.1e} r_cont {info['r_cont']:.1e} ratio {info['yield_ratio']:.6f}", flush=True)
