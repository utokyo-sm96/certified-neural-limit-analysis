import sys, time, numpy as np
sys.path.insert(0, '.'); sys.path.insert(0, '../loops/scratch')
import nlo_train, nlo_refine as R
from lb_full import refine_lb_full
from nlo_refine_eval import certify_lb
m = nlo_train.load('data/nlo_ckpt_final.pt')
d = np.load('data/la_validation_plates.npz', allow_pickle=True)
fe_lb = [132.11, 157.62, 198.53, 180.42, 264.96, 295.30]; zp_floor = [129.68, 154.98, 194.98, 178.29, 261.53, 293.00]
for i in range(int(sys.argv[1])):
    xs, ys, t, sy = d['x_mm'][i], d['y_mm'][i], d['thickness_mm'][i], float(d['sigma_y_MPa'][i])
    out = R.bounds(m, dict(xs=xs, ys=ys, t=t, sigma_y=sy))
    lr = out['lb_res']; lb0, _ = certify_lb(lr, xs, ys, t, sy)
    N = lr['stress'][:, 0, :]; mesh = R.canon()['mesh']; A_cell = R.airy_cell_matrix()
    from nlo_core import AiryFit
    top = R.canon().setdefault('fit', AiryFit(mesh)).top
    for rho in (1.0, 10.0):
        t1 = time.time(); N2, log = refine_lb_full(N, lr['thick'], lr['tri'], mesh, A_cell, xs[-1], ys[-1], sy, top=top, mu=rho, rounds=10, budget=30); dt = time.time() - t1
        lb1, e1 = certify_lb(dict(lr, stress=np.repeat(N2[:, None, :], 3, 1), lb=0.0), xs, ys, t, sy)
        print(f"{d['name'][i]} mu {rho}: LB -{100*(1-lb0/1e3/fe_lb[i]):.2f}% -> -{100*(1-lb1/1e3/fe_lb[i]):.2f}% (ZP floor -{100*(1-zp_floor[i]/fe_lb[i]):.2f}%) {dt:.1f}s {e1 or 'certified'}", flush=True)
        print('   ', ' | '.join(log), flush=True)
