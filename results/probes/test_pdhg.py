import sys, time, numpy as np, torch
sys.path.insert(0, '.'); sys.path.insert(0, '../loops/scratch')
import nlo_train, nlo_refine as R
from pdhg_lb import pdhg_lb
from nlo_core import AiryFit
from nlo_refine_eval import certify_lb
m = nlo_train.load('data/nlo_ckpt_final.pt')
d = np.load('data/la_validation_plates.npz', allow_pickle=True)
fe_lb = [132.11, 157.62, 198.53, 180.42, 264.96, 295.30]; zp = [129.68, 154.98, 194.98, 178.29, 261.53, 293.00]
mesh = R.canon()['mesh']; fit = AiryFit(mesh)
for i in range(int(sys.argv[1])):
    xs, ys, t, sy = d['x_mm'][i], d['y_mm'][i], d['thickness_mm'][i], float(d['sigma_y_MPa'][i])
    out = R.bounds(m, dict(xs=xs, ys=ys, t=t, sigma_y=sy)); lr = out['lb_res']
    lb0, _ = certify_lb(lr, xs, ys, t, sy)
    N0 = lr['stress'][:, 0, :]
    t1 = time.time(); N, hist = pdhg_lb(N0, lr['thick'], lr['tri'], mesh, fit.top, None, xs[-1], ys[-1], sy, iters=int(sys.argv[2]), fit=fit, v=out['ub_res']['v'], nodes=out['ub_res']['nodes']); dt = time.time() - t1
    lb1, e = certify_lb(dict(lr, stress=np.repeat(N[:, None, :], 3, 1), lb=0.0), xs, ys, t, sy)
    print(f"{d['name'][i]}: LB -{100*(1-lb0/1e3/fe_lb[i]):.2f}% -> PDHG -{100*(1-lb1/1e3/fe_lb[i]):.2f}% (ZP floor -{100*(1-zp[i]/fe_lb[i]):.2f}%) {dt:.1f}s {e or 'certified'}")
    print('   ', '  '.join(f"{it}:{100*(1-lb/1e3/fe_lb[i]):.2f}%@{tt:.1f}s" for it, tt, lb in hist[::max(1, len(hist)//10)]), flush=True)
