import sys, time, numpy as np
sys.path.insert(0, '.'); sys.path.insert(0, '../loops/scratch')
import nlo_train, nlo_refine as R
from lb_linear import refine_lb_linear
from nlo_refine_eval import certify_lb
m = nlo_train.load('data/nlo_ckpt_final.pt')
d = np.load('data/la_validation_plates.npz', allow_pickle=True)
fe_lb = [132.11, 157.62, 198.53, 180.42, 264.96, 295.30]
for i in range(int(sys.argv[1])):
    xs, ys, t, sy = d['x_mm'][i], d['y_mm'][i], d['thickness_mm'][i], float(d['sigma_y_MPa'][i])
    out = R.bounds(m, dict(xs=xs, ys=ys, t=t, sigma_y=sy))
    lr = out['lb_res']
    lb0, e0 = certify_lb(lr, xs, ys, t, sy)
    S = lr['stress']; N = S[:, 0, :]
    c = R.canon(); mesh = c['mesh']
    for theta, mt in ((0.9, 1500), (0.9, 3000), (0.8, 6000)):
        t1 = time.time(); S2 = refine_lb_linear(N, lr['nodes'], lr['tri'], lr['thick'], mesh, xs[-1], ys[-1], sy, theta, mt); dt = time.time() - t1
        if S2 is None: print('None'); continue
        lb1, e1 = certify_lb(dict(lr, stress=S2, lb=0.0), xs, ys, t, sy)
        print(f"{d['name'][i]} theta {theta} max_tri {mt}: LB {lb0/1e3:.2f} -> {lb1/1e3:.2f} kN (FE {fe_lb[i]}; -{100*(1-lb0/1e3/fe_lb[i]):.2f}% -> -{100*(1-lb1/1e3/fe_lb[i]):.2f}%) {dt:.1f}s {e1 or 'certified'}", flush=True)
