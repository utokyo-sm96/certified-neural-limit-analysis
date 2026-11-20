import sys, time, numpy as np
sys.path.insert(0, '.'); sys.path.insert(0, '../loops/scratch')
import nlo_train, nlo_refine as R
from lb_coarse import refine_lb_coarse
from nlo_refine_eval import certify_lb
m = nlo_train.load('data/nlo_ckpt_final.pt')
d = np.load('data/la_validation_plates.npz', allow_pickle=True)
fe_lb = [132.11, 157.62, 198.53, 180.42, 264.96, 295.30]
for i in range(int(sys.argv[1])):
    xs, ys, t, sy = d['x_mm'][i], d['y_mm'][i], d['thickness_mm'][i], float(d['sigma_y_MPa'][i])
    out = R.bounds(m, dict(xs=xs, ys=ys, t=t, sigma_y=sy))
    lr = out['lb_res']; lb0, _ = certify_lb(lr, xs, ys, t, sy)
    N = lr['stress'][:, 0, :]; mesh = R.canon()['mesh']; A_cell = R.airy_cell_matrix()
    for ncx, ncy in ((17, 6), (33, 11), (65, 21)):
        t1 = time.time(); N2 = refine_lb_coarse(N, lr['thick'], lr['tri'], mesh, A_cell, xs[-1], ys[-1], sy, ncx, ncy); dt = time.time() - t1
        lb1, e1 = certify_lb(dict(lr, stress=np.repeat(N2[:, None, :], 3, 1), lb=0.0), xs, ys, t, sy)
        # then the existing local fine correction on top
        t2 = time.time()
        for _ in range(4):
            Nl = R.refine_lb_local(N2, lr['thick'], lr['tri'], xs[-1], ys[-1], sy, theta=0.9, dil=2, max_coef=6000)
            if Nl is None or Nl[mesh.right, 0].sum() <= N2[mesh.right, 0].sum() * (1 + 1e-4): break
            N2 = Nl
        lb2, e2 = certify_lb(dict(lr, stress=np.repeat(N2[:, None, :], 3, 1), lb=0.0), xs, ys, t, sy)
        print(f"{d['name'][i]} coarse {ncx}x{ncy}: LB -{100*(1-lb0/1e3/fe_lb[i]):.2f}% -> coarse -{100*(1-lb1/1e3/fe_lb[i]):.2f}% ({dt:.1f}s) -> +local -{100*(1-lb2/1e3/fe_lb[i]):.2f}% ({time.time()-t2:.1f}s)  {(e1 + e2) or 'certified'}", flush=True)
