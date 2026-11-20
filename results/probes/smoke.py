import sys, time, numpy as np
sys.path.insert(0, '.')
import nlo_train
from eval_limit_analysis import verify
m = nlo_train.load()
d = np.load('data/la_validation_plates.npz', allow_pickle=True)
pl = [dict(xs=d['x_mm'][i], ys=d['y_mm'][i], t=d['thickness_mm'][i], sigma_y=float(d['sigma_y_MPa'][i])) for i in range(6)]
t0 = time.time(); res = nlo_train.predict(m, pl); dt = time.time() - t0
for p, r, nm in zip(pl, res, d['name']):
    r['status_ub'] = r['status_lb'] = 'Solved'
    ub, lb, errs, info = verify(r, p['xs'], p['ys'], p['t'], p['sigma_y'], 'rigid')
    print(nm, f'UB {ub/1e3:.1f} LB {lb/1e3:.1f} gap {100*(ub-lb)/ub:.1f}%', errs)
print(f'predict {1000*dt/6:.0f} ms/plate')
