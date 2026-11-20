import os, sys, time, pickle
import numpy as np
sys.path.insert(0, '/home/oldlinux/Research_Projects/Corrosion/PaperX/loops/scratch')
sys.path.insert(0, '/home/oldlinux/Research_Projects/Corrosion/PaperX/cnla')
import clarabel, scipy.sparse as sp
import nlo_train
import nlo_refine_f1 as R
D = '/home/oldlinux/Research_Projects/Corrosion/PaperX/cnla/data/'
d = np.load(D + 'nlo_val.npz', allow_pickle=True); names = [str(n) for n in d['name']]
model = nlo_train.load(D + 'nlo_ckpt_100k.pt')
probs = []
orig = R.solve_cone
def cap(q, A, b, cones):
    probs.append((q, sp.csc_matrix(A), b, cones)); return orig(q, A, b, cones)
R.solve_cone = cap
i = names.index('syn0899'); p = dict(xs=d['x_mm'][i], ys=d['y_mm'][i], t=d['thickness_mm'][i], sigma_y=float(d['sigma_y_MPa'][i]))
R.bounds(model, p)
print(len(probs), "problems:", [A.shape for _, A, _, _ in probs])
def solve(q, A, b, cones, **kw):
    st = clarabel.DefaultSettings(); st.verbose = False
    for k, v in kw.items(): setattr(st, k, v)
    t = time.time(); s = clarabel.DefaultSolver(sp.csc_matrix((len(q), len(q))), q, A, b, cones, st).solve()
    return time.time() - t, s.iterations, float(q @ np.asarray(s.x)), str(s.status)
for name, kw in [("default", {}), ("tol1e-6", dict(tol_gap_abs=1e-6, tol_gap_rel=1e-6, tol_feas=1e-6, tol_ktratio=1e-5)),
                 ("tol1e-5", dict(tol_gap_abs=1e-5, tol_gap_rel=1e-5, tol_feas=1e-5, tol_ktratio=1e-4)),
                 ("faer", dict(direct_solve_method="faer")), ("faer1t", dict(direct_solve_method="faer", max_threads=1))]:
    try:
        res = [solve(*pr, **kw) for pr in probs]
        print(f"{name:8s} total {sum(r[0] for r in res):.2f}s  " + " ".join(f"{r[0]:.2f}/{r[1]}it" for r in res) + "  obj " + " ".join(f"{r[2]:.6g}" for r in res[1:5]), flush=True)
    except Exception as e:
        print(name, "error", e)
