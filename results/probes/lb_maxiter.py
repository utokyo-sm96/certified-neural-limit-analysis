import sys, time, numpy as np, clarabel, scipy.sparse as sp
sys.path.insert(0, '/home/oldlinux/Research_Projects/Corrosion/PaperX/cnla')
import nlo_train, nlo_refine as R
D = '/home/oldlinux/Research_Projects/Corrosion/PaperX/cnla/data/'
d = np.load(D + 'nlo_val.npz', allow_pickle=True); names = [str(n) for n in d['name']]
model = nlo_train.load(D + 'nlo_ckpt_100k.pt')
cap = {}; orig = R.refine_lb_fvs; orig_solve = R.solve_cone
def grab(Nv0, th, tri, L, W, sy, **k):
    cap.setdefault('a', (Nv0.copy(), th, tri, L, W, sy)); return None
R.refine_lb_fvs = grab
MI = [None]
def solve_mi(q, A, b, cones):
    st = clarabel.DefaultSettings(); st.verbose = False
    if MI[0]: st.max_iter = MI[0]
    sol = clarabel.DefaultSolver(sp.csc_matrix((len(q), len(q))), q, sp.csc_matrix(A), b, cones, st).solve()
    s = str(sol.status); return np.asarray(sol.x), ("Solved" if "MaxIter" in s else s)
for nm in sys.argv[1:]:
    i = names.index(nm); p = dict(xs=d['x_mm'][i], ys=d['y_mm'][i], t=d['thickness_mm'][i], sigma_y=float(d['sigma_y_MPa'][i]))
    cap.clear(); R.solve_cone = orig_solve; R.refine_lb_fvs = grab; R.bounds(model, p)
    Nv0, th, tri, L, W, sy = cap['a']; lb0 = R.lb_v(Nv0, W)
    R.solve_cone = solve_mi
    for mi in (None, 14, 11, 8):
        MI[0] = mi; Nv, lb, t0, curve = Nv0, lb0, time.time(), []
        while time.time() - t0 < 4.0:
            Nl = orig(Nv, th, tri, L, W, sy, theta=0.9, dil=1, max_dof=6000)
            if Nl is None or R.lb_v(Nl, W) <= lb * (1 + 1e-4): break
            Nv, lb = Nl, R.lb_v(Nl, W); curve.append((time.time() - t0, lb))
        print(f"{nm} max_iter {mi}: " + " ".join(f"{t:.1f}s:+{100*(l/lb0-1):.2f}%" for t, l in curve), flush=True)
