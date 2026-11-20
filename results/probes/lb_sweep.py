"""LB(time) curves of the FVS rounds for several window caps (dil 1), starting from the combined ZP field."""
import sys, time, numpy as np
sys.path.insert(0, '/home/oldlinux/Research_Projects/Corrosion/PaperX/cnla')
import nlo_train, nlo_refine as R
D = '/home/oldlinux/Research_Projects/Corrosion/PaperX/cnla/data/'
d = np.load(D + 'nlo_val.npz', allow_pickle=True); names = [str(n) for n in d['name']]
model = nlo_train.load(D + 'nlo_ckpt_100k.pt')
cap = {}
orig = R.refine_lb_fvs
def grab(Nv0, th, tri, L, W, sy, **k):
    cap.setdefault('a', (Nv0.copy(), th, tri, L, W, sy)); return None   # stop the built-in rounds; we run our own
R.refine_lb_fvs = grab
for nm in sys.argv[1:]:
    i = names.index(nm); p = dict(xs=d['x_mm'][i], ys=d['y_mm'][i], t=d['thickness_mm'][i], sigma_y=float(d['sigma_y_MPa'][i]))
    cap.clear(); R.bounds(model, p)
    Nv0, th, tri, L, W, sy = cap['a']
    for md in (3000, 6000, 9000, 12000):
        Nv, lb, t0, curve = Nv0, R.lb_v(Nv0, W), time.time(), []
        while time.time() - t0 < 6.0:
            Nl = orig(Nv, th, tri, L, W, sy, theta=0.9, dil=1, max_dof=md)
            if Nl is None or R.lb_v(Nl, W) <= lb * (1 + 1e-4): break
            Nv, lb = Nl, R.lb_v(Nl, W); curve.append((time.time() - t0, lb))
        lb0 = R.lb_v(Nv0, W)
        print(f"{nm} dof {md:5d}: " + " ".join(f"{t:.1f}s:+{100*(l/lb0-1):.2f}%" for t, l in curve), flush=True)
