"""F2 probe: k ZP rounds (refine_lb_local) before the FVS rounds; LB at equal LB-phase time."""
import sys, time, numpy as np
sys.path.insert(0, '/home/oldlinux/Research_Projects/Corrosion/PaperX/cnla')
import nlo_train, nlo_refine as R
D = '/home/oldlinux/Research_Projects/Corrosion/PaperX/cnla/data/'
d = np.load(D + 'nlo_val.npz', allow_pickle=True); names = [str(n) for n in d['name']]
model = nlo_train.load(D + 'nlo_ckpt_100k.pt')
cap = {}; orig = R.refine_lb_fvs
def grab(Nv0, th, tri, L, W, sy, **k):
    cap.setdefault('a', (Nv0.copy(), th, tri, L, W, sy)); return None
R.refine_lb_fvs = grab; R.refine_ub_local = lambda *a, **k: None
for nm in sys.argv[1:]:
    i = names.index(nm); p = dict(xs=d['x_mm'][i], ys=d['y_mm'][i], t=d['thickness_mm'][i], sigma_y=float(d['sigma_y_MPa'][i]))
    cap.clear(); R.bounds(model, p)
    Nv0, th, tri, L, W, sy = cap['a']; lb0 = R.lb_v(Nv0, W); mesh = R.canon()["mesh"]
    for kzp in (0, 1, 2):
        t0 = time.time(); N = Nv0[:, 0, :].copy(); curve = []
        for _ in range(kzp):
            Nl = R.refine_lb_local(N, th, tri, L, W, sy, theta=0.9, dil=2, max_coef=6000)
            if Nl is None or Nl[mesh.right, 0].sum() <= N[mesh.right, 0].sum() * (1 + 1e-4): break
            N = Nl
        Nv = np.repeat(N[:, None, :], 3, 1); lb = R.lb_v(Nv, W); curve.append((time.time() - t0, lb))
        while time.time() - t0 < 4.0:
            Nl = orig(Nv, th, tri, L, W, sy, theta=0.9, dil=1, max_dof=6000)
            if Nl is None or R.lb_v(Nl, W) <= lb * (1 + 1e-4): break
            Nv, lb = Nl, R.lb_v(Nl, W); curve.append((time.time() - t0, lb))
        print(f"{nm} ZP rounds {kzp}: " + " ".join(f"{t:.1f}s:+{100*(l/lb0-1):.2f}%" for t, l in curve), flush=True)
