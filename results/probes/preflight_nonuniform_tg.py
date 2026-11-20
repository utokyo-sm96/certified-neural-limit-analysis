"""Pre-flight (no held-out data) for the held-out run under v2: the final method (THICK_MODEL tg) on validation plates
resampled to non-uniform grids like real scans (shorter last grid step in x and y); certified by verifier v2.
Usage: python loops/scratch/preflight_nonuniform_tg.py syn0022 syn0769 ..."""
import sys, time, numpy as np
from scipy.interpolate import RegularGridInterpolator
sys.path.insert(0, "cnla")
import nlo_train_fvs, nlo_refine as R
from verify_v2 import verify_v2
from nlo_refine_eval import SQ3, tri_area
R.THICK_MODEL = "tg"
d = np.load("cnla/data/nlo_val.npz", allow_pickle=True); names = [str(n) for n in d["name"]]
model = nlo_train_fvs.load("cnla/data/nlo_fvs_ft30k.pt")


def run(p):
    t0 = time.time(); out = R.bounds(model, p); dt = time.time() - t0
    u, l = out["ub_res"], out["lb_res"]
    S = np.zeros((len(u["tri"]), 3, 3)); S[:, :, 0] = 1.0
    ub, _, e1, _ = verify_v2(dict(nodes=np.asarray(u["nodes"], float), tri=np.asarray(u["tri"], int), thick=np.asarray(u["thick"], float),
                                  v=np.asarray(u["v"], float), ub=float(u["ub"]), stress=S, lb=0.0, status_ub="Solved", status_lb="Solved"),
                             p["xs"], p["ys"], p["t"], p["sigma_y"])
    nl, tl, thl = np.asarray(l["nodes"], float), np.asarray(l["tri"], int), np.asarray(l["thick"], float)
    vl = np.stack([nl[:, 0] / p["xs"][-1], np.zeros(len(nl))], 1)
    ubd = float(np.sum(p["sigma_y"] * (2 / SQ3) / p["xs"][-1] * tri_area(nl, tl) * thl[tl].mean(1)))
    _, lb, e2, _ = verify_v2(dict(nodes=nl, tri=tl, thick=thl, v=vl, ub=ubd, stress=np.asarray(l["stress"], float), lb=float(l["lb"]),
                                  status_ub="Solved", status_lb="Solved"), p["xs"], p["ys"], p["t"], p["sigma_y"])
    return 100 * (ub - lb) / ub, dt, e1 + e2


first = True
for nm in sys.argv[1:]:
    i = names.index(nm); xs, ys, t = d["x_mm"][i], d["y_mm"][i], np.asarray(d["thickness_mm"][i], float); sy = float(d["sigma_y_MPa"][i])
    f = RegularGridInterpolator((xs, ys), t)
    xs2 = np.append(xs[:-1], xs[-2] + 0.8); ys2 = np.append(ys[:-1], ys[-2] + 1.2)
    X, Y = np.meshgrid(xs2, ys2, indexing="ij"); t2 = f(np.stack([X.ravel(), Y.ravel()], 1)).reshape(X.shape)
    p1 = dict(xs=xs, ys=ys, t=t, sigma_y=sy); p2 = dict(xs=xs2, ys=ys2, t=t2, sigma_y=sy)
    if first:
        R.bounds(model, p1); first = False
    g1, t1, e1 = run(p1); g2, t2_, e2 = run(p2)
    print(f"{nm}: uniform {g1:.2f} % ({t1:.1f} s) {'ok' if not e1 else e1}; non-uniform {g2:.2f} % ({t2_:.1f} s) {'ok' if not e2 else e2}", flush=True)
