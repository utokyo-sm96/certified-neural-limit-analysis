import os, sys
os.environ.setdefault("OMP_NUM_THREADS", "2")
import numpy as np, torch
sys.path.insert(0, '/home/oldlinux/Research_Projects/Corrosion/PaperX/cnla')
import nlo_train, nlo_train_fvs as nf
from nlo_refine_eval import certify_lb, certify_ub
d = np.load('/home/oldlinux/Research_Projects/Corrosion/PaperX/cnla/data/nlo_val.npz', allow_pickle=True)
plates = [dict(xs=d["x_mm"][i], ys=d["y_mm"][i], t=d["thickness_mm"][i], sigma_y=float(d["sigma_y_MPa"][i])) for i in range(2)]
ck = 'cnla/data/nlo_ckpt_final.pt'
r0 = nlo_train.predict(nlo_train.load(ck), plates)
m = nf.load(ck); m.use_fvs = True                       # zero head -> identical to ZP
r1 = nf.predict(m, plates)
for a, b, p in zip(r0, r1, plates):
    la, _ = certify_lb(a, p["xs"], p["ys"], p["t"], p["sigma_y"]); lb, e = certify_lb(b, p["xs"], p["ys"], p["t"], p["sigma_y"])
    print(f"zero head: LB zp {la:.3f} vs fvs-path {lb:.3f} (rel {lb/la-1:.1e}) errs {e}")
torch.manual_seed(0)
with torch.no_grad():
    m.head_f.weight.normal_(0, 0.05); m.head_f.bias.normal_(0, 0.05)
r2 = nf.predict(m, plates)
for b, p in zip(r2, plates):
    lb, e = certify_lb(b, p["xs"], p["ys"], p["t"], p["sigma_y"]); ub, e2 = certify_ub(b, p["xs"], p["ys"], p["t"], p["sigma_y"])
    print(f"random head: LB {lb:.3f} UB {ub:.3f} errs {e + e2}")
