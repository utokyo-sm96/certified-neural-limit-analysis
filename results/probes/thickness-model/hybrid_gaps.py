"""Hybrid-stage gaps (global stages off) of the current nlo_refine (THICK_MODEL tg) on the 32 val plates: calibration of
the adaptive ADMM budgets. Internal method values (the cap-scaled LB and the t_g UB), no certification needed."""
import sys, time, numpy as np
sys.path.insert(0, "cnla")
import nlo_refine as R, nlo_train_fvs
R.THICK_MODEL = "tg"
R.GLOBAL_SECONDS, R.GLOBAL_ITERS, R.UB_GLOBAL_SECONDS, R.UB_GLOBAL_ITERS = 0.0, None, 0.0, None
d = np.load("cnla/data/nlo_val.npz", allow_pickle=True)
model = nlo_train_fvs.load("cnla/data/nlo_fvs_ft30k.pt")
P = lambda i: dict(xs=d["x_mm"][i], ys=d["y_mm"][i], t=np.asarray(d["thickness_mm"][i], float), sigma_y=float(d["sigma_y_MPa"][i]))
R.bounds(model, P(0))
g = []
for i in range(32):
    out = R.bounds(model, P(i))
    g.append(1 - out["lb_res"]["lb"] / out["ub_res"]["ub"])
    print(d["name"][i], f"{100 * g[-1]:.2f} %", flush=True)
np.save("loops/scratch/hybrid_gaps.npy", np.array(g))
print("quantiles", np.quantile(g, [0, .25, .5, .75, .9, 1]).round(4))
