"""Debug the early exits of the t_g-aware refinement (tg_method_probe) on one validation plate."""
import sys, time, traceback
import numpy as np
sys.path.insert(0, "cnla"); sys.path.insert(0, "loops/scratch")
import nlo_refine as R
import nlo_global as G
import nlo_train_fvs
from nlo_core import plate_on_mesh
from tg_method_probe import safe_capacity
name = sys.argv[1] if len(sys.argv) > 1 else "syn0070"
d = np.load("cnla/data/nlo_val.npz", allow_pickle=True)
i = [str(n) for n in d["name"]].index(name)
xs, ys, t = d["x_mm"][i], d["y_mm"][i], np.asarray(d["thickness_mm"][i], float)
sy = float(d["sigma_y_MPa"][i])
model = nlo_train_fvs.load("cnla/data/nlo_fvs_ft30k.pt")
mesh = R.canon()["mesh"]
pl = plate_on_mesh(mesh, xs, ys, t)
cap = safe_capacity(mesh, pl["nodes"], xs, ys, t)
orig = R.nlo_train.plate_on_mesh
R.nlo_train.plate_on_mesh = lambda m, a, b, c: dict(orig(m, a, b, c), thick=cap)
T0 = time.time()
def wrap(mod, nm):
    f = getattr(mod, nm)
    def w(*a, **k):
        t1 = time.time()
        try:
            out = f(*a, **k)
        except Exception:
            print(f"[{time.time()-T0:5.1f}s] {nm} RAISED:"); traceback.print_exc(); raise
        desc = "None" if out is None else type(out).__name__
        if isinstance(out, dict) and "lb" in out: desc += f" lb={out['lb']:.1f}"
        if isinstance(out, dict) and "ub" in out: desc += f" ub={out['ub']:.1f}"
        if isinstance(out, dict) and "tri" in out:
            desc += f" tri {np.asarray(out['tri']).shape} {np.asarray(out['tri']).dtype} equal_canon {np.array_equal(np.asarray(out['tri']), mesh.tri)}"
        print(f"[{time.time()-T0:5.1f}s] {nm} -> {desc} ({time.time()-t1:.2f}s)", flush=True)
        return out
    setattr(mod, nm, w)
for nm in ("combine_lb", "combine_ub", "refine_lb_fvs", "refine_ub_local", "global_stage", "ub_global_stage"):
    wrap(R, nm)
try:
    out = R.bounds(model, dict(xs=xs, ys=ys, t=t, sigma_y=sy))
    print("lb_res lb", out["lb_res"]["lb"], "ub_res ub", out["ub_res"]["ub"])
except Exception:
    traceback.print_exc()
