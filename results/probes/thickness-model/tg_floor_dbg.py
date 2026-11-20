"""Debug the t_g ceiling probe on one plate: interior-point LB vs certified LB, ratios against cap / bilinear / exact."""
import os, sys, time
os.environ.setdefault("OMP_NUM_THREADS", "4")
import numpy as np
sys.path.insert(0, "cnla")
import nlo_refine as R
from nlo_core import RefMesh, plate_on_mesh
from nlo_fvs import FVS, optimal_lb_fvs, to_res_fvs
from verify_v2 import verify_v2
from thickness_tg import tg_quantities
i = int(sys.argv[1]) if len(sys.argv) > 1 else 0
d = np.load("cnla/data/nlo_val.npz", allow_pickle=True)
xs, ys, t, sy, name = d["x_mm"][i], d["y_mm"][i], np.asarray(d["thickness_mm"][i], float), float(d["sigma_y_MPa"][i]), str(d["name"][i])
mesh = RefMesh(256, 80); tri = mesh.tri
pl = plate_on_mesh(mesh, xs, ys, t)
cap = R.tg_context(pl["nodes"], tri, xs, ys, t)["tv"]
LB, Nv, status, sec = optimal_lb_fvs(mesh, pl, sy, FVS(mesh), cap=cap)
f = np.sqrt(np.maximum(Nv[..., 0] ** 2 - Nv[..., 0] * Nv[..., 1] + Nv[..., 1] ** 2 + 3 * Nv[..., 2] ** 2, 0))
print(name, status, f"{sec:.0f} s; IP LB {LB:.1f} N; ratio vs cap {(f / cap).max():.6f}, vs bilinear {(f / pl['thick'][tri]).max():.6f}")
_, own, lam, tpt = tg_quantities(pl["nodes"], tri, xs, ys, t)
Np = (lam[:, :, None] * Nv[own]).sum(1)
fp = np.sqrt(np.maximum(Np[:, 0] ** 2 - Np[:, 0] * Np[:, 1] + Np[:, 1] ** 2 + 3 * Np[:, 2] ** 2, 0))
k = np.argmax(fp / tpt)
print(f"exact ratio {(fp / tpt).max():.6f} at owner tri {own[k]} lam {lam[k]} t_g {tpt[k]:.4f} cap {cap[own[k]]}  f {fp[k]:.4f}  fv {f[own[k]]}")
v = np.zeros((len(pl["nodes"]), 2)); v[:, 0] = pl["nodes"][:, 0] / pl["L"]
res = to_res_fvs(mesh, pl, v, Nv, sy)
_, lb2, errs, info = verify_v2(res, xs, ys, t, sy)
print(f"to_res lb {res['lb']:.1f}; v2 lb {lb2:.1f}; info {{k: info[k] for k in ('yield_ratio', 'yield_ratio_v2', 'lb_v1') if k in info}}".replace("{{", "{").replace("}}", "}"), errs)
print({k: info[k] for k in info if 'ratio' in k or 'lb' in k})
