"""Checks of the FVS C1-cubic Airy layer (cnla/nlo_fvs.py)."""
import os, sys
os.environ.setdefault("OMP_NUM_THREADS", "2")
import numpy as np, torch
sys.path.insert(0, '/home/oldlinux/Research_Projects/Corrosion/PaperX/cnla')
from nlo_core import RefMesh, AiryFit, plate_on_mesh, ZP
from nlo_fvs import FVS, derive_cell, MON3, _mon, to_res_fvs, CELL_TRIS, _DOFS
from eval_limit_analysis import verify

torch.set_default_dtype(torch.float64)
rng = np.random.default_rng(0)

# 1. reproduction of a random cubic on the cell
Bcoef, Mder = derive_cell()
p = rng.normal(size=10)                                         # cubic in MON3
dofs = np.array([_mon(*pt, *dd) @ p for k, pt, dd in _DOFS])
rec = Bcoef @ dofs                                              # (40,)
err = max(np.abs(rec[10 * k:10 * k + 10] - p).max() for k in range(4))
print(f"[1] cubic reproduction on the cell: max coef error {err:.2e}")

# 2. ZP embedding on a small mesh: FVS vertex derivatives == ZP constant derivatives
mesh = RefMesh(8, 5)
fvs = FVS(mesh)
c = torch.tensor(rng.normal(size=(2, 10, 7)))
dzp = mesh.airy_cell_derivs(c)                                  # (B, NX, NY, 4, 3)
V, H, Vt = fvs.from_zp(c)
dfv = fvs.derivs(V, H, Vt)                                      # (B, NX, NY, 4, 3v, 3)
print(f"[2] ZP -> FVS embedding: max |d_fvs - d_zp| = {(dfv - dzp[..., None, :]).abs().max():.2e} "
      f"(scale {dzp.abs().max():.1f})")

# 3. random FVS DOFs give C1 psi: check traction continuity / equilibrium with the verifier on a fake plate
mesh = RefMesh(16, 6)
fvs = FVS(mesh)
L, W = 60.0, 20.0
xs, ys = np.linspace(0, L, 31), np.linspace(0, W, 11)
t = 3 + 0.5 * rng.random((31, 11))
pl = plate_on_mesh(mesh, xs, ys, t)
V = torch.tensor(rng.normal(size=(1, 17, 7, 3)))
H = torch.tensor(rng.normal(size=(1, 16, 7)))
Vt = torch.tensor(rng.normal(size=(1, 17, 6)))
V, H, Vt = fvs.free_edges(V, H, Vt)
Lt, Wt = torch.tensor([L]), torch.tensor([W])
Nv = fvs.stress(V, H, Vt, Lt, Wt)[0].numpy()
v = np.zeros((len(pl["nodes"]), 2)); v[:, 0] = pl["nodes"][:, 0] / L
res = to_res_fvs(mesh, pl, v, Nv, 280.0)
UB, LB, errs, info = verify(res, xs, ys, t, 280.0, "rigid")
P_load = float(fvs.load(V, Wt)[0])
P_edge = float(np.sum(0.5 * (Nv[mesh.right, 0, 0] + Nv[mesh.right, 1, 0])) * W / mesh.NY)
P_left = float(np.sum(0.5 * (Nv[4 * np.arange(6) + 3, 0, 0] + Nv[4 * np.arange(6) + 3, 1, 0])) * W / mesh.NY)
print(f"[3] random FVS field: verify errs={errs}  r_eq {info['r_eq']:.1e} r_cont {info['r_cont']:.1e} "
      f"r_bnd {info['r_bnd']:.1e} ratio {info['yield_ratio']:.4f};  P c/hy {P_load:.6f} right {P_edge:.6f} left {P_left:.6f}")

# 4. AiryFit output (exact free edges in ZP) maps to FVS DOFs already in the free-edge form
mesh = RefMesh(32, 10)
fvs = FVS(mesh)
fit = AiryFit(mesh)
D = torch.tensor(rng.normal(size=(2, 32, 10, 4, 3)))
cz = fit.fit(D)
V, H, Vt = fvs.from_zp(cz)
V2, H2, Vt2 = fvs.free_edges(V, H, Vt)
print(f"[4] AiryFit -> FVS: bottom max {max(V[:, :, 0].abs().max(), H[:, :, 0].abs().max()):.2e};  "
      f"top deviation from (a+bu, b, c) family {max((V - V2).abs().max(), (H - H2).abs().max()):.2e}")
dzp = mesh.airy_cell_derivs(cz)
print(f"    stresses equal: {(fvs.derivs(V2, H2, Vt2) - dzp[..., None, :]).abs().max():.2e}")
