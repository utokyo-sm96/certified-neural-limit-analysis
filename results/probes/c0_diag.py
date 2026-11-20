"""C0 diagnosis: condition number of the scaled LB operator, and PDLP with a warm-start-balanced primal weight."""
import sys, time, numpy as np, scipy.sparse.linalg as spla, torch
sys.path.insert(0, '/home/oldlinux/Research_Projects/Corrosion/PaperX/loops/scratch')
import c0_pdlp as C
import nlo_train_fvs, nlo_refine as R
from nlo_fvs import FVS
D = C.PF + '/data/'
d = np.load(D + 'nlo_val.npz', allow_pickle=True); names = [str(n) for n in d['name']]
model = nlo_train_fvs.load(D + 'nlo_fvs_ft30k.pt'); mesh = R.canon()["mesh"]; fvs = FVS(mesh)
i = names.index('syn0903'); plate = dict(xs=d['x_mm'][i], ys=d['y_mm'][i], t=d['thickness_mm'][i], sigma_y=float(d['sigma_y_MPa'][i]))
out = R.bounds(model, plate); S = C.setup(plate, out, mesh, fvs)
for nm, Kx in (("K (unscaled)", S["K"]), ("Ks (Ruiz + PC)", S["Ks"])):
    N = (Kx.T @ Kx).tocsc()
    smax = np.sqrt(spla.eigsh(N, k=1, which="LA", return_eigenvectors=False)[0])
    lu = spla.splu(N, permc_spec="COLAMD")
    v = np.random.default_rng(0).standard_normal(N.shape[0])
    for _ in range(60):
        v = lu.solve(v); v /= np.linalg.norm(v)
    smin = np.sqrt(v @ (N @ v))
    print(f"{nm}: sigma_max {smax:.3e}  sigma_min {smin:.3e}  condition {smax/smin:.2e}", flush=True)
