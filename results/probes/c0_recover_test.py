import sys, time, numpy as np, scipy.sparse as sp, scipy.sparse.linalg as spla
sys.path.insert(0, '/home/oldlinux/Research_Projects/Corrosion/PaperX/cnla')
from nlo_core import RefMesh
from nlo_fvs import FVS, fvs_stress_matrix
mesh = RefMesh(256, 80); A = fvs_stress_matrix(mesh, FVS(mesh), 240.0, 80.0).tocsc()
n = A.shape[1]; rng = np.random.default_rng(0)
x_true = rng.standard_normal(n); s = A @ x_true
t = time.time(); N = (A.T @ A).tocsc(); print(f"normal matrix nnz {N.nnz/1e6:.1f} M, built {time.time()-t:.1f}s", flush=True)
t = time.time(); lu = spla.splu(N, permc_spec="COLAMD"); print(f"splu {time.time()-t:.1f}s, fill nnz {(lu.L.nnz+lu.U.nnz)/1e6:.1f} M", flush=True)
t = time.time(); x = lu.solve(A.T @ s); print(f"solve {time.time()-t:.2f}s  rel stress residual {np.linalg.norm(A@x - s)/np.linalg.norm(s):.2e}", flush=True)
