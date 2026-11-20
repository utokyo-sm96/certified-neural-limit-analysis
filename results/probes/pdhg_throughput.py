"""Plan C feasibility: GPU cost of one PDHG iteration on the FVS lower-bound problem (256x80 mesh), float32 vs float64.
Operator: sparse map free FVS unknowns -> per-vertex stresses (nlo_fvs.fvs_stress_matrix), K x, K^T y, and the
projection onto 3T four-dimensional second-order cones (yield at every triangle vertex)."""
import sys, time, numpy as np, torch
sys.path.insert(0, '/home/oldlinux/Research_Projects/Corrosion/PaperX/cnla')
from nlo_core import RefMesh
from nlo_fvs import FVS, fvs_stress_matrix
mesh = RefMesh(256, 80); A = fvs_stress_matrix(mesh, FVS(mesh), 240.0, 80.0).tocsr()
print(f"operator {A.shape}, nnz {A.nnz/1e6:.2f} M, cones {A.shape[0]//3}")
dev = "cuda:0"
def soc_proj(z):                       # z (m, 4): project onto {(t, x): |x| <= t}
    t, x = z[:, 0], z[:, 1:]; nx = x.norm(dim=1)
    a = (t + nx) / 2; out = torch.where((nx <= t)[:, None], z, torch.zeros_like(z))
    mid = (nx > t.abs())
    out[mid, 0] = a[mid]; out[mid, 1:] = x[mid] * (a[mid] / nx[mid])[:, None]
    return out
for dt in (torch.float32, torch.float64):
    K = torch.sparse_csr_tensor(torch.tensor(A.indptr), torch.tensor(A.indices), torch.tensor(A.data), size=A.shape, dtype=dt, device=dev)
    KT = torch.sparse_csr_tensor(*(lambda B: (torch.tensor(B.indptr), torch.tensor(B.indices), torch.tensor(B.data)))(A.T.tocsr()), size=A.shape[::-1], dtype=dt, device=dev)
    x = torch.randn(A.shape[1], dtype=dt, device=dev); y = torch.randn(A.shape[0], dtype=dt, device=dev)
    m = A.shape[0] // 3; tz = torch.rand(m, dtype=dt, device=dev)
    for _ in range(5):
        Kx = K @ x; z = soc_proj(torch.cat([tz[:, None], Kx.view(m, 3)], 1)); g = KT @ y
    torch.cuda.synchronize(); t0 = time.time(); n = 300
    for _ in range(n):
        Kx = K @ x; z = soc_proj(torch.cat([tz[:, None], Kx.view(m, 3)], 1)); y = y + 0.1 * (Kx - z[:, 1:].reshape(-1)); g = KT @ y; x = x - 1e-3 * g
    torch.cuda.synchronize(); print(f"{str(dt):14s}: {1000*(time.time()-t0)/n:.3f} ms per PDHG-type iteration")
