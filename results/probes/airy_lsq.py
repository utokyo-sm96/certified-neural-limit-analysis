import sys, time, numpy as np, torch, scipy.sparse as sp
from scipy.linalg import cholesky_banded, cho_solve_banded
sys.path.insert(0, '.')
import nlo_train as nt
from nlo_core import RefMesh, plate_on_mesh, yield_ratio, resultant, zp_stress_matrix, ZP
dev = 'cuda:0'
mesh = RefMesh(nt.NX, nt.NY); NX, NY = nt.NX, nt.NY; nc = (NX+2)*(NY+2)
A = zp_stress_matrix(mesh, NX, NY).tocsr()                 # c -> cell-unit (ww, uu, -uw)
cid = np.arange(nc).reshape(NX+2, NY+2); rows = []
for j, k in ((0, 0), (NY-1, 2)):
    for comp in (1, 2):
        for i in range(NX):
            r = {}
            for a in range(3):
                for b in range(3):
                    if ZP[k, comp, a, b]: r[cid[i+a, j+b]] = r.get(cid[i+a, j+b], 0) + ZP[k, comp, a, b]
            rows.append(r)
Bc = sp.csr_matrix((np.concatenate([list(r.values()) for r in rows]), (np.repeat(np.arange(len(rows)), [len(r) for r in rows]), np.concatenate([list(r.keys()) for r in rows]))), shape=(len(rows), nc))
lam = float(sys.argv[2]) if len(sys.argv) > 2 else 10.0
K = (A.T @ A + lam * (Bc.T @ Bc)).tolil()
for q in (cid[0, 0], cid[1, 0], cid[0, 1], cid[1, 1]): K[q, q] += 1.0   # pin affine + ZP-dependency null space
K = K.tocsr()
u = 2 * (NY + 2) + 2
t0 = time.time()
Kd = K.todia(); ab = np.zeros((u + 1, nc))
for off, dat in zip(Kd.offsets, Kd.data):
    if 0 <= off <= u: ab[u - off, off:] = dat[off:]
assert max(Kd.offsets) <= u
cb = cholesky_banded(ab); print(f'banded Cholesky n={nc} u={u}: {time.time()-t0:.2f}s', flush=True)
rhs = np.random.randn(nc, 8); t0 = time.time(); x = cho_solve_banded((cb, False), rhs); print(f'solve 8 rhs {1000*(time.time()-t0):.0f} ms, resid {np.abs(K@x-rhs).max()/np.abs(rhs).max():.1e}', flush=True)
class Solve(torch.autograd.Function):
    @staticmethod
    def forward(ctx, rhs):
        return torch.from_numpy(cho_solve_banded((cb, False), rhs.detach().double().cpu().numpy().T).T.copy()).to(rhs.device, rhs.dtype)
    @staticmethod
    def backward(ctx, g):
        return Solve.apply(g)
At = torch.sparse_csr_tensor(torch.tensor(A.indptr), torch.tensor(A.indices), torch.tensor(A.data), size=A.shape).to(dev)
AtT = torch.tensor(0)
A_T = A.T.tocsr(); AT = torch.sparse_csr_tensor(torch.tensor(A_T.indptr), torch.tensor(A_T.indices), torch.tensor(A_T.data), size=A_T.shape).to(dev)
d = np.load('data/la_validation_plates.npz', allow_pickle=True)
i = int(sys.argv[1]); ref = [132.11, 157.62, 198.53, 180.42, 264.96, 295.30][i]
xs, ys, t = d['x_mm'][i], d['y_mm'][i], d['thickness_mm'][i]; sy = float(d['sigma_y_MPa'][i])
pl = plate_on_mesh(mesh, xs, ys, t)
th = torch.tensor(pl['thick'][None], device=dev, dtype=torch.float64)
L = torch.tensor([pl['L']], device=dev, dtype=torch.float64); W = torch.tensor([pl['W']], device=dev, dtype=torch.float64)
hx, hy = L / NX, W / NY
tri = torch.tensor(mesh.tri, device=dev); tmin = th[:, tri].min(-1).values; tmean = th[:, tri].mean(-1)
def lb_of(sig, soft=None):
    Ns = tmean[..., None] * sig                                      # target, physical (B, T, 3)
    Nc = torch.stack([Ns[..., 0] * hy**2, Ns[..., 1] * hx**2, Ns[..., 2] * hx * hy], -1).reshape(1, -1)
    c = Solve.apply((AT @ Nc.T).T)                                    # (1, nc) least-squares Airy coefficients
    c = mesh.project_edges(c.reshape(1, NX+2, NY+2))
    N = mesh.stress(c, L, W)
    r = yield_ratio(N, tmin)
    if soft is None: return float(resultant(mesh, N, W)[0] / r.max()) * sy / 1e3
    m = r.max(-1).values.detach(); rs = m * soft * torch.logsumexp(r / (m[:, None] * soft), -1)
    return resultant(mesh, N, W) / rs
sig0 = torch.zeros(1, mesh.T, 3, device=dev, dtype=torch.float64); sig0[..., 0] = 1
print(f'start N* = t (1,0,0): LB/FE = {lb_of(sig0)/ref:.3f}', flush=True)
for tau, lr in [(0.05, 0.01), (0.05, 0.003)]:
    ds = torch.zeros_like(sig0, requires_grad=True); opt = torch.optim.Adam([ds], lr=lr); log = []; t0 = time.time()
    for s in range(1501):
        loss = -lb_of(sig0 + ds, tau).mean(); opt.zero_grad(); loss.backward(); opt.step()
        if s % 300 == 0:
            with torch.no_grad(): log.append(f'{s}:{lb_of(sig0 + ds)/ref:.3f}')
    print(f'tau {tau} lr {lr}: ' + '  '.join(log), f'({time.time()-t0:.0f}s)', flush=True)
