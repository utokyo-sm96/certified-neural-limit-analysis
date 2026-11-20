import sys, numpy as np, scipy.sparse as sp, scipy.sparse.linalg as spl
sys.path.insert(0, '.')
from nlo_core import RefMesh, zp_stress_matrix, ZP
NX, NY = 64, 20
mesh = RefMesh(NX, NY); nc = (NX+2)*(NY+2)
A = zp_stress_matrix(mesh, NX, NY).tocsr()
cid = np.arange(nc).reshape(NX+2, NY+2)
q = np.arange(NY+2); c_true = np.tile(0.5*q**2, (NX+2, 1)).ravel()
Nt = A @ c_true
print('target from exact quadratic: ww', Nt[0::3].min(), Nt[0::3].max(), 'uu', abs(Nt[1::3]).max(), 'uw', abs(Nt[2::3]).max())
K = (A.T @ A).tolil()
for qq in (cid[0, 0], cid[1, 0], cid[0, 1]): K[qq, qq] += 1.0
K = K.tocsc()
x = spl.spsolve(K, A.T @ Nt)
print('fit residual (spsolve):', np.abs(A @ x - Nt).max())
ev = np.linalg.eigvalsh((A.T @ A).toarray()); print('AtA smallest eigenvalues', ev[:6], 'largest', ev[-1])
