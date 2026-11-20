import sys, time, numpy as np, torch
sys.path.insert(0, '.')
from nlo_core import RefMesh, AiryFit
from scipy.linalg import cho_solve_banded
from concurrent.futures import ThreadPoolExecutor
mesh = RefMesh(256, 80); fit = AiryFit(mesh)
n = fit.cb.shape[1]; r = np.random.randn(n, 8)
t = time.time(); [fit._band(r) for _ in range(5)]; print(f'banded solve 8 rhs: {(time.time()-t)/5*1000:.1f} ms')
t = time.time(); [fit._band(r[:, :1]) for _ in range(5)]; print(f'banded solve 1 rhs: {(time.time()-t)/5*1000:.1f} ms')
ex = ThreadPoolExecutor(8)
t = time.time()
for _ in range(5): list(ex.map(lambda k: fit._band(r[:, k:k+1]), range(8)))
print(f'threaded 8 x 1 rhs: {(time.time()-t)/5*1000:.1f} ms')
