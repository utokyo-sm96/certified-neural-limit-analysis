import sys, time, numpy as np, torch
sys.path.insert(0, '.')
import nlo_train as nt
from nlo_core import RefMesh, AiryFit
mesh = RefMesh(256, 80); fit = AiryFit(mesh)
t = time.time(); gs = nt.GPUSolve(fit, 'cuda:0'); print(f'GPUSolve setup {time.time()-t:.1f}s')
rz = np.random.randn(8, fit.cb.shape[1] + 4)
ref = fit.solve_np(rz.T).T
g = gs.solve(torch.tensor(rz, device='cuda:0')); torch.cuda.synchronize()
t = time.time()
for _ in range(10): g = gs.solve(torch.tensor(rz, device='cuda:0'))
torch.cuda.synchronize(); print(f'GPU solve 8 rhs: {(time.time()-t)/10*1000:.1f} ms; rel diff vs CPU {np.abs(g.cpu().numpy()-ref).max()/np.abs(ref).max():.1e}')
