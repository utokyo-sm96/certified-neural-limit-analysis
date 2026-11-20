import sys, time, numpy as np, torch
sys.path.insert(0, '.')
import nlo_train as nt
from nlo_core import RefMesh, AiryFit
mesh = RefMesh(256, 80); fit = AiryFit(mesh)
gs = nt.GPUSolve(fit, 'cuda:0')
B = 8
static_in = torch.zeros(B, fit.cb.shape[1] + 4, dtype=torch.float64, device='cuda:0')
s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
with torch.cuda.stream(s):
    for _ in range(3): out = gs.solve(static_in)
torch.cuda.current_stream().wait_stream(s)
g = torch.cuda.CUDAGraph()
with torch.cuda.graph(g):
    static_out = gs.solve(static_in)
rz = np.random.randn(B, fit.cb.shape[1] + 4); ref = fit.solve_np(rz.T).T
static_in.copy_(torch.tensor(rz)); g.replay(); torch.cuda.synchronize()
print('graph rel diff vs CPU', np.abs(static_out.cpu().numpy() - ref).max() / np.abs(ref).max())
t = time.time()
for _ in range(20): static_in.copy_(torch.tensor(rz, device='cuda:0')); g.replay()
torch.cuda.synchronize(); print(f'graph solve 8 rhs: {(time.time()-t)/20*1000:.1f} ms')
