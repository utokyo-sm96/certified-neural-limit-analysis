import sys, time, numpy as np, torch
sys.path.insert(0, '.')
import nlo_train as nt
from nlo_core import RefMesh
mesh = RefMesh(nt.NX, nt.NY)
t = time.time(); mesh.edge_projector('cuda'); print(f'edge basis {time.time()-t:.1f}s', flush=True)
tri = torch.tensor(mesh.tri, device='cuda')
model = nt.UNet(10).cuda(); opt = torch.optim.AdamW(model.parameters(), 1e-3)
B = nt.BATCH
th = 10 + torch.rand(B, len(mesh.unit), device='cuda'); L = torch.full((B,), 236., device='cuda'); W = torch.full((B,), 76., device='cuda')
def step():
    cell, dc = model(nt.features(mesh, th, L, W)); ub, lb = nt.bounds(mesh, tri, cell, dc, th, L, W)
    loss = (ub - lb).mean(); opt.zero_grad(); loss.backward(); opt.step()
for k in range(3):
    t = time.time(); step(); torch.cuda.synchronize(); print(f'warm step {k}: {time.time()-t:.2f}s', flush=True)
t = time.time()
for k in range(20): step()
torch.cuda.synchronize(); print(f'{(time.time()-t)/20*1000:.0f} ms/step', flush=True)
with torch.autograd.profiler.profile(use_cuda=True) as prof:
    step(); torch.cuda.synchronize()
print(prof.key_averages().table(sort_by='cuda_time_total', row_limit=12))
