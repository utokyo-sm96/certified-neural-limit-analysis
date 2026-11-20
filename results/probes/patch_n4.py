"""Patch nlo_train.py with idea N4: anneal the soft-max temperature from 0.01 to 0.002 over training."""
p = '/home/oldlinux/Research_Projects/Corrosion/PaperX/cnla/nlo_train.py'
s = open(p).read()
old = "def bounds(mesh, fit, tri, out_v, out_s, thick, L, W, soft=True):"
assert old in s
s = s.replace(old, "def bounds(mesh, fit, tri, out_v, out_s, thick, L, W, soft=True, tau=None):")
old2 = "        rmax = m * TAU * torch.logsumexp(r / (m[:, None] * TAU), -1)"
assert old2 in s
s = s.replace(old2, "        tt = TAU if tau is None else tau\n        rmax = m * tt * torch.logsumexp(r / (m[:, None] * tt), -1)")
old3 = "        ub, lb = bounds(mesh, fit, tri, out_v, out_s, th, L, W)"
assert old3 in s
s = s.replace(old3, "        ub, lb = bounds(mesh, fit, tri, out_v, out_s, th, L, W, tau=0.01 * (0.2 ** frac))   # N4: 0.01 -> 0.002")
open(p, 'w').write(s)
print("patched")
