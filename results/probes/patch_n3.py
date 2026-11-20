"""Patch nlo_train.py with idea N3: hard-plate sampling (probability ~ running per-plate gap, mixed 50/50 with uniform)."""
p = '/home/oldlinux/Research_Projects/Corrosion/PaperX/cnla/nlo_train.py'
s = open(p).read()
old = "        ii = rng.integers(0, len(L_np), BATCH)\n"
assert old in s
s = s.replace(old, '''        prob = 0.5 / len(score) + 0.5 * score / score.sum()             # N3: hard plates more often
        ii = rng.choice(len(score), BATCH, p=prob)
''')
old2 = "    rng = np.random.default_rng(SEED)\n"
assert old2 in s
s = s.replace(old2, old2 + "    score = np.full(len(L_np), 0.3)                                     # running certified-gap estimate per plate\n", 1)
old3 = "        hist.append([ub.mean().item(), lb.mean().item()])\n"
assert old3 in s
s = s.replace(old3, old3 + '''        g = (1 - lb / ub).detach().cpu().numpy()
        score[ii] = 0.7 * score[ii] + 0.3 * np.clip(g, 0.01, 1.0)
''')
open(p, 'w').write(s)
print("patched")
