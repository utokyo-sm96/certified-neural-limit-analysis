"""Patch nlo_train.py with idea N1: parallel banded Airy solves in a forked process pool (training only)."""
p = '/home/oldlinux/Research_Projects/Corrosion/PaperX/cnla/nlo_train.py'
s = open(p).read()
anchor = "# ----------------------------------------------------------------------------- train / load / predict"
add = '''# ----------------------------------------------------------------------------- parallel Airy solves (training)
_CB = None


def _pool_init(cb):
    global _CB
    _CB = cb


def _pool_band(r):
    from scipy.linalg import cho_solve_banded
    return cho_solve_banded((_CB, False), r)


def parallel_solver(fit, workers):
    """AiryFit.solve_np with the banded solves of the right-hand sides spread over a forked process pool
    (scipy keeps the GIL, so threads do not help). Fork before CUDA is initialised."""
    import multiprocessing as mp
    pool = mp.get_context("fork").Pool(workers, initializer=_pool_init, initargs=(fit.cb,))

    def solve_np(r):
        ri, rt = r[:-4], r[-4:]
        y = np.concatenate(pool.map(_pool_band, [ri[:, k:k + 1] for k in range(ri.shape[1])]), 1)
        zt = fit.Sinv @ (rt - fit.Kit.T @ y)
        return np.concatenate([y - fit.X @ zt, zt])
    return pool, solve_np


'''
assert anchor in s
s = s.replace(anchor, add + anchor)
old = '''    fit = AiryFit(mesh)
    t_load = time.time()'''
assert old in s
s = s.replace(old, '''    fit = AiryFit(mesh)
    pool, fit.solve_np = parallel_solver(fit, BATCH)                    # N1: before CUDA is initialised
    t_load = time.time()''')
old2 = '''    print(f"done: {step} steps'''
if old2 in s:
    s = s.replace(old2, '''    pool.terminate()
    print(f"done: {step} steps''')
open(p, 'w').write(s)
print("patched")
