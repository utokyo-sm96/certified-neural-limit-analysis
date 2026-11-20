import sys, time, numpy as np
sys.path.insert(0, '/home/oldlinux/Research_Projects/Corrosion/PaperX/loops/scratch')
sys.path.insert(0, '/home/oldlinux/Research_Projects/Corrosion/PaperX/cnla')
import nlo_train, nlo_train_fvs, nlo_refine as NEW, nlo_refine_main as OLD
from nlo_refine_eval import certify_lb, certify_ub
D = '/home/oldlinux/Research_Projects/Corrosion/PaperX/cnla/data/'
d = np.load(D + 'nlo_val.npz', allow_pickle=True); names = [str(n) for n in d['name']]
# deterministic comparison: no anytime deadline effects -> fixed 2 FVS rounds by a large budget and round cap via monkeypatch
for M in (NEW, OLD):
    M.LB_SECONDS = 1e9
    f = M.refine_lb_fvs; cnt = {'n': 0}
    def lim(*a, _f=f, _c=cnt, **k):
        _c['n'] += 1
        return _f(*a, **k) if _c['n'] <= 2 else None
    M.refine_lb_fvs = lim; M._cnt = cnt
zp = nlo_train.load(D + 'nlo_ckpt_100k.pt'); fv = nlo_train_fvs.load(D + 'nlo_fvs_ft30k.pt')
def run(M, model, p):
    M._cnt['n'] = 0; out = M.bounds(model, p)
    ub, e1 = certify_ub(out['ub_res'], p['xs'], p['ys'], p['t'], p['sigma_y']); lb, e2 = certify_lb(out['lb_res'], p['xs'], p['ys'], p['t'], p['sigma_y'])
    return ub, lb, e1 + e2
for nm in sys.argv[1:]:
    i = names.index(nm); p = dict(xs=d['x_mm'][i], ys=d['y_mm'][i], t=d['thickness_mm'][i], sigma_y=float(d['sigma_y_MPa'][i]))
    u0, l0, e0 = run(OLD, zp, p); u1, l1, e1 = run(NEW, zp, p); u2, l2, e2 = run(NEW, fv, p)
    print(f"{nm}: ZP net old code UB {u0/1e3:.3f} LB {l0/1e3:.3f} | new code UB {u1/1e3:.3f} LB {l1/1e3:.3f} (LB rel {l1/l0-1:+.1e}) {e0+e1} | "
          f"FVS net new code UB {u2/1e3:.3f} LB {l2/1e3:.3f} gap {100*(u2-l2)/u2:.2f}% (ZP net {100*(u1-l1)/u1:.2f}%) {e2}", flush=True)
