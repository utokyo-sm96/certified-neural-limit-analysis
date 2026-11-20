"""Shared helpers of the second-review experiments (results/runs/review2/): plate sets, certification by verifier v2, the
network's identity-mirror fields on the canonical mesh and their exact t_g scaling."""
import os
import sys
import time

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
PF = os.path.join(ROOT, "cnla")
sys.path.insert(0, PF)
sys.path.insert(0, os.path.join(ROOT, "paper", "figures"))
OUT = os.path.dirname(os.path.abspath(__file__))


def plate_sets(sets=("val", "test", "real")):
    from val_histories import load_set
    n = int(os.environ.get("R2_N", 10 ** 6))                              # smoke tests: first n plates of each set
    return [(s, p) for s in sets for p in load_set(s)[:n]]


def v2_ub(nodes, tri, v, xs, ys, t, sy, thick):
    """Certified UB (verifier v2 recomputes the dissipation exactly on t_g) and the verification time."""
    from verify_v2 import verify_v2
    S = np.zeros((len(tri), 3, 3))
    S[:, :, 0] = 1.0
    c = time.perf_counter()
    ub, _, e, _ = verify_v2(dict(nodes=nodes, tri=tri, thick=thick, v=v, ub=0.0, stress=S, lb=0.0,
                                 status_ub="Solved", status_lb="Solved"), xs, ys, t, sy)
    return ub, [x for x in e if not x.startswith("UB reported")], time.perf_counter() - c


def v2_lb(nodes, tri, stress, xs, ys, t, sy, thick):
    """Certified LB of a per-vertex stress field (grip resultant / max(1, exact ratio)) and the verification time."""
    from nlo_refine_eval import SQ3, tri_area
    from verify_v2 import verify_v2
    vl = np.stack([nodes[:, 0] / xs[-1], np.zeros(len(nodes))], 1)
    ubd = float(np.sum(sy * (2 / SQ3) * (1 / xs[-1]) * tri_area(nodes, tri) * thick[tri].mean(1)))
    c = time.perf_counter()
    _, lb, e, _ = verify_v2(dict(nodes=nodes, tri=tri, thick=thick, v=vl, ub=ubd, stress=stress, lb=0.0,
                                 status_ub="Solved", status_lb="Solved"), xs, ys, t, sy)
    return lb, [x for x in e if not x.startswith("UB reported")], time.perf_counter() - c


def network_fields(model, p, fvs=True):
    """Identity-mirror network fields on the canonical mesh: (nodes, tri, v, Nv per vertex, N/mm)."""
    import nlo_refine as R
    import nlo_train
    import nlo_train_fvs
    pred = nlo_train_fvs.predict if fvs else nlo_train.predict
    r0 = R.mirror_back(pred(model, [dict(xs=p["xs"], ys=p["ys"], t=p["t"], sigma_y=p["sigma_y"])])[0], False, False,
                       p["xs"][-1], p["ys"][-1])
    vv, Nv = R.to_canonical(r0, p["xs"][-1], p["ys"][-1])
    mesh = R.canon()["mesh"]
    return mesh.nodes(p["xs"][-1], p["ys"][-1]), mesh.tri, vv, Nv


def scaled_exact(Nv, nodes, tri, p):
    """Field scaled to its exact t_g yield (piece vertices), as the method does with its final field."""
    from tg_capacity import tg_ratio
    from thickness_tg import tg_quantities
    r = tg_ratio(Nv, tg_quantities(nodes, tri, p["xs"], p["ys"], np.asarray(p["t"], float)), p["sigma_y"])
    return Nv / r
