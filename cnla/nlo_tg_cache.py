#!/usr/bin/env python3
"""Training cache for a fine-tune on the common thickness model t_g (review R1).
For every training plate and each of its 4 mirror images (the mirror image is a plate of its own, with its own scan
grid and alternating diagonals), on the canonical nodes of the 256 x 80 reference mesh:
  tg   (N,): nodal values of t_g (thickness of the dissipation in the loss);
  cap  (N,): t_g-safe nodal capacities: min over the adjacent triangles of the vertex values of the best linear minorant
             (tg_capacity.tg_minorant, as in the method), so that the vertex yield test with cap implies
             f(N) <= sigma_y t_g on every triangle (always positive).
Mirror order = nlo_refine.MIRRORS: (F,F), (T,F), (F,T), (T,T). Output: data/nlo_train_tg_cache.npz (float32).
Usage: python nlo_tg_cache.py [workers]"""
import os
import sys
from multiprocessing import Pool

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
MIRRORS = ((False, False), (True, False), (False, True), (True, True))


def one(i):
    from nlo_core import RefMesh, plate_on_mesh
    from thickness_tg import tg_eval, tg_quantities
    from tg_capacity import tg_minorant
    d = np.load(os.path.join(HERE, "data", "synthetic_plates.npz"), allow_pickle=True)
    xs, ys, t = (np.asarray(d[k][i], float) for k in ("x_mm", "y_mm", "thickness_mm"))
    L, W = xs[-1], ys[-1]
    mesh = RefMesh(256, 80)
    tg_out, cap_out = [], []
    for fx, fy in MIRRORS:
        xm = L - xs[::-1] if fx else xs
        ym = W - ys[::-1] if fy else ys
        tm = t[::-1 if fx else 1, ::-1 if fy else 1]
        nodes = plate_on_mesh(mesh, xm, ym, tm)["nodes"]
        tri = mesh.tri
        q = tg_quantities(nodes, tri, xm, ym, tm)
        tgn = tg_eval(nodes, xm, ym, tm)
        c = tg_minorant(nodes, tri, xm, ym, tm, quantities=q)            # (T, 3), the method's capacities
        cn = np.full(len(nodes), np.inf)
        np.minimum.at(cn, tri.ravel(), c.ravel())                       # nodal: min over the adjacent triangles
        tg_out.append(tgn.astype(np.float32))
        cap_out.append(cn.astype(np.float32))
    return np.stack(tg_out), np.stack(cap_out)


def main():
    workers = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    idx = np.load(os.path.join(HERE, "data", "nlo_split.npz"))["idx_train"]
    with Pool(workers) as p:
        out = p.map(one, list(idx), chunksize=4)
    tg = np.stack([o[0] for o in out])
    cap = np.stack([o[1] for o in out])
    np.savez(os.path.join(HERE, "data", os.environ.get("TG_CACHE_OUT", "nlo_train_tg_cache.npz")), tg=tg, cap=cap, idx=idx)
    rel = cap / np.maximum(tg, 1e-9)
    print(f"saved data/nlo_train_tg_cache.npz: {tg.shape}; cap/tg min {rel.min():.3f} mean {rel.mean():.4f}")


if __name__ == "__main__":
    main()
