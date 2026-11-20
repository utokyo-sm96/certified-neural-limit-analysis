"""Vertex capacities of the method (best linear minorant of t_g per triangle, tg_capacity.tg_minorant) relative to t_g at the
vertices, on the 32 validation plates (reference mesh 256 x 80). Output: capacities_val.csv (plate, min_ratio, mean_ratio,
additive_min_ratio). Usage: python results/probes/thickness-model/capacities_val.py"""
import csv
import os
import sys

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "cnla"))
from nlo_core import RefMesh, plate_on_mesh  # noqa: E402
from tg_capacity import tg_minorant  # noqa: E402
from thickness_tg import tg_eval, tg_quantities  # noqa: E402

d = np.load(os.path.join(ROOT, "cnla", "data", "nlo_val.npz"), allow_pickle=True)
mesh = RefMesh(256, 80)
tri = mesh.tri
rows = []
for i, name in enumerate(map(str, d["name"])):
    xs, ys, t = d["x_mm"][i], d["y_mm"][i], np.asarray(d["thickness_mm"][i], float)
    nodes = plate_on_mesh(mesh, xs, ys, t)["nodes"]
    q = tg_quantities(nodes, tri, xs, ys, t)
    _, own, lam, tpt = q
    tv = tg_eval(nodes, xs, ys, t)[tri]
    c = tg_minorant(nodes, tri, xs, ys, t, quantities=q)
    delta = np.full(len(tri), -np.inf)
    np.maximum.at(delta, own, (lam * tv[own]).sum(1) - tpt)
    r, ra = c / tv, (tv - delta[:, None]) / tv
    rows.append(dict(plate=name, min_ratio=r.min(), mean_ratio=r.mean(), additive_min_ratio=ra.min()))
    print(name, f"min {r.min():.4f} mean {r.mean():.5f} (additive min {ra.min():.4f})", flush=True)
with open(os.path.join(os.path.dirname(__file__), "capacities_val.csv"), "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0]))
    w.writeheader()
    w.writerows(rows)
