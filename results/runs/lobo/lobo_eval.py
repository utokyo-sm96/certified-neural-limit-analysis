#!/usr/bin/env python3
"""Leave-one-bridge-out evaluation (review R4). Networks: data/nlo_lobo_ananai30k.pt (trained on Ananai-family
synthetic plates only), data/nlo_lobo_amarube30k.pt (Amarube family only) and the control data/nlo_ckpt_final.pt (both
families, same 30k recipe). Each is evaluated, network alone and full method (THICK_MODEL "tg"), every bracket certified
by verifier v2, on (a) the validation plates split by generator family and (b) the real scans of each bridge. The real
scans are evaluated once per network for this experiment; nothing is tuned on them.
Output: loops/lobo/lobo_eval.csv. Usage: python loops/lobo/lobo_eval.py"""
import csv
import os
import sys
import time

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
PF = os.path.join(ROOT, "cnla")
sys.path.insert(0, PF)
NETS = {"ananai_only": "nlo_lobo_ananai30k.pt", "amarube_only": "nlo_lobo_amarube30k.pt", "both_control": "nlo_ckpt_final.pt"}


def plates():
    from scans import load_surface, surface_to_grid
    out = []
    d = np.load(os.path.join(PF, "data", "nlo_val.npz"), allow_pickle=True)
    sp = np.load(os.path.join(PF, "data", "synthetic_plates.npz"), allow_pickle=True)
    for i, n in enumerate(map(str, d["name"])):
        fam = str(sp["family"][int(n[3:])])
        out.append(dict(set="val", family=fam, name=n, xs=d["x_mm"][i], ys=d["y_mm"][i],
                        t=np.asarray(d["thickness_mm"][i], float), sigma_y=float(d["sigma_y_MPa"][i])))
    sig = {"ananai": 281.6, "amarube": 269.4}
    data = os.path.join(ROOT, "corroded_steel_plates_data_en")
    for fam in ("ananai", "amarube"):
        folder = os.path.join(data, "surfaces", fam)
        for fn in sorted(os.listdir(folder)):
            xs, ys, g = surface_to_grid(load_surface(os.path.join(folder, fn)))
            out.append(dict(set="real", family=fam, name=fn[:-4], xs=xs, ys=ys,
                            t=np.clip(g["thickness_mm"], 0.05, None), sigma_y=sig[fam]))
    return out


def certify(res_u, res_l, p):
    from nlo_refine_eval import SQ3, tri_area
    from verify_v2 import verify_v2
    xs, ys, t, sy = p["xs"], p["ys"], p["t"], p["sigma_y"]
    nodes, tri = np.asarray(res_u["nodes"], float), np.asarray(res_u["tri"], int)
    S = np.zeros((len(tri), 3, 3))
    S[:, :, 0] = 1.0
    ub, _, eu, _ = verify_v2(dict(nodes=nodes, tri=tri, thick=np.asarray(res_u["thick"], float), v=np.asarray(res_u["v"], float),
                                  ub=float(res_u["ub"]), stress=S, lb=0.0, status_ub="Solved", status_lb="Solved"), xs, ys, t, sy)
    nl, tl = np.asarray(res_l["nodes"], float), np.asarray(res_l["tri"], int)
    thl = np.asarray(res_l["thick"], float)
    vl = np.stack([nl[:, 0] / xs[-1], np.zeros(len(nl))], 1)
    ubd = float(np.sum(sy * (2 / SQ3) * (1 / xs[-1]) * tri_area(nl, tl) * thl[tl].mean(1)))
    _, lb, el, _ = verify_v2(dict(nodes=nl, tri=tl, thick=thl, v=vl, ub=ubd, stress=np.asarray(res_l["stress"], float),
                                  lb=float(res_l["lb"]), status_ub="Solved", status_lb="Solved"), xs, ys, t, sy)
    return lb, ub, ";".join(eu + el)


def main():
    import nlo_refine as R
    import nlo_train
    R.THICK_MODEL = "tg"
    P = plates()
    rows = []
    out = os.path.join(os.path.dirname(__file__), "lobo_eval.csv")
    for tag, fn in NETS.items():
        model = nlo_train.load(os.path.join(PF, "data", fn))
        R.bounds(model, dict(xs=P[0]["xs"], ys=P[0]["ys"], t=P[0]["t"], sigma_y=P[0]["sigma_y"]))      # warm-up
        for p in P:
            pp = dict(xs=p["xs"], ys=p["ys"], t=p["t"], sigma_y=p["sigma_y"])
            net = nlo_train.predict(model, [pp])[0]
            nlb, nub, nerr = certify(net, net, p)
            t0 = time.time()
            o = R.bounds(model, pp)
            dt = time.time() - t0
            flb, fub, ferr = certify(o["ub_res"], o["lb_res"], p)
            rows.append(dict(net=tag, set=p["set"], family=p["family"], plate=p["name"], net_lb=nlb, net_ub=nub,
                             net_gap=(nub - nlb) / nub, full_lb=flb, full_ub=fub, full_gap=(fub - flb) / fub, seconds=dt,
                             errors=";".join(e for e in (nerr, ferr) if e)))
            print(f"{tag:13s} {p['set']:4s} {p['family']:7s} {p['name']:8s} network {100 * rows[-1]['net_gap']:6.2f} %  full "
                  f"{100 * rows[-1]['full_gap']:5.2f} %  {dt:5.1f} s  {rows[-1]['errors']}", flush=True)
            with open(out, "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(rows[0]))
                w.writeheader()
                w.writerows(rows)


if __name__ == "__main__":
    main()
