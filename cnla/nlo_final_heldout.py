#!/usr/bin/env python3
"""ONE-SHOT held-out evaluation of the final method (plan D), run once per final method.

Network: FVS network data/nlo_fvs_ft30k.pt. Three outputs per plate, each certified independently by the frozen step-1
verifier (nlo_refine_eval.certify_ub / certify_lb) and timed one plate at a time in this process:
  * "network": the network alone (nlo_train_fvs.predict);
  * "fast":    the certified hybrid of loop fvs-net-hybrid-1108 (global stages off, ~5.6 s on validation);
  * "full":    the hybrid + tuned global ADMM stages of loop plan-c-global-1109 (default nlo_refine, ~14.4 s).
Held-out sets: the 64 synthetic test plates (FE references data/nlo_test_ref.csv) and the 28 real scans (step-1 FE
brackets from results/tuning/nlo-1106/heldout_nlo.csv, same solver and settings), plus the 18 measured yield loads.

Usage: python nlo_final_heldout.py --data ../corroded_steel_plates_data_en --out ../results/runs/heldout-final
"""
import argparse
import csv
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
CKPT = os.path.join(HERE, "data", "nlo_fvs_ft30k.pt")


def load_plates(data):
    from scans import load_surface, read_table, surface_to_grid
    plates = []
    d = np.load(os.path.join(HERE, "data", "nlo_test.npz"), allow_pickle=True)
    refs = {r["name"]: r for r in csv.DictReader(open(os.path.join(HERE, "data", "nlo_test_ref.csv")))}
    for i in range(len(d["name"])):
        rf = refs[str(d["name"][i])]
        plates.append(dict(set="synthetic-test", name=str(d["name"][i]), xs=d["x_mm"][i], ys=d["y_mm"][i],
                           t=d["thickness_mm"][i], sigma_y=float(d["sigma_y_MPa"][i]),
                           FE_LB=float(rf["LB_ref_N"]), FE_UB=float(rf["UB_ref_N"]), meas=""))
    prev = {r["name"]: r for r in csv.DictReader(open(os.path.join(HERE, "..", "docs", "loops", "nlo-1106", "heldout_nlo.csv")))
            if r["set"] == "real-scan"}
    tests = {r["specimen_id"]: r for r in read_table(os.path.join(data, "tensile_tests.csv"))}
    sig = {"ananai": 281.6, "amarube": 269.4}
    for fam in ("ananai", "amarube"):
        folder = os.path.join(data, "surfaces", fam)
        for fn in sorted(os.listdir(folder)):
            sid = fn[:-4]
            xs, ys, g = surface_to_grid(load_surface(os.path.join(folder, fn)))
            plates.append(dict(set="real-scan", name=sid, xs=xs, ys=ys, t=np.clip(g["thickness_mm"], 0.05, None),
                               sigma_y=sig[fam], FE_LB=1e3 * float(prev[sid]["FE_LB_kN"]),
                               FE_UB=1e3 * float(prev[sid]["FE_UB_kN"]), meas=tests.get(sid, {}).get("yield_load_kN", "")))
    return plates


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--dry", action="store_true", help="2 validation plates instead of held-out data (script test)")
    a = ap.parse_args()
    import nlo_refine as R
    import nlo_train_fvs
    from nlo_refine_eval import certify_lb, certify_ub
    model = nlo_train_fvs.load(CKPT)
    full = {k: getattr(R, k) for k in ("LB_SECONDS", "UB_ROUNDS", "GLOBAL_SECONDS", "UB_GLOBAL_SECONDS", "UB_GLOBAL_ITERS")}
    fast = dict(full, LB_SECONDS=2.3, UB_ROUNDS=1, GLOBAL_SECONDS=0.0, UB_GLOBAL_SECONDS=0.0, UB_GLOBAL_ITERS=None)

    def set_mode(cfg):
        for k, v in cfg.items():
            setattr(R, k, v)

    if a.dry:
        d = np.load(os.path.join(HERE, "data", "nlo_val.npz"), allow_pickle=True)
        refs = {r["name"]: r for r in csv.DictReader(open(os.path.join(HERE, "data", "nlo_val_ref.csv")))}
        plates = [dict(set="synthetic-test", name=str(d["name"][i]), xs=d["x_mm"][i], ys=d["y_mm"][i], t=d["thickness_mm"][i],
                       sigma_y=float(d["sigma_y_MPa"][i]), FE_LB=float(refs[str(d["name"][i])]["LB_ref_N"]),
                       FE_UB=float(refs[str(d["name"][i])]["UB_ref_N"]), meas=str(100.0 + i)) for i in (0, 1)]
        plates[1]["set"] = "real-scan"
    else:
        plates = load_plates(a.data)
    set_mode(full)
    R.bounds(model, dict(xs=plates[0]["xs"], ys=plates[0]["ys"], t=plates[0]["t"], sigma_y=plates[0]["sigma_y"]))  # warm-up
    os.makedirs(a.out, exist_ok=True)
    rows = []
    for p in plates:
        pp = dict(xs=p["xs"], ys=p["ys"], t=p["t"], sigma_y=p["sigma_y"])
        row = dict(set=p["set"], name=p["name"], FE_LB_kN=p["FE_LB"] / 1e3, FE_UB_kN=p["FE_UB"] / 1e3, measured_yield_kN=p["meas"])
        t1 = time.time()
        r = nlo_train_fvs.predict(model, [pp])[0]
        dt = time.time() - t1
        res = {"network": (dict(ub_res=r, lb_res=r), dt)}
        for mode, cfg in (("fast", fast), ("full", full)):
            set_mode(cfg)
            t1 = time.time()
            out = R.bounds(model, pp)
            res[mode] = (out, time.time() - t1)
        for mode, (out, dt) in res.items():
            ub, eu = certify_ub(out["ub_res"], p["xs"], p["ys"], p["t"], p["sigma_y"])
            lb, el = certify_lb(out["lb_res"], p["xs"], p["ys"], p["t"], p["sigma_y"])
            row.update({f"{mode}_LB_kN": lb / 1e3, f"{mode}_UB_kN": ub / 1e3, f"{mode}_gap": (ub - lb) / ub, f"{mode}_s": dt,
                        f"{mode}_certified": "yes" if not (eu or el) else ";".join(eu + el),
                        f"{mode}_consistent": lb <= p["FE_UB"] * (1 + 1e-6) and ub >= p["FE_LB"] * (1 - 1e-6)})
        rows.append(row)
        print(f"{p['set']:14s} {p['name']}: net {100 * row['network_gap']:5.1f}%  fast {100 * row['fast_gap']:5.2f}% "
              f"({row['fast_s']:.1f}s)  full {100 * row['full_gap']:5.2f}% ({row['full_s']:.1f}s)  FE "
              f"{100 * (p['FE_UB'] - p['FE_LB']) / p['FE_UB']:.2f}%  "
              f"{'certified' if all(row[f'{m}_certified'] == 'yes' for m in res) else 'NOT CERTIFIED'}", flush=True)
    path = os.path.join(a.out, "heldout_final.csv")
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        for r in rows:
            w.writerow({k: (f"{v:.6g}" if isinstance(v, float) else v) for k, v in r.items()})
    print("---")
    for s in ("synthetic-test", "real-scan"):
        S = [r for r in rows if r["set"] == s]
        fe = np.array([(r["FE_UB_kN"] - r["FE_LB_kN"]) / r["FE_UB_kN"] for r in S])
        print(f"{s}: n={len(S)}  adaptive FE gap mean {100 * fe.mean():.2f}%")
        for m in ("network", "fast", "full"):
            g = np.array([r[f"{m}_gap"] for r in S])
            print(f"  {m:8s} certified {sum(r[f'{m}_certified'] == 'yes' for r in S)}/{len(S)}  consistent "
                  f"{sum(bool(r[f'{m}_consistent']) for r in S)}/{len(S)}  gap mean {100 * g.mean():.2f}% (median "
                  f"{100 * np.median(g):.2f}%, max {100 * g.max():.1f}%)  UB +{100 * np.mean([r[f'{m}_UB_kN'] / r['FE_UB_kN'] - 1 for r in S]):.2f}%  "
                  f"LB -{100 * np.mean([1 - r[f'{m}_LB_kN'] / r['FE_LB_kN'] for r in S]):.2f}%  time {np.mean([r[f'{m}_s'] for r in S]):.2f} s "
                  f"(max {np.max([r[f'{m}_s'] for r in S]):.1f})")
    T = [r for r in rows if r["measured_yield_kN"] not in ("", None)]
    meas = np.array([float(r["measured_yield_kN"]) for r in T])
    for m in ("FE", "network", "fast", "full"):
        lo = np.array([r[f"{m}_LB_kN"] for r in T])
        hi = np.array([r[f"{m}_UB_kN"] for r in T])
        q = meas / (0.5 * (lo + hi))
        print(f"tensile tests (n={len(T)}) {m:8s}: measured / mid-bracket = {q.mean():.3f} +- {q.std(ddof=1):.3f}, "
              f"MAPE {100 * np.mean(np.abs(q - 1)):.1f}%, measured inside bracket {int(np.sum((meas >= lo) & (meas <= hi)))}/{len(T)}")
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
