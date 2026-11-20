#!/usr/bin/env python3
"""One-shot held-out evaluation of the final neural limit operator (never used inside a loop).

  0. validation plates (reported only to compare the final model with the loop's 8-minute runs)
  1. 64 synthetic test plates: certified NLO bracket vs the step-1 FE bracket (data/nlo_test_ref.csv)
  2. real scans (18 Ananai + 10 Amarube): certified NLO bracket vs a step-1 FE bracket computed here;
     for the 18 Ananai tensile specimens also vs the measured yield load (Chun et al. 2013)
Every NLO result is certified by the frozen step-1 verifier; the FE results are certified too.

Usage: python nlo_heldout.py --data ../corroded_steel_plates_data_en --out ../loops/nlo-1106 \
                             --ckpt data/nlo_ckpt_final.pt
"""
import argparse
import csv
import os
import sys
import time

os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
SIGMA_Y = {"ananai": 281.6, "amarube": 269.4}


def certify(res, xs, ys, t, sy):
    from eval_limit_analysis import verify
    res = dict(res)
    res["status_ub"] = res["status_lb"] = "Solved"
    return verify(res, xs, ys, t, sy, "rigid")


def fe_job(p):
    from eval_limit_analysis import verify
    from limit_analysis import solve_bounds
    t1 = time.time()
    r = solve_bounds(p["xs"], p["ys"], p["t"], p["sigma_y"], grips="rigid")
    ub, lb, errs, _ = verify(r, p["xs"], p["ys"], p["t"], p["sigma_y"], "rigid")
    return ub, lb, errs, time.time() - t1


def run_nlo(nlo, model, plates):
    t1 = time.time()
    res = nlo.predict(model, plates)
    ms = 1000 * (time.time() - t1) / len(plates)
    return [certify(r, p["xs"], p["ys"], p["t"], p["sigma_y"])[:3] for p, r in zip(plates, res)], ms


def synthetic(nlo, model, split, rows):
    d = np.load(os.path.join(HERE, "data", f"nlo_{split}.npz"), allow_pickle=True)
    refs = {r["name"]: r for r in csv.DictReader(open(os.path.join(HERE, "data", f"nlo_{split}_ref.csv")))}
    plates = [dict(xs=d["x_mm"][i], ys=d["y_mm"][i], t=d["thickness_mm"][i], sigma_y=float(d["sigma_y_MPa"][i]))
              for i in range(len(d["name"]))]
    res, ms = run_nlo(nlo, model, plates)
    for name, fam, (ub, lb, errs) in zip(d["name"], d["family"], res):
        rf = refs[str(name)]
        rows.append(dict(set=f"synthetic-{split}", name=str(name), family=str(fam), NLO_LB_kN=lb / 1e3,
                         NLO_UB_kN=ub / 1e3, NLO_gap=(ub - lb) / ub, NLO_certified="yes" if not errs else ";".join(errs),
                         FE_LB_kN=float(rf["LB_ref_N"]) / 1e3, FE_UB_kN=float(rf["UB_ref_N"]) / 1e3,
                         FE_gap=float(rf["gap_ref"]), FE_s=float(rf["ref_s"]), measured_yield_kN=""))
    return ms


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    from scans import load_surface, read_table, surface_to_grid
    import nlo_train as nlo
    model = nlo.load(a.ckpt or nlo.CKPT)
    rows = []
    ms = {s: synthetic(nlo, model, s, rows) for s in ("val", "test")}

    tests = {r["specimen_id"]: r for r in read_table(os.path.join(a.data, "tensile_tests.csv"))}
    real = []
    for fam in ("ananai", "amarube"):
        folder = os.path.join(a.data, "surfaces", fam)
        for fn in sorted(os.listdir(folder)):
            xs, ys, g = surface_to_grid(load_surface(os.path.join(folder, fn)))
            real.append(dict(name=fn[:-4], family=fam, xs=xs, ys=ys, t=np.clip(g["thickness_mm"], 0.05, None),
                             sigma_y=SIGMA_Y[fam]))
    res, ms["real"] = run_nlo(nlo, model, real)
    from multiprocessing import Pool
    with Pool(a.workers) as pool:
        fe = pool.map(fe_job, real)
    for p, (ub, lb, errs), (ub_fe, lb_fe, errs_fe, fe_s) in zip(real, res, fe):
        meas = tests.get(p["name"], {}).get("yield_load_kN", "")
        rows.append(dict(set="real-scan", name=p["name"], family=p["family"], NLO_LB_kN=lb / 1e3, NLO_UB_kN=ub / 1e3,
                         NLO_gap=(ub - lb) / ub, NLO_certified="yes" if not errs else ";".join(errs),
                         FE_LB_kN=lb_fe / 1e3, FE_UB_kN=ub_fe / 1e3, FE_gap=(ub_fe - lb_fe) / ub_fe, FE_s=fe_s,
                         measured_yield_kN=meas))
        print(f"{p['name']}: NLO [{lb / 1e3:.1f}, {ub / 1e3:.1f}] ({100 * (ub - lb) / ub:.1f}%)  "
              f"FE [{lb_fe / 1e3:.1f}, {ub_fe / 1e3:.1f}]  measured {meas or '-'}  {errs or ''} {errs_fe or ''}", flush=True)

    out = os.path.join(a.out, "heldout_nlo.csv")
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        for r in rows:
            w.writerow({k: (f"{v:.6g}" if isinstance(v, float) else v) for k, v in r.items()})

    print("---")
    for s in ("synthetic-val", "synthetic-test", "real-scan"):
        R = [r for r in rows if r["set"] == s]
        g = np.array([r["NLO_gap"] for r in R])
        ub_ex = np.array([r["NLO_UB_kN"] / r["FE_UB_kN"] - 1 for r in R])
        lb_def = np.array([1 - r["NLO_LB_kN"] / r["FE_LB_kN"] for r in R])
        dual = all(r["NLO_LB_kN"] <= r["FE_UB_kN"] * (1 + 1e-6) and r["NLO_UB_kN"] >= r["FE_LB_kN"] * (1 - 1e-6) for r in R)
        cert = all(r["NLO_certified"] == "yes" for r in R)
        print(f"{s}: n={len(R)} certified={cert} consistent-with-FE={dual}  NLO gap mean {100 * g.mean():.1f}% "
              f"(median {100 * np.median(g):.1f}%, max {100 * g.max():.1f}%)  UB +{100 * ub_ex.mean():.1f}%  "
              f"LB -{100 * lb_def.mean():.1f}%  | FE gap {100 * np.mean([r['FE_gap'] for r in R]):.2f}%, "
              f"{np.mean([r['FE_s'] for r in R]):.0f} s/plate")
    print("NLO inference ms/plate: " + ", ".join(f"{k} {v:.0f}" for k, v in ms.items()))
    T = [r for r in rows if r["measured_yield_kN"] not in ("", None)]
    m = np.array([float(r["measured_yield_kN"]) for r in T])
    for key, lo, hi in (("NLO", "NLO_LB_kN", "NLO_UB_kN"), ("FE", "FE_LB_kN", "FE_UB_kN")):
        lo_a, hi_a = np.array([r[lo] for r in T]), np.array([r[hi] for r in T])
        q = m / (0.5 * (lo_a + hi_a))
        inside = int(np.sum((m >= lo_a) & (m <= hi_a)))
        print(f"tensile tests (n={len(T)}): measured / {key} mid-bracket = {q.mean():.3f} +- {q.std(ddof=1):.3f}, "
              f"MAPE {100 * np.mean(np.abs(q - 1)):.1f}%, measured inside {key} bracket {inside}/{len(T)}")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ink2, grid_c, blue, orange = "#52514e", "#e4e3df", "#2a78d6", "#eb6834"
    fig, axs = plt.subplots(1, 2, figsize=(11, 4.6), constrained_layout=True)
    R = sorted([r for r in rows if r["set"] == "synthetic-test"], key=lambda r: r["FE_UB_kN"])
    x = np.arange(len(R))
    ax = axs[0]
    ax.vlines(x, [r["NLO_LB_kN"] for r in R], [r["NLO_UB_kN"] for r in R], color=blue, lw=2.2,
              label="neural operator, certified bracket")
    ax.vlines(x, [r["FE_LB_kN"] for r in R], [r["FE_UB_kN"] for r in R], color=orange, lw=3.2,
              label="FE limit analysis, certified bracket")
    ax.set_xlabel("synthetic test plate (sorted by FE collapse load)")
    ax.set_ylabel("collapse load, kN")
    ax.set_title("64 held-out synthetic plates", loc="left", fontsize=10)
    ax.legend(frameon=False, loc="upper left", fontsize=8.5)
    ax = axs[1]
    lo, hi = 60, 240
    ax.plot([lo, hi], [lo, hi], color=ink2, lw=0.9, ls="--", zorder=1)
    ax.text(hi - 3, hi - 10, "prediction = measurement", color=ink2, ha="right", fontsize=8)
    for key, lo_k, hi_k, c, mk, off in (("neural operator", "NLO_LB_kN", "NLO_UB_kN", blue, "o", -0.6),
                                        ("FE limit analysis", "FE_LB_kN", "FE_UB_kN", orange, "s", 0.6)):
        l = np.array([r[lo_k] for r in T])
        h = np.array([r[hi_k] for r in T])
        ax.errorbar(m + off, 0.5 * (l + h), yerr=[0.5 * (h - l), 0.5 * (h - l)], fmt=mk, ms=5.5, color=c, mec="white",
                    mew=1.0, ecolor=c, elinewidth=1.4, capsize=0, label=f"{key}, certified bracket")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel("measured yield load, kN (18 Ananai tensile tests)")
    ax.set_ylabel("predicted collapse load, kN")
    ax.set_title("Real scans vs tensile tests", loc="left", fontsize=10)
    ax.legend(frameon=False, loc="upper left", fontsize=8.5)
    for ax in axs:
        ax.grid(color=grid_c, lw=0.6)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    fig.savefig(os.path.join(a.out, "heldout_nlo.png"), dpi=150)
    print(f"wrote {out} and heldout_nlo.png")


if __name__ == "__main__":
    main()
