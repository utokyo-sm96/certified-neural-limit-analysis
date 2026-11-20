#!/usr/bin/env python3
"""Speed/accuracy figure: mean certified bracket vs mean wall time per plate, 32 validation plates.

Inputs (results/runs/pareto/): pareto_val.csv (nlo_pareto.py), fe_timing_val.csv (nlo_fe_timing.py),
fvs_floor_val.csv (nlo_fvs_floor.py); reference CSV data/nlo_val_ref.csv. Output: results/figures/pareto_val.{png,pdf}.
Usage: python plot_pareto.py
"""
import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
RUNS = os.path.join(HERE, "..", "docs", "runs", "pareto")
OUT = os.path.join(HERE, "..", "docs", "figures")

# reference palette (light mode): categorical slots 1-3, ink, grid
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#8a8984", "#e4e3df"

# points measured elsewhere (32 validation plates, certified): network alone (nlo_train_fvs --eval, batch inference)
NETWORK = (0.056, 8.56)
ZP_HYBRID = (5.64, 3.662)          # loop fvs-hybrid-1107 final, 100k ZP network
TUNED = (14.43, 1.772)             # loop plan-c-global-1109 final (hybrid + tuned global LB/UB stages, iteration budgets)


def rows(name):
    return list(csv.DictReader(open(os.path.join(RUNS, name))))


def main():
    par = rows("pareto_val.csv")
    levels = list(dict.fromkeys(r["level"] for r in par))
    curve = [(np.mean([float(r["seconds"]) for r in par if r["level"] == lv]),
              100 * np.mean([float(r["gap"]) for r in par if r["level"] == lv])) for lv in levels]
    fe = rows("fe_timing_val.csv")
    fe_t = np.array([float(r["seconds"]) for r in fe])
    ref = list(csv.DictReader(open(os.path.join(HERE, "data", "nlo_val_ref.csv"))))
    fe_gap = 100 * np.mean([float(r["gap_ref"]) for r in ref])            # all 32 plates (bounds do not depend on timing)
    fl = rows("fvs_floor_val.csv")
    floor_fvs = 100 * np.mean([float(r["gap_fvs"]) for r in fl])
    floor_zp = 100 * np.mean([float(r["gap_zp"]) for r in fl])
    ipm_t = np.mean([float(r["solve_s"]) for r in fl])

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                         "xtick.color": INK2, "ytick.color": INK2, "axes.linewidth": 0.8})
    fig, ax = plt.subplots(figsize=(6.6, 4.1))
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(0.03, 600)
    ax.set_ylim(0.35, 12)
    ax.grid(True, which="major", color=GRID, linewidth=0.7)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)

    # mesh floors (reference levels)
    # mesh floors = best bracket certifiable on the network's 256 x 80 mesh (global optimum, interior point)
    for y, lab, x0, ha, va, f in ((floor_zp, f"mesh floor, constant stress (ZP): {floor_zp:.2f} %", 560, "right", "bottom", 1.05),
                                  (floor_fvs, f"mesh floor, linear stress (FVS): {floor_fvs:.2f} %", 0.034, "left", "bottom", 1.05)):
        ax.axhline(y, color=MUTED, linewidth=1.0, linestyle=(0, (4, 3)), zorder=1)
        ax.text(x0, y * f, lab, color=INK2, fontsize=7.5, va=va, ha=ha)

    # our method: network alone -> hybrid budgets
    xs = [NETWORK[0]] + [c[0] for c in curve]
    ys = [NETWORK[1]] + [c[1] for c in curve]
    ax.plot(xs, ys, color=BLUE, linewidth=1.6, marker="o", markersize=5.5, markeredgecolor="white",
            markeredgewidth=1.0, zorder=4, label="network + certified hybrid (FVS network), budgets")
    ax.annotate("network alone", (xs[0], ys[0]), xytext=(6, 4), textcoords="offset points", color=INK, fontsize=7.5)
    ax.annotate("mirror + convex\ncombination", (xs[1], ys[1]), xytext=(7, 2), textcoords="offset points", color=INK,
                fontsize=7.5)
    i_def = 1 + levels.index("b3")
    ax.annotate(f"default ({ys[i_def]:.2f} %, {xs[i_def]:.1f} s)", (xs[i_def], ys[i_def]), xytext=(-8, -12),
                textcoords="offset points", color=INK, fontsize=7.5, ha="right")
    ax.annotate(f"plateau {ys[-1]:.2f} %", (xs[-1], ys[-1]), xytext=(6, 5), textcoords="offset points", color=INK,
                fontsize=7.5)

    # previous hybrid on the ZP network (earlier version, muted)
    ax.plot(*ZP_HYBRID, linestyle="none", marker="D", markersize=5.0, markerfacecolor="white", markeredgecolor=MUTED,
            markeredgewidth=1.2, zorder=5, label="earlier version: hybrid on the constant-stress (ZP) network")

    # plan C: hybrid + global ADMM stage(s) with exact plate-independent solves + local repair
    pts = []
    for fn in ("pareto_global_val.csv", "pareto_global_ub_val.csv"):
        path = os.path.join(HERE, "..", "docs", "runs", "plan-c", fn)
        if os.path.exists(path):
            g = list(csv.DictReader(open(path)))
            for lv in dict.fromkeys(r["level"] for r in g):
                rr = [r for r in g if r["level"] == lv]
                pts.append((np.mean([float(r["seconds"]) for r in rr]), 100 * np.mean([float(r["gap"]) for r in rr]), lv))
    if pts:
        pts.sort()
        front, best = [], np.inf                                         # time/gap trade-off envelope
        for t_, g_, lv in pts:
            if g_ < best:
                front.append((t_, g_))
                best = g_
        ax.plot([p[0] for p in pts], [p[1] for p in pts], linestyle="none", marker="o", markersize=5.0, color=AQUA,
                markeredgecolor="white", markeredgewidth=1.0, zorder=5)
        ax.plot([p[0] for p in front], [p[1] for p in front], color=AQUA, linewidth=1.6, zorder=4,
                label="+ global ADMM stage, exact solves + local repair (plan C)")
        ax.plot(*TUNED, linestyle="none", marker="*", markersize=11, color=AQUA, markeredgecolor="white",
                markeredgewidth=0.8, zorder=6)
        ax.annotate(f"tuned: {TUNED[1]:.2f} %, {TUNED[0]:.0f} s", TUNED, xytext=(8, -4), textcoords="offset points",
                    color=INK, fontsize=7.5)

    # adaptive FE (step-1), single-process time range of the timed sample
    ax.errorbar([np.mean(fe_t)], [fe_gap], xerr=[[np.mean(fe_t) - fe_t.min()], [fe_t.max() - np.mean(fe_t)]],
                fmt="s", markersize=5.5, color=ORANGE, markeredgecolor="white", markeredgewidth=1.0, elinewidth=1.0,
                capsize=2.5, zorder=5, label="adaptive FE limit analysis (step 1, 0.25 mm)")
    # global interior point on the reference mesh reaches the FVS floor
    ax.plot([ipm_t], [floor_fvs], linestyle="none", marker="s", markersize=5.5, color=MUTED, markeredgecolor="white",
            markeredgewidth=1.0, zorder=5, label="global interior-point solve on the mesh (FVS LB only)")

    ax.set_xlabel("mean wall time per plate [s] (one CPU process + GPU inference)")
    ax.set_ylabel("mean certified bracket (UB − LB)/UB [%]")
    ax.set_yticks([0.5, 1, 2, 5, 10])
    ax.set_yticklabels(["0.5", "1", "2", "5", "10"])
    ax.set_xticks([0.1, 1, 10, 100])
    ax.set_xticklabels(["0.1", "1", "10", "100"])
    leg = ax.legend(loc="lower left", bbox_to_anchor=(0.0, 0.235), fontsize=7.5, frameon=False, labelcolor=INK)
    ax.set_title("Certified collapse-load brackets vs time, 32 validation plates", fontsize=9.5, color=INK, loc="left")
    fig.tight_layout()
    os.makedirs(OUT, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(OUT, f"pareto_val.{ext}"), dpi=220)
    print("saved", os.path.join(OUT, "pareto_val.png"))
    print(f"curve: {list(zip(['network'] + levels, np.round(xs, 2), np.round(ys, 3)))}")
    print(f"FE {fe_gap:.3f} % at {np.mean(fe_t):.1f} s (range {fe_t.min():.0f}-{fe_t.max():.0f}); floors ZP {floor_zp:.3f} % "
          f"FVS {floor_fvs:.3f} %; IPM FVS LB {ipm_t:.0f} s/plate")


if __name__ == "__main__":
    main()
