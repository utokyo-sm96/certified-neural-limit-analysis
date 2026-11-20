#!/usr/bin/env python3
"""Held-out figures from results/runs/heldout-final/heldout_final.csv (one-shot run of nlo_final_heldout.py).

F6 (heldout_brackets.png): per plate, certified brackets normalised by the FE mid-bracket: fast hybrid (blue) and full
   method (aqua) over the adaptive-FE bracket (gray), synthetic test plates and real scans, sorted by the full gap.
F7 (heldout_tensile.png): the 18 tensile tests: measured yield load against the full method's certified bracket and the
   FE bracket, as ratios to the measured load.
"""
import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
RUN = os.path.join(HERE, "..", "docs", "runs", "heldout-final", "heldout_final.csv")
OUT = os.path.join(HERE, "..", "docs", "figures")
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#8a8984", "#e4e3df"


def style(ax):
    ax.grid(True, axis="y", color=GRID, linewidth=0.7)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def main():
    rows = list(csv.DictReader(open(RUN)))
    f = lambda r, k: float(r[k])
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                         "xtick.color": INK2, "ytick.color": INK2, "axes.linewidth": 0.8})
    # ---- F6
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.6), gridspec_kw=dict(width_ratios=[64, 28]), sharey=True)
    for ax, s, title in zip(axes, ("synthetic-test", "real-scan"), ("64 synthetic test plates", "28 real scans")):
        S = [r for r in rows if r["set"] == s]
        S.sort(key=lambda r: f(r, "full_gap"))
        x = np.arange(len(S))
        mid = np.array([0.5 * (f(r, "FE_LB_kN") + f(r, "FE_UB_kN")) for r in S])
        for lo_k, hi_k, col, w, off, z, lab in (("fast_LB_kN", "fast_UB_kN", BLUE, 0.9, -0.3, 2, "fast hybrid (~6 s)"),
                                                ("full_LB_kN", "full_UB_kN", AQUA, 1.8, 0.3, 3, "full method (~14 s)"),
                                                ("FE_LB_kN", "FE_UB_kN", INK2, 2.6, 0.0, 4, "adaptive FE (~54 s)")):
            lo = np.array([f(r, lo_k) for r in S]) / mid
            hi = np.array([f(r, hi_k) for r in S]) / mid
            ax.vlines(x + off, 100 * (lo - 1), 100 * (hi - 1), color=col, linewidth=w, label=lab, zorder=z)
        ax.axhline(0, color=MUTED, linewidth=0.6, zorder=1)
        style(ax)
        ax.set_xticks([])
        ax.set_title(title, fontsize=9, color=INK, loc="left")
    axes[0].set_ylabel("certified bracket relative to the FE mid-bracket [%]")
    axes[0].legend(loc="lower left", fontsize=7.5, frameon=False)
    fig.supxlabel("plates, each panel sorted by the full method's bracket", fontsize=9, color=INK)
    fig.tight_layout()
    os.makedirs(OUT, exist_ok=True)
    fig.savefig(os.path.join(OUT, "heldout_brackets.png"), dpi=220)
    fig.savefig(os.path.join(OUT, "heldout_brackets.pdf"))
    # ---- F7
    T = [r for r in rows if r["measured_yield_kN"] not in ("", None)]
    T.sort(key=lambda r: r["name"])
    fig, ax = plt.subplots(figsize=(6.6, 3.4))
    x = np.arange(len(T))
    meas = np.array([float(r["measured_yield_kN"]) for r in T])
    for lo_k, hi_k, col, w, off, lab in (("FE_LB_kN", "FE_UB_kN", MUTED, 4.0, -0.18, "adaptive FE bracket"),
                                         ("full_LB_kN", "full_UB_kN", AQUA, 2.2, 0.18, "certified bracket, full method")):
        lo = np.array([f(r, lo_k) for r in T]) / meas
        hi = np.array([f(r, hi_k) for r in T]) / meas
        ax.vlines(x + off, lo, hi, color=col, linewidth=w, label=lab, zorder=3)
    ax.axhline(1.0, color=ORANGE, linewidth=1.2, label="measured yield load", zorder=2)
    style(ax)
    ax.set_xticks(x)
    ax.set_xticklabels([r["name"] for r in T], rotation=60, fontsize=7)
    ax.set_ylabel("predicted / measured yield load")
    ax.legend(loc="lower left", bbox_to_anchor=(0.0, 1.0), fontsize=7.5, frameon=False, ncol=3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "heldout_tensile.png"), dpi=220)
    fig.savefig(os.path.join(OUT, "heldout_tensile.pdf"))
    print("saved heldout_brackets.png, heldout_tensile.png")


if __name__ == "__main__":
    main()
