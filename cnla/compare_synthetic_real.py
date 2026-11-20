#!/usr/bin/env python3
"""Compare synthetic plates (corrosion_generator.py) with the 28 scanned plates.

Fitted by construction : thickness histogram (mean, SD, skewness, deepest pit), corroded-face
                         share, planar corrosion gradient.
Partly fitted          : correlation lengths (fitted after de-trending; re-measured here raw).
Held out (not fitted)  : minimum-average thickness (min over x-sections of the width-averaged t),
                         and two mechanical results from the plane-stress FEM of each plate:
                         axial stiffness (membrane-equivalent thickness) and the 99th-percentile
                         stress concentration. These depend on where pits sit and how they line up.

Usage
-----
python compare_synthetic_real.py --data ../corroded_steel_plates_data_en \
    --synthetic ./data/synthetic_plates.npz --out ./data [--fem_per_family 150]
"""
import argparse
import csv
import os

import numpy as np

from corrosion_generator import FAMILIES, calibrate, load_synthetic, plate_stats
from scans import E_REF, EDGE_ZONE, L_HAT, grid_mesh, solve_membrane, von_mises_hat

NU = {"ananai": 0.271, "amarube": 0.284}      # coupon means (material_properties.csv)
STATS = [  # key, label, how it is constrained
    ("mean_t", "mean t / t0", "fitted"),
    ("sd_t", "SD of t / t0", "fitted"),
    ("skew_t", "skewness of t", "fitted"),
    ("max_depth", "deepest pit / t0", "fitted"),
    ("gradient_share", "gradient share of variance", "fitted"),
    ("corr_len_x_mm", "corr. length along x, mm", "partly"),
    ("corr_len_y_mm", "corr. length across y, mm", "partly"),
    ("sd_corroded_face_mm", "SD corroded face, mm", "fitted"),
    ("sd_other_face_mm", "SD other face, mm", "fitted"),
    ("min_avg_t", "min-avg t / t0", "held out"),
    ("stiffness_ratio", "t_eff / mean t  (FEM)", "held out"),
    ("scf_p99", "stress conc. p99  (FEM)", "held out"),
]


def membrane_metrics(xs, ys, t, t0, nu):
    """Plane-stress FEM, as in scans.py: axial stiffness and stress concentration."""
    xy_mm, tri, left, right = grid_mesh(np.asarray(xs, float), np.asarray(ys, float))
    xy = xy_mm / (float(xs[-1]) / L_HAT)
    E = E_REF * np.clip(np.asarray(t, float).ravel() / t0, 0.02, None)
    nuv = np.full(len(xy), nu)
    uv, comp = solve_membrane(xy, tri, left, right, E, nuv)
    _, comp0 = solve_membrane(xy, tri, left, right, np.full(len(xy), E_REF), nuv)
    vm = von_mises_hat(xy, tri, uv, nu)
    xc = xy[tri].mean(1)[:, 0]
    core = (xc > EDGE_ZONE * L_HAT) & (xc < (1 - EDGE_ZONE) * L_HAT)
    return comp0 / comp, float(np.percentile(vm[core], 99))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True)
    ap.add_argument("--synthetic", required=True)
    ap.add_argument("--out", default="./data")
    ap.add_argument("--fem_per_family", type=int, default=150, help="synthetic plates per family run through FEM")
    args = ap.parse_args()

    parents = calibrate(args.data)
    real = []
    for p in parents:
        xs, ys, g = p["grid"]
        s = plate_stats(xs, ys, g["thickness_mm"], p["t0"], g["z_top_mm"], g["z_bottom_mm"])
        te, scf = membrane_metrics(xs, ys, g["thickness_mm"], p["t0"], NU[p["family"]])
        real.append(dict(id=p["specimen_id"], family=p["family"], t0=p["t0"], xs=xs, ys=ys,
                         t=g["thickness_mm"], stiffness_ratio=te / s["mean_t"], scf_p99=scf, **s))
    synth, nfem = [], {f: 0 for f in FAMILIES}
    for k, q in enumerate(load_synthetic(args.synthetic)):
        s = plate_stats(q["xs"], q["ys"], q["thickness"], q["t0"], q["z_top"], q["z_bottom"])
        row = dict(id=f"S{k:04d}", family=q["family"], t0=q["t0"], xs=q["xs"], ys=q["ys"], t=q["thickness"],
                   parent_a=q["parent_a"], parent_b=q["parent_b"], lam=q["lam"],
                   stiffness_ratio=np.nan, scf_p99=np.nan, **s)
        if nfem[q["family"]] < args.fem_per_family:
            te, scf = membrane_metrics(q["xs"], q["ys"], q["thickness"], q["t0"], NU[q["family"]])
            row.update(stiffness_ratio=te / s["mean_t"], scf_p99=scf)
            nfem[q["family"]] += 1
        synth.append(row)

    # ---- table: real range vs synthetic 2.5-97.5 % band, and how many real plates fall outside it
    rows = []
    print(f"\n{'family':8s} {'statistic':28s} {'constraint':10s} {'real median [min, max]':>26s} "
          f"{'synthetic median [2.5%, 97.5%]':>32s} {'real outside band':>18s}")
    for fam in FAMILIES:
        R = [r for r in real if r["family"] == fam]
        S = [s for s in synth if s["family"] == fam]
        for key, label, how in STATS:
            rv = np.array([r[key] for r in R])
            sv = np.array([s[key] for s in S])
            sv = sv[np.isfinite(sv)]
            lo, hi = np.percentile(sv, [2.5, 97.5])
            outside = [f"{r['id']}({'low' if r[key] < lo else 'high'})" for r in R if r[key] < lo or r[key] > hi]
            out = len(outside)
            rows.append(dict(family=fam, statistic=key, constraint=how,
                             real_median=np.median(rv), real_min=rv.min(), real_max=rv.max(),
                             synth_median=np.median(sv), synth_p2_5=lo, synth_p97_5=hi,
                             real_outside_band=out, n_real=len(rv), n_synth=len(sv),
                             outside_ids=" ".join(outside)))
            print(f"{fam:8s} {label:28s} {how:10s} {np.median(rv):8.3f} [{rv.min():7.3f},{rv.max():7.3f}] "
                  f"{np.median(sv):10.3f} [{lo:8.3f},{hi:8.3f}]    {out:3d} / {len(rv)}  {' '.join(outside)}")
    for fam in FAMILIES:
        fr = [r for r in rows if r["family"] == fam]
        tot, n = sum(r["real_outside_band"] for r in fr), sum(r["n_real"] for r in fr)
        print(f"{fam}: {tot} of {n} real values ({100 * tot / n:.1f} %) fall outside the synthetic 95 % band "
              f"(a perfect generator would give about 5 %)")
    with open(os.path.join(args.out, "synthetic_vs_real_stats.csv"), "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0]))
        wr.writeheader()
        wr.writerows(rows)

    make_figures(args.out, real, synth)


# ----------------------------------------------------------------------------- figures
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
BLUE, ORANGE = "#2a78d6", "#eb6834"


def style(plt):
    plt.rcParams.update({"font.size": 9, "axes.edgecolor": INK2, "axes.labelcolor": INK2,
                         "xtick.color": INK2, "ytick.color": INK2, "axes.titlecolor": INK,
                         "axes.spines.top": False, "axes.spines.right": False})


def pick_match(real_plate, synth):
    """Synthetic plate of the same family closest in (mean, SD) that does not have this plate as a parent."""
    best, bd = None, np.inf
    for s in synth:
        if s["family"] != real_plate["family"] or real_plate["id"] in (s["parent_a"], s["parent_b"]):
            continue
        d = ((s["mean_t"] - real_plate["mean_t"]) / 0.05) ** 2 + ((s["sd_t"] - real_plate["sd_t"]) / 0.02) ** 2
        if d < bd:
            best, bd = s, d
    return best


def make_figures(out, real, synth):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    style(plt)

    # ---------------- figure 1: maps, real vs matched synthetic
    picks = {"ananai": ["AN-08", "AN-13", "AN-17", "AN-20"], "amarube": ["AM-01", "AM-05", "AM-09", "AM-03"]}
    byid = {r["id"]: r for r in real}
    fig, axes = plt.subplots(4, 4, figsize=(17, 7.6), constrained_layout=True)
    for f_i, fam in enumerate(FAMILIES):
        for c, sid in enumerate(picks[fam]):
            r = byid[sid]
            s = pick_match(r, synth)
            lo, hi = np.percentile(r["t"] / r["t0"], [0.5, 99.5])
            for k, (plate, tag) in enumerate([(r, f"real {sid}"), (s, f"synthetic {s['id']}")]):
                ax = axes[2 * f_i + k, c]
                X, Y = np.meshgrid(plate["xs"], plate["ys"], indexing="ij")
                h = ax.pcolormesh(X, Y, plate["t"] / plate["t0"], cmap="Blues", vmin=lo, vmax=hi, shading="nearest")
                ax.set_aspect("equal")
                ax.set_xlim(0, 300)
                ax.set_ylim(0, 90)
                ax.set_yticks([0, 80])
                ax.set_xticks([0, 100, 200, 300] if 2 * f_i + k == 3 else [])
                ax.set_title(f"{tag}: mean {plate['mean_t']:.2f}, SD {plate['sd_t']:.3f}, "
                             f"min-avg {plate['min_avg_t']:.2f}", fontsize=7.5, loc="left",
                             color=INK if k == 0 else BLUE)
            fig.colorbar(h, ax=[axes[2 * f_i, c], axes[2 * f_i + 1, c]], shrink=0.8, pad=0.01, label="t / t0")
    axes[0, 0].set_ylabel("Ananai\nreal", color=INK)
    axes[1, 0].set_ylabel("Ananai\nsynthetic", color=BLUE)
    axes[2, 0].set_ylabel("Amarube\nreal", color=INK)
    axes[3, 0].set_ylabel("Amarube\nsynthetic", color=BLUE)
    axes[3, 0].set_xlabel("x (loading direction), mm")
    fig.suptitle("Real scans (rows 1, 3) vs synthetic plates with similar mean and SD (rows 2, 4), "
                 "never generated from that same scan. Each pair shares a colour scale.", fontsize=10, color=INK)
    p1 = os.path.join(out, "synthetic_vs_real_maps.png")
    fig.savefig(p1, dpi=140)
    plt.close(fig)

    # ---------------- figure 2: statistics, synthetic cloud vs real plates
    panels = [("mean_t", "sd_t", "Mean vs SD", "fitted", False),
              ("gradient_share", "corr_len_x_mm", "Gradient vs corr. length along x", "fitted / partly", False),
              ("corr_len_x_mm", "corr_len_y_mm", "Correlation lengths, mm", "partly fitted", True),
              ("sd_corroded_face_mm", "sd_other_face_mm", "Face roughness, mm", "fitted", False),
              ("mean_t", "min_avg_t", "Mean vs min-avg thickness", "held out", False),
              ("mean_t", "scf_p99", "Stress concentration (FEM)", "held out", False),
              ("mean_t", "stiffness_ratio", "Axial stiffness (FEM)", "held out", False)]
    labels = {k: l for k, l, _ in STATS}
    fig, axes = plt.subplots(2, len(panels), figsize=(3.05 * len(panels), 6.6), constrained_layout=True)
    for f_i, fam in enumerate(FAMILIES):
        R = [r for r in real if r["family"] == fam]
        S = [s for s in synth if s["family"] == fam]
        for c, (kx, ky, title, how, logs) in enumerate(panels):
            ax = axes[f_i, c]
            sx, sy = np.array([s[kx] for s in S]), np.array([s[ky] for s in S])
            ok = np.isfinite(sx) & np.isfinite(sy)
            rx, ry = np.array([r[kx] for r in R]), np.array([r[ky] for r in R])
            ax.scatter(sx[ok], sy[ok], s=8, color=BLUE, alpha=0.35, linewidth=0,
                       label=f"synthetic (n={ok.sum()})", zorder=2)
            ax.scatter(rx, ry, s=48, marker="^", color=ORANGE, edgecolor="white", linewidth=1.3,
                       label=f"real scans (n={len(R)})", zorder=3)
            if ky == "min_avg_t":
                v = np.concatenate([sx, rx])
                ax.plot([v.min(), 1.0], [v.min(), 1.0], color=INK2, lw=0.8, ls="--", zorder=1)
            if logs:
                from matplotlib.ticker import FixedLocator, NullFormatter, NullLocator, ScalarFormatter
                ax.set_xscale("log")
                ax.set_yscale("log")
                for axis in (ax.xaxis, ax.yaxis):
                    axis.set_major_locator(FixedLocator([2, 5, 10, 20, 50, 100]))
                    axis.set_major_formatter(ScalarFormatter())
                    axis.set_minor_locator(NullLocator())
                    axis.set_minor_formatter(NullFormatter())
            ax.set_xlabel(labels[kx])
            ax.set_ylabel(labels[ky])
            ax.grid(color=GRID, lw=0.6)
            fam_name = "Ananai" if fam == "ananai" else "Amarube"
            ax.set_title(f"{fam_name}: {title}\n({how})", loc="left", fontsize=9,
                         color=ORANGE if how == "held out" else INK)
            if c == 0:
                ax.legend(frameon=False, loc="upper right", fontsize=8)
    p2 = os.path.join(out, "synthetic_vs_real_stats.png")
    fig.savefig(p2, dpi=140)
    plt.close(fig)
    print(f"\nwrote {p1}\nwrote {p2}")


if __name__ == "__main__":
    main()
