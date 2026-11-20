#!/usr/bin/env python3
"""Synthetic corroded steel plates, calibrated on the scanned plates.

Model of one plate
------------------
Thickness map t(x, y) on a 2 mm grid (x = loading direction):

    t / t0 = Q( rank of G(x, y) ),     G = sqrt(1 - s) * G_stat + sqrt(s) * ramp

* Q is a thickness histogram (quantile function of t/t0). It carries the mean loss, the
  roughness and the deep-pit tail (thickness is strongly skewed: a few deep pits).
* ramp is a planar corrosion gradient (unit variance) in direction theta; s is its share of
  the variance. In the scans it explains 6-64 % of the variance, mostly ACROSS the width
  (one edge of the flange more corroded than the other).
* G_stat is a stationary Gaussian random field whose correlation has three parts:
      rho(dx, dy) = w0 * [dx = dy = 0]                      fine roughness / scan noise
                  + w1 * exp(-(dx/lx)^2 - (dy/ly)^2)         pits: smooth, round blobs
                  + w2 * exp(-sqrt((dx/Lx)^2 + (dy/Ly)^2))   patches
  Each part may be stretched along x by at most 4x (no artificial streaks).
* The loss 1 - t/t0 is split between the faces: share p on the top face, 1 - p on the
  bottom, plus an independent redistribution field, so the two faces are uncorrelated
  with variance shares p and 1 - p (Amarube: p ~ 0.95, one face; Ananai: mixed).

Calibration (one parameter set per real plate)
----------------------------------------------
1. Q, p, s, theta: read directly from the scan (histogram; plane-removed SD of each face;
   least-squares planar trend of t/t0).
2. w0, w1, w2, lx, ly, Lx, Ly: least-squares fit to the autocorrelation of the de-trended scan.
3. Indirect inference: the rank transform shortens correlations, so the lengths are rescaled per
   axis until simulated plates reproduce the scan's de-trended 1/e correlation lengths.

Sampling a synthetic plate
--------------------------
Pick a bridge family (Ananai-like or Amarube-like, in proportion to the scans), two real parents
of that family and a weight lam ~ Beta(0.5, 0.5); mix their parameter sets (histograms by quantile
averaging, lengths geometrically, the rest linearly), add small random perturbations, choose which
edge / end is more corroded at random, and draw a field. Plates are flat (no initial distortion);
the output matches surfaces/*.csv.

Usage
-----
python corrosion_generator.py --data ../corroded_steel_plates_data_en --n 1000 --out ./data
-> data/synthetic_plates.npz, data/generator_calibration.csv
"""
import argparse
import csv
import glob
import os
import zlib

import numpy as np
from scipy.fft import next_fast_len
from scipy.optimize import least_squares

from scans import load_surface, read_table, surface_to_grid

H = 2.0                       # grid spacing, mm
NQ = 512                      # resolution of the stored histograms (quantile functions)
PROBS = (np.arange(NQ) + 0.5) / NQ
T_MIN = 0.03                  # floor on t/t0 (no through-holes)
S_MAX = 0.8                   # cap on the gradient share
FAMILIES = {"ananai": "Ananai River Bridge", "amarube": "Amarube Bridge"}
ANISO = (0.25, 2.0)           # allowed ly/lx and Ly/Lx (x may be up to 4x longer, y up to 2x)
JITTER = dict(loss=0.12, length=0.12, size=0.03, weight=0.05, share=0.05, angle=0.15)
LEN_KEYS = ("lx", "ly", "Lx", "Ly")


# ----------------------------------------------------------------------------- statistics (identical for real and synthetic)
def acf_biased(g, axis):
    a = np.moveaxis(g - g.mean(), axis, 0)
    n = a.shape[0]
    F = np.fft.rfft(a, n=2 * n, axis=0)
    c = np.fft.irfft(F * np.conj(F), n=2 * n, axis=0)[:n].mean(axis=1)
    return c / c[0]


def corr_len(g, axis):
    """1/e correlation length in mm, linearly interpolated between grid lags."""
    c = acf_biased(g, axis)
    k = int(np.argmax(c < np.exp(-1.0)))
    if k == 0:
        return np.nan
    return (k - 1 + (c[k - 1] - np.exp(-1.0)) / (c[k - 1] - c[k])) * H


def std_coords(nx, ny):
    X, Y = np.meshgrid(np.arange(nx) * H, np.arange(ny) * H, indexing="ij")
    return (X - X.mean()) / X.std(), (Y - Y.mean()) / Y.std()


def planar_trend(g):
    """Least-squares plane through g: returns (trend field, share of variance, |angle| in [0, pi/2])."""
    xh, yh = std_coords(*g.shape)
    A = np.stack([np.ones(g.size), xh.ravel(), yh.ravel()], 1)
    coef, *_ = np.linalg.lstsq(A, g.ravel(), rcond=None)
    trend = (coef[1] * xh + coef[2] * yh)
    return trend, float(trend.var() / g.var()), float(np.arctan2(abs(coef[2]), abs(coef[1])))


def plane_sd(xs, ys, z):
    X, Y = np.meshgrid(xs, ys, indexing="ij")
    A = np.stack([np.ones(X.size), X.ravel(), Y.ravel()], 1)
    coef, *_ = np.linalg.lstsq(A, z.ravel(), rcond=None)
    return float(np.std(z.ravel() - A @ coef))


def plate_stats(xs, ys, t, t0, z_top, z_bottom):
    tt = t / t0
    v = tt.ravel() - tt.mean()
    sd_top, sd_bot = plane_sd(xs, ys, z_top), plane_sd(xs, ys, z_bottom)
    trend, share, angle = planar_trend(tt)
    return dict(
        mean_t=float(tt.mean()),
        sd_t=float(tt.std()),
        skew_t=float((v ** 3).mean() / (v ** 2).mean() ** 1.5),
        min_avg_t=float(tt.mean(axis=1).min()),            # min over x-sections of the width-averaged t
        max_depth=float(1.0 - tt.min()),
        gradient_share=share,
        gradient_angle_deg=float(np.degrees(angle)),       # 0 = along the plate, 90 = across the width
        corr_len_x_mm=float(corr_len(tt, 0)),
        corr_len_y_mm=float(corr_len(tt, 1)),
        corr_len_x_detrended_mm=float(corr_len(tt - trend, 0)),
        corr_len_y_detrended_mm=float(corr_len(tt - trend, 1)),
        sd_corroded_face_mm=max(sd_top, sd_bot),
        sd_other_face_mm=min(sd_top, sd_bot),
    )


# ----------------------------------------------------------------------------- correlation model
def acf_1d(r, w, l, L):
    return w[0] * (r == 0) + w[1] * np.exp(-(r / l) ** 2) + w[2] * np.exp(-r / L)


def fit_acf(resid):
    """Fit the three-part correlation to the (overlap-corrected) autocorrelation along x and y."""
    def unbiased(axis, max_lag):
        a = np.moveaxis(resid - resid.mean(), axis, 0)
        n = a.shape[0]
        F = np.fft.rfft(a, n=2 * n, axis=0)
        c = np.fft.irfft(F * np.conj(F), n=2 * n, axis=0)[:n].mean(axis=1) / (n - np.arange(n))
        k = min(max_lag, n // 2)
        return np.arange(1, k + 1) * H, (c / c[0])[1:k + 1]

    rx, cx = unbiased(0, 40)       # 2 ... 80 mm along the plate
    ry, cy = unbiased(1, 15)       # 2 ... 30 mm across

    def unpack(q):
        e = np.exp([0.0, q[0], q[1]])
        lx, Lx = np.exp(q[2]), np.exp(q[4])
        return dict(w=e / e.sum(), lx=lx, ly=lx * np.exp(q[3]), Lx=Lx, Ly=Lx * np.exp(q[5]))

    def res(q):
        p = unpack(q)
        return np.concatenate([acf_1d(rx, p["w"], p["lx"], p["Lx"]) - cx,
                               acf_1d(ry, p["w"], p["ly"], p["Ly"]) - cy])

    q0 = [1.0, 1.0, np.log(8.0), np.log(0.7), np.log(40.0), np.log(0.5)]
    lo = [-8, -8, np.log(1.5), np.log(ANISO[0]), np.log(5.0), np.log(ANISO[0])]
    hi = [8, 8, np.log(40.0), np.log(ANISO[1]), np.log(400.0), np.log(ANISO[1])]
    sol = least_squares(res, q0, bounds=(lo, hi))
    p = unpack(sol.x)
    p["acf_rmse"] = float(np.sqrt(np.mean(sol.fun ** 2)))
    return p


# ----------------------------------------------------------------------------- random fields
def gaussian_field_pair(rng, nx, ny, prm):
    """Two independent unit-variance stationary Gaussian fields (nx x ny, spacing H)."""
    reach_x = max(3 * prm["Lx"], 3 * prm["lx"])
    reach_y = max(3 * prm["Ly"], 3 * prm["ly"])
    Mx = next_fast_len(int(min(4 * nx, nx + np.ceil(reach_x / H))))
    My = next_fast_len(int(min(6 * ny, ny + np.ceil(reach_y / H))))
    dx = np.minimum(np.arange(Mx), Mx - np.arange(Mx)) * H
    dy = np.minimum(np.arange(My), My - np.arange(My)) * H
    DX, DY = np.meshgrid(dx, dy, indexing="ij")
    w = prm["w"]
    c = (w[1] * np.exp(-(DX / prm["lx"]) ** 2 - (DY / prm["ly"]) ** 2)
         + w[2] * np.exp(-np.hypot(DX / prm["Lx"], DY / prm["Ly"])))
    c[0, 0] += w[0]
    lam = np.clip(np.fft.fft2(c).real, 0.0, None)        # circulant embedding (negatives clipped)
    xi = rng.standard_normal((Mx, My)) + 1j * rng.standard_normal((Mx, My))
    z = np.fft.fft2(np.sqrt(lam / (Mx * My)) * xi)
    return z.real[:nx, :ny], z.imag[:nx, :ny]


def realize(rng, nx, ny, prm, Q, p, t0):
    """One plate: (thickness, z_top, z_bottom) in mm on an nx x ny grid."""
    gS, gD = gaussian_field_pair(rng, nx, ny, prm)
    xh, yh = std_coords(nx, ny)
    sx, sy = rng.choice([-1.0, 1.0], size=2)               # which end / edge is more corroded
    ramp = sx * np.cos(prm["theta"]) * xh + sy * np.sin(prm["theta"]) * yh
    G = np.sqrt(1.0 - prm["s"]) * gS + np.sqrt(prm["s"]) * ramp
    u = (np.argsort(np.argsort(G.ravel())) + 0.5) / G.size          # ranks -> uniform
    tt = np.interp(u, PROBS, Q).reshape(nx, ny)
    loss = 1.0 - tt
    delta = (gD - gD.mean()) / gD.std() * np.sqrt(p * (1 - p)) * loss.std()
    d_top, d_bot = p * loss + delta, (1 - p) * loss - delta
    z_top, z_bot = t0 * (0.5 - d_top), t0 * (-0.5 + d_bot)
    mid = 0.5 * (z_top + z_bot).mean()
    return t0 * tt, z_top - mid, z_bot - mid


# ----------------------------------------------------------------------------- calibration
def match_lengths(prm, Q, p, t0, nx, ny, target_x, target_y, seed, n_sim=6, n_iter=5):
    """Rescale lengths per axis until simulated de-trended 1/e lengths match the scan's."""
    kx = ky = 1.0
    for _ in range(n_iter):
        trial = dict(prm, lx=prm["lx"] * kx, Lx=prm["Lx"] * kx, ly=prm["ly"] * ky, Ly=prm["Ly"] * ky)
        rng = np.random.default_rng(seed)                 # common random numbers: smooth iteration
        sx, sy = [], []
        for _ in range(n_sim):
            t, _, _ = realize(rng, nx, ny, trial, Q, p, t0)
            d = t - planar_trend(t)[0]
            sx.append(corr_len(d, 0))
            sy.append(corr_len(d, 1))
        kx = float(np.clip(kx * (target_x / np.nanmedian(sx)) ** 0.9, 0.3, 4.0))
        ky = float(np.clip(ky * (target_y / np.nanmedian(sy)) ** 0.9, 0.3, 4.0))
    return dict(prm, lx=prm["lx"] * kx, Lx=prm["Lx"] * kx, ly=prm["ly"] * ky, Ly=prm["Ly"] * ky,
                scale_x=kx, scale_y=ky)


def calibrate(data_dir, verbose=False):
    spec = {r["specimen_id"]: r for r in read_table(os.path.join(data_dir, "specimens.csv"))}
    parents = []
    for fam in FAMILIES:
        for path in sorted(glob.glob(os.path.join(data_dir, "surfaces", fam, "*.csv"))):
            sid = os.path.splitext(os.path.basename(path))[0]
            t0 = float(spec[sid]["t0_mm"])
            xs, ys, g = surface_to_grid(load_surface(path))
            tt = g["thickness_mm"] / t0
            sd_top, sd_bot = plane_sd(xs, ys, g["z_top_mm"]), plane_sd(xs, ys, g["z_bottom_mm"])
            trend, s, theta = planar_trend(tt)
            resid = tt - trend
            prm = dict(fit_acf(resid), s=min(s, S_MAX), theta=theta)
            Q = np.quantile(tt.ravel(), PROBS)
            p = sd_top ** 2 / (sd_top ** 2 + sd_bot ** 2)
            prm = match_lengths(prm, Q, p, t0, len(xs), len(ys), corr_len(resid, 0), corr_len(resid, 1),
                                seed=zlib.crc32(sid.encode()))
            parents.append(dict(specimen_id=sid, family=fam, t0=t0, L=float(xs.max()), W=float(ys.max()),
                                Q=Q, p_top=p, grid=(xs, ys, g), **prm))
            if verbose:
                print(f"  {sid}: gradient {100 * s:3.0f}% at {np.degrees(theta):3.0f} deg | "
                      f"w=({prm['w'][0]:.2f},{prm['w'][1]:.2f},{prm['w'][2]:.2f}) "
                      f"pits {prm['lx']:.1f}x{prm['ly']:.1f} mm, patches {prm['Lx']:.0f}x{prm['Ly']:.0f} mm, "
                      f"scale ({prm['scale_x']:.2f},{prm['scale_y']:.2f}), p_top={p:.2f}")
    return parents


# ----------------------------------------------------------------------------- sampling
def generate_plate(rng, parents, family):
    pool = [p for p in parents if p["family"] == family]
    i, j = rng.choice(len(pool), size=2, replace=False)
    A, B = pool[i], pool[j]
    lam = rng.beta(0.5, 0.5)

    def gmix(a, b, s):
        return float(np.exp((1 - lam) * np.log(a) + lam * np.log(b) + rng.normal(0.0, s)))

    def lmix(a, b, s, lo, hi):
        return float(np.clip((1 - lam) * a + lam * b + rng.normal(0.0, s), lo, hi))

    a_loss = np.exp(rng.normal(0.0, JITTER["loss"]))
    Q = np.clip(1.0 - a_loss * (1.0 - ((1 - lam) * A["Q"] + lam * B["Q"])), T_MIN, None)
    w = np.clip((1 - lam) * A["w"] + lam * B["w"] + rng.normal(0.0, JITTER["weight"], 3), 1e-3, None)
    prm = dict(w=w / w.sum(), **{k: gmix(A[k], B[k], JITTER["length"]) for k in LEN_KEYS},
               s=lmix(A["s"], B["s"], JITTER["share"], 0.0, S_MAX),
               theta=lmix(A["theta"], B["theta"], JITTER["angle"], 0.0, np.pi / 2))
    p = A["p_top"] if rng.uniform() > lam else B["p_top"]
    L, W = gmix(A["L"], B["L"], JITTER["size"]), gmix(A["W"], B["W"], JITTER["size"])
    nx, ny = int(round(L / H)) + 1, int(round(W / H)) + 1
    t, z_top, z_bot = realize(rng, nx, ny, prm, Q, p, A["t0"])
    return dict(xs=np.arange(nx) * H, ys=np.arange(ny) * H, thickness=t, z_top=z_top, z_bottom=z_bot,
                t0=A["t0"], family=family, parent_a=A["specimen_id"], parent_b=B["specimen_id"],
                lam=float(lam), p_top=float(p), **prm)


def generate(parents, n, seed=0):
    rng = np.random.default_rng(seed)
    counts = {f: sum(p["family"] == f for p in parents) for f in FAMILIES}
    fams = rng.choice(list(counts), size=n, p=np.array(list(counts.values())) / sum(counts.values()))
    return [generate_plate(rng, parents, f) for f in fams]


# ----------------------------------------------------------------------------- I/O
def save_synthetic(plates, path):
    def obj(key):
        o = np.empty(len(plates), dtype=object)
        for k, s in enumerate(plates):
            o[k] = np.asarray(s[key], dtype=np.float32)
        return o

    np.savez_compressed(
        path,
        x_mm=obj("xs"), y_mm=obj("ys"),
        thickness_mm=obj("thickness"), z_top_mm=obj("z_top"), z_bottom_mm=obj("z_bottom"),
        t0_mm=np.array([s["t0"] for s in plates]),
        family=np.array([s["family"] for s in plates]),
        parent_a=np.array([s["parent_a"] for s in plates]),
        parent_b=np.array([s["parent_b"] for s in plates]),
        lam=np.array([s["lam"] for s in plates]),
        p_top=np.array([s["p_top"] for s in plates]),
        gradient_share=np.array([s["s"] for s in plates]),
        gradient_angle_rad=np.array([s["theta"] for s in plates]),
        weights_nugget_pit_patch=np.array([s["w"] for s in plates]),
        lengths_lx_ly_Lx_Ly_mm=np.array([[s[k] for k in LEN_KEYS] for s in plates]),
        note=np.array("fields are (nx, ny) arrays on x_mm (loading direction) x y_mm, 2 mm grid; "
                      "thickness = z_top - z_bottom; mean mid-surface z = 0; generated by corrosion_generator.py"),
    )


def load_synthetic(path):
    d = np.load(path, allow_pickle=True)
    cols = {k: d[k] for k in d.files}                      # read each array once
    return [dict(xs=cols["x_mm"][k], ys=cols["y_mm"][k], thickness=cols["thickness_mm"][k],
                 z_top=cols["z_top_mm"][k], z_bottom=cols["z_bottom_mm"][k], t0=float(cols["t0_mm"][k]),
                 family=str(cols["family"][k]), parent_a=str(cols["parent_a"][k]),
                 parent_b=str(cols["parent_b"][k]), lam=float(cols["lam"][k]))
            for k in range(len(cols["family"]))]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True, help="corroded_steel_plates_data_en folder")
    ap.add_argument("--n", type=int, default=1000, help="number of synthetic plates")
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--out", default="./data")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    print("calibrating on the scans ...")
    parents = calibrate(args.data, verbose=True)
    cal = os.path.join(args.out, "generator_calibration.csv")
    with open(cal, "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["specimen_id", "family", "t0_mm", "L_mm", "W_mm", "p_top", "gradient_share",
                     "gradient_angle_deg", "w_nugget", "w_pit", "w_patch", "pit_lx_mm", "pit_ly_mm",
                     "patch_Lx_mm", "patch_Ly_mm", "length_scale_x", "length_scale_y", "acf_fit_rmse"])
        for p in parents:
            wr.writerow([p["specimen_id"], p["family"], p["t0"], p["L"], p["W"], round(p["p_top"], 4),
                         round(p["s"], 4), round(float(np.degrees(p["theta"])), 1),
                         *np.round(p["w"], 4), *[round(p[k], 2) for k in LEN_KEYS],
                         round(p["scale_x"], 3), round(p["scale_y"], 3), round(p["acf_rmse"], 4)])
    print(f"-> {cal}")

    plates = generate(parents, args.n, args.seed)
    out = os.path.join(args.out, "synthetic_plates.npz")
    save_synthetic(plates, out)
    print(f"generated {len(plates)} plates -> {out}")


if __name__ == "__main__":
    main()
