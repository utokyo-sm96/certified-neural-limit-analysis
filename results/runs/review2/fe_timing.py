"""Controlled timing of the classical solvers on t_g (review 2, point 3): ONE process, plates one at a time, on an idle
machine, exactly as the method's histories were measured; no load-correction factor. For every plate of the validation,
synthetic test and real-scan sets:
  * fibre LB (cheap, non-learned): N = (sigma_y min_{strip} t_g, 0, 0) on each row strip of the scan grid; statically
    admissible (x-independent, zero tractions on horizontal lines and free edges, yield by construction);
  * scan-grid UB: the first pass of the adaptive FE solver (optimal UB on the scan triangulation of t_g);
  * adaptive FE levels 0, 1, ... (both bounds; level 4 = reference solver), stopped as soon as the certified gap is at most
    1 % (every threshold and screening decision with |rho - 1| >= 0.05 is then settled);
each bound certified by verifier v2, with its own verification time. Output: fe_timing.csv
(set, plate, kind, level, t_cum, t_verify_lb, t_verify_ub, lb, ub, errors)."""
import csv
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import OUT, plate_sets, v2_lb, v2_ub  # noqa: E402


def main():
    import limit_analysis as LA
    from thickness_tg import tg_eval
    rows = []
    out = os.path.join(OUT, "fe_timing.csv")
    for s, p in plate_sets():
        xs, ys, t, sy, name = np.asarray(p["xs"], float), np.asarray(p["ys"], float), np.asarray(p["t"], float), p["sigma_y"], p["name"]
        L, W = xs[-1], ys[-1]
        t0 = float(t.max())
        tol = 1e-6 * max(L, W)
        interp = lambda q: tg_eval(np.asarray(q, float), xs, ys, t)
        # ---- fibre LB (analytic, ms) on the scan triangulation
        c = time.perf_counter()
        nodes, tri = LA.grid_mesh(xs, ys)
        cy = nodes[tri][:, :, 1].mean(1)
        j = np.clip(np.searchsorted(ys, cy) - 1, 0, len(ys) - 2)
        strip = sy * np.minimum(t[:, :-1].min(0), t[:, 1:].min(0))      # min of t_g over strip j = min of rows j, j+1
        S = np.zeros((len(tri), 3, 3))
        S[:, :, 0] = strip[j][:, None]
        lb_fibre_claim = float(np.sum(strip * np.diff(ys)))
        t_fibre = time.perf_counter() - c
        th = interp(nodes)
        lbf, ef, tvf = v2_lb(nodes, tri, S, xs, ys, t, sy, th)
        rows.append(dict(set=s, plate=name, kind="fibre_lb", level=-1, t_cum=t_fibre, t_verify_lb=tvf, t_verify_ub=0.0,
                         lb=lbf, ub=np.nan, errors=";".join(ef)))
        # ---- adaptive FE: pass 1 (UB on the scan grid), then levels
        clock = time.perf_counter()
        nodes, tri = LA.grid_mesh(xs, ys)
        ub0 = LA.upper_bound(nodes, tri, interp(nodes), t0, L, "rigid", tol)
        t_pass1 = time.perf_counter() - clock
        ubp, eu, tvu = v2_ub(nodes, tri, ub0["v"], xs, ys, t, sy, interp(nodes))
        rows.append(dict(set=s, plate=name, kind="grid_ub", level=-1, t_cum=t_pass1, t_verify_lb=0.0, t_verify_ub=tvu,
                         lb=np.nan, ub=ubp, errors=";".join(eu)))
        ncell = (len(xs) - 1) * (len(ys) - 1)
        cell_diss = ub0["diss"][:ncell] + ub0["diss"][ncell:]
        rank = np.argsort(cell_diss)[::-1]
        if cell_diss.max() < LA.UNIFORM_RATIO * cell_diss.mean():
            rank = rank[:0]
        spent = t_pass1
        for cap in range(5):
            level = np.zeros(ncell, int)
            for lv, frac in ((1, LA.REFINE_FRACTION), (2, LA.LEVEL2_FRACTION), (3, LA.LEVEL3_FRACTION), (4, LA.LEVEL4_FRACTION)):
                if lv <= cap:
                    level[rank[:int(frac * ncell)]] = lv
            c0 = time.perf_counter()
            nodes, tri = LA.adaptive_mesh(xs, ys, level.reshape(len(xs) - 1, len(ys) - 1))
            thick = interp(nodes)
            ub = LA.upper_bound(nodes, tri, thick, t0, L, "rigid", tol)
            lb = LA.lower_bound(nodes, tri, thick, t0, L, W, "rigid", tol)
            spent += time.perf_counter() - c0
            scale = sy * t0
            ubc, eu, tvu = v2_ub(nodes, tri, ub["v"], xs, ys, t, sy, thick)
            lbc, el, tvl = v2_lb(nodes, tri, lb["stress_hat"] * scale, xs, ys, t, sy, thick)
            rows.append(dict(set=s, plate=name, kind="fe", level=cap, t_cum=spent, t_verify_lb=tvl, t_verify_ub=tvu,
                             lb=lbc, ub=ubc, errors=";".join(el + eu)))
            if (ubc - lbc) / ubc <= 0.01:
                break
        R_ = [r for r in rows if r["plate"] == name and r["set"] == s]
        print(f"{s:5s} {name}: fibre LB {lbf / 1e3:.1f} kN ({t_fibre * 1e3:.0f} ms), grid UB {ubp / 1e3:.1f} kN ({t_pass1:.2f} s); "
              + "  ".join(f"L{r['level']} {100 * (r['ub'] - r['lb']) / r['ub']:.2f}% {r['t_cum']:.0f}s" for r in R_ if r["kind"] == "fe")
              + ("" if not any(r["errors"] for r in R_) else "  ERRORS"), flush=True)
        with open(out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
    print(f"done: {len(rows)} rows, errors {sum(bool(r['errors']) for r in rows)}")


if __name__ == "__main__":
    main()
