#!/usr/bin/env python3
"""Screening analysis (reframing after checkpoint 1): how much of the assessment work a certified triage takes off FE.

A plate with demand D is decided as soon as a certified bracket [LB, UB] satisfies D < LB (pass) or D > UB (fail).
Demand scenarios: D = rho * P_FE with P_FE the FE mid-bracket on the common thickness model t_g, rho in RHOS.
Data (no new runs; validation plates only, plus the leave-one-bridge-out brackets of the real scans):
  - method: paper/figures/cache/val_histories_tg.csv (corrected method, every event certified by verifier v2, idle);
  - FE: loops/scratch/fe_levels_0_32_L2.csv (levels 0-2, idle) and fe_levels_0_32.csv (levels 3-4, loaded; scaled to
    idle by the measured factor of the level-0..2 timings);
  - real scans: results/runs/lobo/lobo_eval.csv (network alone and full method, t_g, v2), demand relative to the FE
    mid-bracket of results/runs/heldout-final/heldout_final.csv (bilinear model; differs from t_g by ~0.2 %).
Triage time per plate = time until the anytime method's best-so-far bracket decides (capped at its final bracket) +,
for plates still undecided, FE with "stop when decided" (levels 0..4). Baselines: FE with "stop when decided" for every
plate, and the FE reference (all levels). Usage: python loops/scratch/screening.py"""
import csv
import os

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
RHOS = (0.70, 0.80, 0.90, 0.95, 1.05, 1.10, 1.20, 1.30)
S = lambda *p: os.path.join(ROOT, *p)


def method_histories():
    ev = {}
    for r in csv.DictReader(open(S("paper", "figures", "cache", "val_histories_tg.csv"))):
        ev.setdefault(r["plate"], []).append((float(r["t"]), r["stage"], r["bound"], float(r["value"])))
    out = {}
    for p, E in ev.items():
        E.sort(key=lambda e: e[0])
        net = {bd: v for t, st, bd, v in E if st == "network"}
        t_net = min(t for t, st, _, _ in E if st == "network")
        steps, lb, ub = [], -np.inf, np.inf
        for t, st, bd, v in E:
            if st in ("network", "admm_scaled"):
                continue
            lb, ub = (max(lb, v), ub) if bd == "lb" else (lb, min(ub, v))
            if np.isfinite(lb) and np.isfinite(ub):
                steps.append((t, lb, ub))
        out[p] = dict(t_net=t_net, net=(net["lb"], net["ub"]), steps=steps, t_end=max(t for t, *_ in E))
    return out


def fe_levels():
    idle = list(csv.DictReader(open(S("loops", "scratch", "fe_levels_0_32_L2.csv"))))
    load = list(csv.DictReader(open(S("loops", "scratch", "fe_levels_0_32.csv"))))
    li = {(r["plate"], int(r["level"])): r for r in idle}
    ll = {(r["plate"], int(r["level"])): r for r in load}
    factor = np.median([float(ll[k]["t_cum"]) / float(li[k]["t_cum"]) for k in li])
    out = {}
    for p in sorted(set(r["plate"] for r in load)):
        lv = []
        for k in range(5):
            if (p, k) in li:
                t = float(li[(p, k)]["t_cum"])
                r = li[(p, k)]
            else:
                t = float(li[(p, 2)]["t_cum"]) + (float(ll[(p, k)]["t_cum"]) - float(ll[(p, 2)]["t_cum"])) / factor
                r = ll[(p, k)]
            lv.append((t, float(r["lb"]), float(r["ub"])))
        out[p] = lv
    return out, factor


def decided(lb, ub, d):
    return d < lb or d > ub


def main():
    M, (F, factor) = method_histories(), fe_levels()
    plates = sorted(set(M) & set(F))
    print(f"validation plates: {len(plates)}; FE level 3-4 times scaled by 1/{factor:.2f} (loaded -> idle)")
    rows, wrong = [], 0
    hdr = (f"{'demand':>7s} {'net':>6s} {'<=1s':>6s} {'<=5s':>6s} {'full':>6s} {'to FE':>6s} {'FE undec':>8s} | "
           f"{'triage s':>8s} {'FE-stop s':>9s} {'FE-ref s':>8s} {'speed-up':>8s}")
    print(hdr)
    for rho in RHOS:
        n_net = n1 = n5 = nfull = nfe = nund = 0
        t_tri, t_fes, t_fer = [], [], []
        for p in plates:
            m, fe = M[p], F[p]
            pref = 0.5 * (fe[-1][1] + fe[-1][2])
            d = rho * pref
            truth = "pass" if d < fe[-1][1] else "fail" if d > fe[-1][2] else None
            # method stages
            dec_net = decided(*m["net"], d)
            t_dec = None
            for t, lb, ub in m["steps"]:
                if decided(lb, ub, d):
                    t_dec = t
                    if truth and ("pass" if d < lb else "fail") != truth:
                        wrong += 1
                    break
            if dec_net:
                n_net += 1
                if truth and ("pass" if d < m["net"][0] else "fail") != truth:
                    wrong += 1
            n1 += int(dec_net or (t_dec is not None and t_dec <= 1.0))
            n5 += int(dec_net or (t_dec is not None and t_dec <= 5.0))
            nfull += int(dec_net or t_dec is not None)
            # FE with "stop when decided"
            t_fe_stop = next((t for t, lb, ub in fe if decided(lb, ub, d)), fe[-1][0])
            if not any(decided(lb, ub, d) for _, lb, ub in fe):
                nund += 1
            t_fes.append(t_fe_stop)
            t_fer.append(fe[-1][0])
            if dec_net:
                t_tri.append(m["t_net"])
            elif t_dec is not None:
                t_tri.append(t_dec)
            else:
                nfe += 1
                t_tri.append(m["t_end"] + t_fe_stop)
        n = len(plates)
        pc = lambda k: 100 * k / n
        rows.append(dict(rho=rho, net=pc(n_net), le1s=pc(n1), le5s=pc(n5), full=pc(nfull), to_fe=pc(nfe),
                         fe_undecided=pc(nund), triage_s=np.mean(t_tri), fe_stop_s=np.mean(t_fes), fe_ref_s=np.mean(t_fer)))
        r = rows[-1]
        print(f"{rho:7.2f} {r['net']:5.0f}% {r['le1s']:5.0f}% {r['le5s']:5.0f}% {r['full']:5.0f}% {r['to_fe']:5.0f}% "
              f"{r['fe_undecided']:7.0f}% | {r['triage_s']:8.1f} {r['fe_stop_s']:9.1f} {r['fe_ref_s']:8.1f} "
              f"{r['fe_stop_s'] / r['triage_s']:7.1f}x")
    print(f"wrong decisions against the FE reference: {wrong}")
    # real scans: leave-one-bridge-out brackets
    ho = {r["name"]: r for r in csv.DictReader(open(S("docs", "runs", "heldout-final", "heldout_final.csv"))) if r["set"] == "real-scan"}
    L = list(csv.DictReader(open(S("docs", "runs", "lobo", "lobo_eval.csv"))))
    print("\nreal scans (demand relative to the FE mid-bracket): share decided by the network alone / by the full method")
    print(f"{'demand':>7s} | {'Ananai: control':>16s} {'unseen bridge':>14s} | {'Amarube: control':>17s} {'unseen bridge':>14s}")
    wrong_r = 0
    for rho in RHOS:
        cells = []
        for fam, unseen in (("ananai", "amarube_only"), ("amarube", "ananai_only")):
            for net in ("both_control", unseen):
                rr = [r for r in L if r["set"] == "real" and r["family"] == fam and r["net"] == net]
                a = b = 0
                for r in rr:
                    f = ho[r["plate"]]
                    pref = 1e3 * 0.5 * (float(f["FE_LB_kN"]) + float(f["FE_UB_kN"]))
                    d = rho * pref
                    for lo, hi, flag in ((float(r["net_lb"]), float(r["net_ub"]), "a"), (float(r["full_lb"]), float(r["full_ub"]), "b")):
                        if decided(lo, hi, d):
                            if flag == "a":
                                a += 1
                            else:
                                b += 1
                            truth = "pass" if d < 1e3 * float(f["FE_LB_kN"]) else "fail" if d > 1e3 * float(f["FE_UB_kN"]) else None
                            if truth and ("pass" if d < lo else "fail") != truth:
                                wrong_r += 1
                cells.append(f"{100 * a / len(rr):3.0f}% / {100 * b / len(rr):3.0f}%")
        print(f"{rho:7.2f} | {cells[0]:>16s} {cells[1]:>14s} | {cells[2]:>17s} {cells[3]:>14s}")
    print(f"wrong decisions on the real scans against the FE reference: {wrong_r}")
    with open(S("loops", "scratch", "screening_val.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


if __name__ == "__main__":
    main()
