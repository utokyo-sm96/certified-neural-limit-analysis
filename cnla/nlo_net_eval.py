#!/usr/bin/env python3
"""FROZEN evaluator for loop nlo-net-1107: network training judged end to end.

1. Train:   runs `python nlo_train.py` (EDITABLE in this loop) in a subprocess with a hard timeout; it must
            write nlo_train.CKPT. The training budget is the script's own (480 s); the evaluator kills it at 720 s.
2. Network: certified bracket of the network alone (one forward pass)          -> net_gap_mean (secondary)
3. Hybrid:  the frozen refinement pipeline nlo_refine.bounds (mirror combination + local cone-program
            corrections) on the newly trained network, one plate at a time, timed  -> gap_mean (the metric)
Both are certified by the frozen step-1 verifier through nlo_refine_eval.certify_ub / certify_lb.

Gates (any failure means discard):
  G0  training exits cleanly within the timeout and writes a fresh checkpoint
  G1  uniform strip 236 x 76 x 10.5 mm (hybrid): certified; UB >= sigma_y W t; LB <= step-1 FE UB
  G2  every validation plate: network and hybrid UB/LB certified
  G3  weak duality against the step-1 FE references (network and hybrid), LB <= UB
  G4  hybrid time per plate: mean <= 5 s, max <= 10 s; network inference <= 0.5 s per plate
  G5  no element edge > 2.0 mm
Metric: gap_mean = mean over the 32 validation plates of the hybrid's certified (UB - LB) / UB.
"""
import csv
import os
import subprocess
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from eval_limit_analysis import verify  # noqa: E402
from nlo_refine_eval import MAX_EDGE_MM, certify_lb, certify_ub, max_edge  # noqa: E402

TRAIN_TIMEOUT = 720.0
VAL = os.path.join(HERE, "data", "nlo_val.npz")
REF = os.path.join(HERE, "data", "nlo_val_ref.csv")


def main():
    t_start = time.time()
    gate_errs = []
    env = {k: v for k, v in os.environ.items() if not k.startswith("NLO_")}
    t0 = time.time()
    try:
        p = subprocess.run([sys.executable, "-u", os.path.join(HERE, "nlo_train.py")], cwd=HERE, env=env,
                           capture_output=True, text=True, timeout=TRAIN_TIMEOUT)
        log, code = p.stdout + p.stderr, p.returncode
    except subprocess.TimeoutExpired as e:
        log = e.stdout.decode() if isinstance(e.stdout, bytes) else (e.stdout or "")
        code = "timeout"
    train_s = time.time() - t0
    print("---- training log (tail)")
    print("\n".join(log.strip().splitlines()[-8:]))
    steps = next((ln.split()[1] for ln in log.splitlines() if ln.startswith("done:")), "nan")
    import nlo_train
    if code != 0 or not os.path.exists(nlo_train.CKPT) or os.path.getmtime(nlo_train.CKPT) < t0:
        print("---")
        print("gap_mean: 1.000000")
        print(f"gates: FAIL G0 training (code {code})")
        print(f"runtime_s: {time.time() - t_start:.1f}")
        return
    import nlo_refine
    model = nlo_train.load(nlo_train.CKPT)

    from limit_analysis import solve_bounds
    sy, t0u = 281.6, 10.5
    xs, ys = np.arange(0, 236, 2.0), np.arange(0, 76, 2.0)
    tu = np.full((len(xs), len(ys)), t0u)
    ub_fe, _, _, _ = verify(solve_bounds(xs, ys, tu, sy, grips="rigid"), xs, ys, tu, sy, "rigid")
    out = nlo_refine.bounds(model, dict(xs=xs, ys=ys, t=tu, sigma_y=sy))                    # also the warm-up
    ub, eu = certify_ub(out["ub_res"], xs, ys, tu, sy)
    lb, el = certify_lb(out["lb_res"], xs, ys, tu, sy)
    ex = sy * ys[-1] * t0u
    print(f"G1 uniform strip: UB/(sy W t)={ub / ex:.5f}  LB/(sy W t)={lb / ex:.5f}  {eu + el}")
    if eu or el or ub < ex * (1 - 1e-6) or lb > ub_fe * (1 + 1e-6):
        gate_errs.append("G1")

    d = np.load(VAL, allow_pickle=True)
    refs = {row["name"]: row for row in csv.DictReader(open(REF))}
    plates = [dict(xs=d["x_mm"][i], ys=d["y_mm"][i], t=d["thickness_mm"][i], sigma_y=float(d["sigma_y_MPa"][i]))
              for i in range(len(d["name"]))]
    t1 = time.time()
    net = nlo_train.predict(model, plates)
    infer = (time.time() - t1) / len(plates)
    if infer > 0.5:
        gate_errs.append(f"G4 network {infer:.2f}s/plate")
    gaps, net_gaps, times, ub_ex, lb_def = [], [], [], [], []
    for i, (p, r) in enumerate(zip(plates, net)):
        name = str(d["name"][i])
        rf = refs[name]
        ub_ref, lb_ref = float(rf["UB_ref_N"]), float(rf["LB_ref_N"])
        nub, e1 = certify_ub(r, p["xs"], p["ys"], p["t"], p["sigma_y"])
        nlb, e2 = certify_lb(r, p["xs"], p["ys"], p["t"], p["sigma_y"])
        net_gaps.append((nub - nlb) / nub)
        t2 = time.time()
        out = nlo_refine.bounds(model, p)
        dt = time.time() - t2
        ub, eu = certify_ub(out["ub_res"], p["xs"], p["ys"], p["t"], p["sigma_y"])
        lb, el = certify_lb(out["lb_res"], p["xs"], p["ys"], p["t"], p["sigma_y"])
        gaps.append((ub - lb) / ub)
        times.append(dt)
        ub_ex.append(ub / ub_ref - 1)
        lb_def.append(1 - lb / lb_ref)
        if e1 or e2 or eu or el:
            gate_errs.append(f"G2 {name}")
        for u_, l_ in ((nub, nlb), (ub, lb)):
            if l_ > ub_ref * (1 + 1e-6) or u_ < lb_ref * (1 - 1e-6) or l_ > u_ * (1 + 1e-6):
                gate_errs.append(f"G3 {name}")
        if max(max_edge(out["ub_res"]), max_edge(out["lb_res"]), max_edge(r)) > MAX_EDGE_MM:
            gate_errs.append(f"G5 {name}")
        print(f"{name}  net gap {100 * net_gaps[-1]:6.2f}%  hybrid UB {ub / 1e3:7.2f} LB {lb / 1e3:7.2f} kN gap {100 * gaps[-1]:6.2f}% "
              f"(UB +{100 * ub_ex[-1]:.2f}%, LB -{100 * lb_def[-1]:.2f}%)  {dt:4.1f}s {(e1 + e2 + eu + el) or ''}", flush=True)
    if np.mean(times) > 5.0 or np.max(times) > 10.0:
        gate_errs.append(f"G4 hybrid time mean {np.mean(times):.2f}s max {np.max(times):.2f}s")
    print("---")
    print(f"gap_mean: {np.mean(gaps):.6f}")
    print("gates: " + ("PASS" if not gate_errs else "FAIL " + ",".join(gate_errs[:6])))
    print(f"runtime_s: {time.time() - t_start:.1f}")
    print(f"net_gap_mean: {np.mean(net_gaps):.6f}")
    print(f"gap_max: {np.max(gaps):.6f}")
    print(f"ub_excess_mean: {np.mean(ub_ex):.6f}")
    print(f"lb_deficit_mean: {np.mean(lb_def):.6f}")
    print(f"time_mean_s: {np.mean(times):.3f}")
    print(f"train_s: {train_s:.1f}")
    print(f"train_steps: {steps}")


if __name__ == "__main__":
    main()
