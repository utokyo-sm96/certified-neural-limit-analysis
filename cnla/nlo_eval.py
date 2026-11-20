#!/usr/bin/env python3
"""FROZEN evaluator for loop nlo-1106 (neural limit operator). Do not edit inside the loop.

1. Train:   runs `python nlo_train.py` in a subprocess with a hard timeout; it must write nlo_train.CKPT.
2. Predict: imports nlo_train and calls predict(load(CKPT), plates) on a uniform strip and on the
            32 validation plates (data/nlo_val.npz). Only geometry, thickness and sigma_y are passed.
3. Certify: every result goes through the frozen step-1 verifier eval_limit_analysis.verify, which
            re-computes the upper bound from the velocities and re-checks equilibrium, traction
            continuity, free edges and yield of the stresses with its own code. The network has no
            solver status, so the evaluator sets it; nothing else is taken on trust.

Gates (all must pass; any failure means discard):
  G1  uniform strip 236 x 76 mm, t = 10.5 mm, rigid grips: certified; UB >= sigma_y W t (1 - 1e-6), which
      is a proven lower bound; LB <= UB of the step-1 FE solution (1 + 1e-6)
  G2  every validation plate: certified by verify() (no errors)
  G3  weak duality against the FE references: LB_nn <= UB_ref (1 + 1e-6) and UB_nn >= LB_ref (1 - 1e-6)
  G4  training subprocess exits with code 0 within TRAIN_TIMEOUT and writes a new checkpoint
  G5  inference time <= 0.5 s per plate (all of predict(), including the float64 certification layer)
  G6  no element edge longer than 2.0 mm (no mesh coarser than the 2 mm scan grid)
Metric: gap_mean = mean over validation plates of (UB - LB) / UB, both certified (lower is better).
"""
import csv
import os
import subprocess
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from eval_limit_analysis import verify  # noqa: E402  (frozen step-1 verifier)

TRAIN_TIMEOUT = 720.0
INFER_LIMIT_S = 0.5
MAX_EDGE_MM = 2.0
VAL = os.path.join(HERE, "data", "nlo_val.npz")
REF = os.path.join(HERE, "data", "nlo_val_ref.csv")


def max_edge(res):
    p = np.asarray(res["nodes"])[np.asarray(res["tri"])]
    return float(max(np.linalg.norm(p[:, a] - p[:, b], axis=1).max() for a, b in ((0, 1), (1, 2), (2, 0))))


def certify(res, xs, ys, t, sy):
    res = dict(res)
    res["status_ub"] = res["status_lb"] = "Solved"          # a network has no solver status
    return verify(res, xs, ys, t, sy, "rigid")


def main():
    t_start = time.time()
    gate_errs = []

    # ---- 1. train
    env = {k: v for k, v in os.environ.items() if not k.startswith("NLO_")}
    ckpt_before = time.time()
    t1 = time.time()
    try:
        p = subprocess.run([sys.executable, "-u", os.path.join(HERE, "nlo_train.py")], cwd=HERE, env=env,
                           capture_output=True, text=True, timeout=TRAIN_TIMEOUT)
        log, code = p.stdout + p.stderr, p.returncode
    except subprocess.TimeoutExpired as e:
        log, code = (e.stdout or "") + (e.stderr or ""), "timeout"
        log = log.decode() if isinstance(log, bytes) else log
    train_s = time.time() - t1
    print("---- training log (tail)")
    print("\n".join(log.strip().splitlines()[-12:]))
    import nlo_train  # noqa: E402  (imported after training, so the editable module is the one just run)
    if code != 0 or not os.path.exists(nlo_train.CKPT) or os.path.getmtime(nlo_train.CKPT) < ckpt_before:
        print("---")
        print("gap_mean: 1.000000")
        print(f"gates: FAIL G4 training (code {code})")
        print(f"runtime_s: {time.time() - t_start:.1f}")
        return
    model = nlo_train.load(nlo_train.CKPT)
    print("---- evaluation")

    # ---- G1 uniform strip
    from limit_analysis import solve_bounds
    sy, t0 = 281.6, 10.5
    xs, ys = np.arange(0, 236, 2.0), np.arange(0, 76, 2.0)
    tu = np.full((len(xs), len(ys)), t0)
    exact_lb = sy * ys[-1] * t0
    ref = solve_bounds(xs, ys, tu, sy, grips="rigid")
    ub_fe, _, _, _ = verify(ref, xs, ys, tu, sy, "rigid")
    r = nlo_train.predict(model, [dict(xs=xs, ys=ys, t=tu, sigma_y=sy)])[0]
    ub, lb, errs, _ = certify(r, xs, ys, tu, sy)
    print(f"G1 uniform strip: UB/(sy W t)={ub / exact_lb:.5f}  LB/(sy W t)={lb / exact_lb:.5f}  "
          f"FE UB/(sy W t)={ub_fe / exact_lb:.5f}  {errs if errs else ''}")
    if errs or ub < exact_lb * (1 - 1e-6) or lb > ub_fe * (1 + 1e-6):
        gate_errs.append("G1")
    if max_edge(r) > MAX_EDGE_MM:
        gate_errs.append(f"G6 max edge {max_edge(r):.2f} mm")

    # ---- validation plates
    d = np.load(VAL, allow_pickle=True)
    refs = {row["name"]: row for row in csv.DictReader(open(REF))}
    plates = [dict(xs=d["x_mm"][i], ys=d["y_mm"][i], t=d["thickness_mm"][i], sigma_y=float(d["sigma_y_MPa"][i]))
              for i in range(len(d["name"]))]
    t1 = time.time()
    results = nlo_train.predict(model, plates)
    infer = (time.time() - t1) / len(plates)
    if infer > INFER_LIMIT_S:
        gate_errs.append(f"G5 inference {infer:.2f} s/plate")
    gaps, ub_ex, lb_def, floor = [], [], [], []
    for name, p, res in zip(d["name"], plates, results):
        ub, lb, errs, info = certify(res, p["xs"], p["ys"], p["t"], p["sigma_y"])
        rf = refs[str(name)]
        ub_ref, lb_ref = float(rf["UB_ref_N"]), float(rf["LB_ref_N"])
        gap = (ub - lb) / ub
        gaps.append(gap)
        ub_ex.append(ub / ub_ref - 1)
        lb_def.append(1 - lb / lb_ref)
        floor.append(float(rf["gap_mesh"]))
        me = max_edge(res)
        if errs:
            gate_errs.append(f"G2 {name}")
        if lb > ub_ref * (1 + 1e-6) or ub < lb_ref * (1 - 1e-6):
            gate_errs.append(f"G3 {name}")
        if me > MAX_EDGE_MM:
            gate_errs.append(f"G6 {name}")
        print(f"{name}  UB {ub / 1e3:7.2f}  LB {lb / 1e3:7.2f} kN  gap {100 * gap:6.2f}%  "
              f"(ref [{lb_ref / 1e3:.2f}, {ub_ref / 1e3:.2f}], UB +{100 * ub_ex[-1]:.2f}%, LB -{100 * lb_def[-1]:.2f}%, "
              f"mesh floor {100 * floor[-1]:.2f}%)  yield {info['yield_ratio']:.4f}  {errs if errs else ''}")
    print("---")
    print(f"gap_mean: {np.mean(gaps):.6f}")
    print("gates: " + ("PASS" if not gate_errs else "FAIL " + ",".join(gate_errs[:6])))
    print(f"runtime_s: {time.time() - t_start:.1f}")
    print(f"gap_median: {np.median(gaps):.6f}")
    print(f"gap_max: {np.max(gaps):.6f}")
    print(f"ub_excess_mean: {np.mean(ub_ex):.6f}")
    print(f"lb_deficit_mean: {np.mean(lb_def):.6f}")
    print(f"mesh_floor_gap_mean: {np.mean(floor):.6f}")
    print(f"train_s: {train_s:.1f}")
    print(f"infer_ms_per_plate: {1000 * infer:.0f}")


if __name__ == "__main__":
    main()
