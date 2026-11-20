#!/usr/bin/env python3
"""NLO with the C1-cubic (FVS) Airy layer: ZP load path + self-equilibrated FVS corrections (plan 6.B, probe stage).

The ZP path of nlo_train is unchanged: target stresses -> AiryFit -> ZP field N_zp (constant per triangle), which
carries the load. A new zero-initialised head outputs FVS DOF corrections (Psi, Psi_u, Psi_w at vertices, normal
derivatives at edge midpoints, cell units), zero on both free edges. Any such correction is a self-equilibrated,
traction-continuous, traction-free field with LINEAR stresses per triangle and zero resultant, so
    N(vertex) = N_zp + N_fvs(delta)     is statically admissible for every network output,
    P = resultant of N_zp,              yield checked at the 3 vertices of every triangle (exact).
Since derivatives are linear, the ZP part is never embedded during training (no float32 cancellation).
At initialisation (zero head) the model equals the ZP model exactly.

Usage (fine-tune probe; NLO_FVS=0 is the control with the same schedule but no FVS head):
  NLO_INIT=data/nlo_ckpt_100k.pt NLO_FVS=1 NLO_SMOKE_SECONDS=480 NLO_CKPT_OUT=data/nlo_fvs_ft.pt python nlo_train_fvs.py
  python nlo_train_fvs.py --eval data/nlo_fvs_ft.pt --split val
"""
import math
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import nlo_train as base  # noqa: E402
from nlo_core import AiryFit, RefMesh, derivs_from_stress, dissipation, plate_on_mesh, resultant  # noqa: E402
from nlo_fvs import FVS, to_res_fvs  # noqa: E402

NX, NY = base.NX, base.NY
DEVICE = base.DEVICE
USE_FVS = int(os.environ.get("NLO_FVS", 1))
INIT = os.environ.get("NLO_INIT", "")
LR = float(os.environ.get("NLO_LR", 2e-4))          # fine-tune learning rate (cosine to 0)
F_SCALE = float(os.environ.get("NLO_F_SCALE", 1.0))
TRAIN_SECONDS = float(os.environ.get("NLO_SMOKE_SECONDS", 480))
TRAIN_STEPS = int(os.environ.get("NLO_STEPS", 0))
CKPT_OUT = os.environ.get("NLO_CKPT_OUT", os.path.join(HERE, "data", "nlo_fvs_ckpt.pt"))
SEED = int(os.environ.get("NLO_SEED", 0))
TG = int(os.environ.get("NLO_TG", 0))       # 1: loss on the common thickness model t_g (data/nlo_train_tg_cache.npz)


class UNetF(base.UNet):
    """The nlo_train U-Net plus a zero-initialised FVS head (5 channels per cell)."""

    def __init__(self, cin):
        super().__init__(cin)
        self.head_f = nn.Conv2d(base.WIDTH, 5, 1)
        nn.init.zeros_(self.head_f.weight)
        nn.init.zeros_(self.head_f.bias)

    def forward(self, x):
        skips = [self.inp(x)]
        h = skips[0]
        for blk in self.down:
            h = blk(F.avg_pool2d(h, 2))
            skips.append(h)
        for blk, s in zip(self.up, reversed(skips[:-1])):
            h = F.interpolate(h, size=s.shape[-2:], mode="bilinear", align_corners=False)
            h = blk(torch.cat([h, s], 1))
        return self.head_v(h), self.head_s(h), self.head_f(h)


def fvs_dofs(out_f, thick, L, W):
    """out_f (B, 5, NX, NY) -> correction DOFs (V (B,NX+1,NY+1,3), H (B,NX,NY+1), Vt (B,NX+1,NY)), zero on free edges.
    Cell outputs are averaged onto vertices / edge midpoints (symmetric under mirroring)."""
    B = out_f.shape[0]
    tref = torch.quantile(thick, 0.9, dim=1)
    s = (F_SCALE * tref * (L / NX) * (W / NY))[:, None, None, None]
    o = out_f * s
    V = F.avg_pool2d(F.pad(o[:, 0:3], (1, 1, 1, 1), mode="replicate"), 2, stride=1)       # (B, 3, NX+1, NY+1)
    Hp = F.pad(o[:, 3], (1, 1), mode="replicate")                                           # (B, NX, NY+2)
    H = 0.5 * (Hp[..., :-1] + Hp[..., 1:])                                                   # (B, NX, NY+1)
    Vp = F.pad(o[:, 4:5], (0, 0, 1, 1), mode="replicate")[:, 0]                              # (B, NX+2, NY)
    Vt = 0.5 * (Vp[:, :-1] + Vp[:, 1:])                                                      # (B, NX+1, NY)
    mj = torch.ones(NY + 1, dtype=o.dtype, device=o.device)
    mj[0] = mj[-1] = 0
    V = V.permute(0, 2, 3, 1) * mj[None, None, :, None]
    H = H * mj[None, None, :]
    return V, H, Vt


def fields(mesh, fit, fvs, tri, out_s, out_f, thick, L, W):
    """Per-vertex admissible stresses (B, T, 3, 3) and the load P (B,)."""
    N0 = base.stress(mesh, fit, tri, out_s, thick, L, W)                # (B, T, 3) ZP, carries the load
    P = resultant(mesh, N0, W)
    Nv = N0[:, :, None, :].expand(-1, -1, 3, -1)
    if out_f is not None:
        V, H, Vt = fvs_dofs(out_f, thick, L, W)
        Nv = Nv + fvs.stress(V, H, Vt, L, W)
    return Nv, P


def yield_v(Nv, tv):
    Nxx, Nyy, Nxy = Nv[..., 0], Nv[..., 1], Nv[..., 2]
    return torch.sqrt(torch.clamp(Nxx ** 2 - Nxx * Nyy + Nyy ** 2 + 3 * Nxy ** 2, min=0.0) + 1e-30) / tv


def bounds(mesh, fit, fvs, tri, out_v, out_s, out_f, thick, L, W, soft=True, cap=None, tg=None):
    """cap, tg (B, N): optional t_g-safe nodal capacities (LB) and nodal t_g (UB); default: the nodal thickness."""
    unit = torch.tensor(mesh.unit, dtype=thick.dtype, device=thick.device)
    nodes = unit[None] * torch.stack([L, W], -1)[:, None, :]
    ub = dissipation(base.velocity(mesh, out_v), nodes, tri, thick if tg is None else tg)
    Nv, P = fields(mesh, fit, fvs, tri, out_s, out_f, thick, L, W)
    r = yield_v(Nv, (thick if cap is None else cap)[:, tri]).reshape(Nv.shape[0], -1)
    if soft:
        m = r.max(-1).values.detach()
        rmax = m * base.TAU * torch.logsumexp(r / (m[:, None] * base.TAU), -1)
    else:
        rmax = r.max(-1).values
    norm = torch.quantile(thick, 0.9, dim=1) * W
    return ub / norm, P / rmax / norm


def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    mesh = RefMesh(NX, NY)
    fit = AiryFit(mesh)
    fvs = FVS(mesh)
    pool = None
    if base.POOL_WORKERS > 0:
        pool, fit.solve_np = base.parallel_solver(fit, base.POOL_WORKERS)
    thick_np, L_np, W_np = base.load_train(mesh)
    thick_all = torch.tensor(thick_np, device=DEVICE)
    L_all, W_all = torch.tensor(L_np, device=DEVICE), torch.tensor(W_np, device=DEVICE)
    tri = torch.tensor(mesh.tri, device=DEVICE)
    if TG:
        tc = np.load(os.path.join(HERE, "data", "nlo_train_tg_cache.npz"))
        assert tc["tg"].shape[0] == len(L_np), "t_g cache does not match the training set (NLO_FAMILY?)"
        tg_all = torch.tensor(tc["tg"], device=DEVICE)                   # (n, 4 mirrors, N)
        tg_np = tc["tg"]
        cap_all = torch.tensor(np.maximum(tc["cap"], 0.3 * tg_np), device=DEVICE)   # loss proxy only: the additive
        # minorant goes negative at a few deep narrow pits (3e-7 of the nodes); certification never uses this cache
        print(f"NLO_TG: loss on t_g, cache {tuple(tg_all.shape)}", flush=True)
    model = UNetF(10).to(DEVICE)
    if INIT:
        missing, unexpected = model.load_state_dict(torch.load(INIT, map_location=DEVICE)["state"], strict=False)
        print(f"init from {INIT}; new parameters: {missing}", flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=base.WEIGHT_DECAY)
    rng = np.random.default_rng(SEED)
    t0 = time.time()
    step, last, hist = 0, t0, []
    while True:
        frac = step / TRAIN_STEPS if TRAIN_STEPS else (time.time() - t0) / TRAIN_SECONDS
        if frac >= 1:
            break
        for g in opt.param_groups:
            g["lr"] = LR * 0.5 * (1 + math.cos(math.pi * frac))
        ii = rng.integers(0, len(L_np), base.BATCH)
        fl = [(rng.random() < 0.5, rng.random() < 0.5) for _ in ii]
        th = torch.stack([base.flip(mesh, thick_all[i], fx, fy) for i, (fx, fy) in zip(ii, fl)])
        L, W = L_all[ii], W_all[ii]
        out_v, out_s, out_f = model(base.features(mesh, th, L, W))
        kk = [int(fx) + 2 * int(fy) for fx, fy in fl]                    # mirror index, nlo_refine.MIRRORS order
        cap = torch.stack([cap_all[i, k] for i, k in zip(ii, kk)]) if TG else None
        tgb = torch.stack([tg_all[i, k] for i, k in zip(ii, kk)]) if TG else None
        ub, lb = bounds(mesh, fit, fvs, tri, out_v, out_s, out_f if USE_FVS else None, th, L, W, cap=cap, tg=tgb)
        loss = (ub - lb).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        hist.append([ub.mean().item(), lb.mean().item()])
        step += 1
        if time.time() - last > 60:
            h = np.array(hist[-50:])
            print(f"step {step:6d}  {time.time() - t0:5.0f}s  ub {h[:, 0].mean():.4f}  lb {h[:, 1].mean():.4f}  "
                  f"gap {1 - h[:, 1].mean() / h[:, 0].mean():.4f}", flush=True)
            last = time.time()
    if pool is not None:
        pool.terminate()
    torch.save(dict(state=model.state_dict(), steps=step, seconds=time.time() - t0, fvs=USE_FVS), CKPT_OUT)
    print(f"done: {step} steps in {time.time() - t0:.0f}s, saved {CKPT_OUT}", flush=True)


# ----------------------------------------------------------------------------- prediction / evaluation
def load(path):
    ck = torch.load(path, map_location=DEVICE)
    model = UNetF(10).to(DEVICE)
    model.load_state_dict(ck["state"], strict=False)
    model.eval()
    model.use_fvs = bool(ck.get("fvs", 0))
    return model


_C = {}


@torch.no_grad()
def predict(model, plates, batch=16):
    """Certifiable results (per-vertex stresses) for the frozen verifier, float64."""
    if "mesh" not in _C:
        _C["mesh"] = RefMesh(NX, NY)
        _C["fit"] = AiryFit(_C["mesh"])
        _C["fvs"] = FVS(_C["mesh"])
    mesh, fit, fvs = _C["mesh"], _C["fit"], _C["fvs"]
    tri = torch.tensor(mesh.tri)
    out = []
    for k in range(0, len(plates), batch):
        chunk = plates[k:k + batch]
        pls = [plate_on_mesh(mesh, p["xs"], p["ys"], p["t"]) for p in chunk]
        th64 = torch.tensor(np.stack([p["thick"] for p in pls]), dtype=torch.float64)
        L64 = torch.tensor([p["L"] for p in pls], dtype=torch.float64)
        W64 = torch.tensor([p["W"] for p in pls], dtype=torch.float64)
        out_v, out_s, out_f = model(base.features(mesh, th64.to(DEVICE), L64.to(DEVICE), W64.to(DEVICE)))
        out_v, out_s, out_f = out_v.double().cpu(), out_s.double().cpu(), out_f.double().cpu()
        v = base.velocity(mesh, out_v)
        Nv, _ = fields(mesh, fit, fvs, tri, out_s, out_f if model.use_fvs else None, th64, L64, W64)
        for j, (p, pl) in enumerate(zip(chunk, pls)):
            out.append(to_res_fvs(mesh, pl, v[j].numpy(), Nv[j].numpy(), p["sigma_y"]))
    return out


def evaluate(path, split="val"):
    import csv
    from nlo_refine_eval import certify_lb, certify_ub
    model = load(path)
    d = np.load(os.path.join(HERE, "data", f"nlo_{split}.npz"), allow_pickle=True)
    refs = {r["name"]: r for r in csv.DictReader(open(os.path.join(HERE, "data", f"nlo_{split}_ref.csv")))}
    plates = [dict(xs=d["x_mm"][i], ys=d["y_mm"][i], t=d["thickness_mm"][i], sigma_y=float(d["sigma_y_MPa"][i]))
              for i in range(len(d["name"]))]
    t1 = time.time()
    res = predict(model, plates)
    dt = (time.time() - t1) / len(plates)
    gaps, ubx, lbd, ok = [], [], [], True
    for name, p, r in zip(d["name"], plates, res):
        ub, e1 = certify_ub(r, p["xs"], p["ys"], p["t"], p["sigma_y"])
        lb, e2 = certify_lb(r, p["xs"], p["ys"], p["t"], p["sigma_y"])
        ok &= not (e1 or e2)
        rf = refs[str(name)]
        gaps.append((ub - lb) / ub)
        ubx.append(ub / float(rf["UB_ref_N"]) - 1)
        lbd.append(1 - lb / float(rf["LB_ref_N"]))
    print(f"{os.path.basename(path)} fvs={model.use_fvs}: split {split} n={len(gaps)} certified={ok}  network gap "
          f"{100 * np.mean(gaps):.2f}%  (UB +{100 * np.mean(ubx):.2f}%, LB -{100 * np.mean(lbd):.2f}%)  {dt:.3f} s/plate",
          flush=True)
    return np.array(gaps)


if __name__ == "__main__":
    if len(sys.argv) > 2 and sys.argv[1] == "--eval":
        split = sys.argv[sys.argv.index("--split") + 1] if "--split" in sys.argv else "val"
        evaluate(sys.argv[2], split)
    else:
        main()
