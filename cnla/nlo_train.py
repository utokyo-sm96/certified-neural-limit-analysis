#!/usr/bin/env python3
"""EDITABLE in loop nlo-1106: neural limit operator (NLO) - model, label-free training, prediction.

The frozen evaluator nlo_eval.py runs `python nlo_train.py` (it must finish inside the evaluator's
timeout and write CKPT), then imports this module and calls predict(load(CKPT), plates). Every
prediction is certified independently by the step-1 verifier, so nothing here can fake a bound.

One U-Net reads the thickness field and outputs, per cell of the reference mesh,
  * a velocity field: x-velocity = per-row cumulative softmax (exactly 0 at x = 0 and 1 at x = L),
    y-velocity with a bubble that vanishes at both grips, free centre-node corrections;
  * a target stress per triangle, N* = t (1 + s_xx, s_yy, s_xy). nlo_core.AiryFit maps N* to the closest
    exactly equilibrated, traction-free Airy field (a fixed linear least-squares layer), so every
    output is statically admissible; the zero-initialised head starts from N* = t (1, 0, 0).
Training is label-free; the loss is the smoothed certified gap itself,
    loss = UB(v) / (t_ref W) - LB(N) / (t_ref W),   LB = grip resultant / softmax_tau(yield ratio).
No FE solution is ever used.
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
from nlo_core import (AiryFit, RefMesh, derivs_from_stress, dissipation, plate_on_mesh,  # noqa: E402
                      resultant, to_res, yield_ratio)

NX, NY = 256, 80              # reference mesh (cells); max(hx, hy) must stay <= 2 mm (evaluator gate)
TRAIN_SECONDS = float(os.environ.get("NLO_SMOKE_SECONDS", 480))  # training budget; evaluator kills at 720 s
TRAIN_STEPS = int(os.environ.get("NLO_STEPS", 0))   # final long run only: fixed step count instead of time
CKPT_OUT = os.environ.get("NLO_CKPT_OUT", "")       # final long run only: separate checkpoint path
SEED = 0
BATCH = 4
LR = 1e-3
WEIGHT_DECAY = 1e-4
WIDTH = 32                    # U-Net base channels
TAU = 0.003                   # relative temperature of the soft maximum yield ratio
VY_SCALE = 0.1
S_SCALE = 1.0                 # scale of the target-stress correction
CKPT = os.path.join(HERE, "data", "nlo_ckpt.pt")
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"


# ----------------------------------------------------------------------------- model
class Block(nn.Module):
    def __init__(self, cin, cout):
        super().__init__()
        self.c1 = nn.Conv2d(cin, cout, 3, padding=1)
        self.n1 = nn.GroupNorm(8, cout)
        self.c2 = nn.Conv2d(cout, cout, 3, padding=1)
        self.n2 = nn.GroupNorm(8, cout)
        self.skip = nn.Conv2d(cin, cout, 1) if cin != cout else nn.Identity()

    def forward(self, x):
        h = F.gelu(self.n1(self.c1(x)))
        h = self.n2(self.c2(h))
        return F.gelu(h + self.skip(x))


class UNet(nn.Module):
    def __init__(self, cin, w=WIDTH):
        super().__init__()
        ch = [w, 2 * w, 4 * w, 8 * w, 8 * w]                         # 4 levels: 256x80 -> 16x5 (whole-plate context)
        self.inp = Block(cin, ch[0])
        self.down = nn.ModuleList([Block(ch[i], ch[i + 1]) for i in range(4)])
        self.up = nn.ModuleList([Block(ch[i + 1] + ch[i], ch[i]) for i in reversed(range(4))])
        self.head_v = nn.Conv2d(ch[0], 4, 1)             # velocity logit, vy, centre dvx, centre dvy
        self.head_s = nn.Conv2d(ch[0], 12, 1)            # target stress (sxx, syy, sxy) on the 4 triangles
        for h in (self.head_v, self.head_s):             # start from v = (x/L, 0) and N* = t (1, 0, 0)
            nn.init.zeros_(h.weight)
            nn.init.zeros_(h.bias)

    def forward(self, x):
        skips = [self.inp(x)]
        h = skips[0]
        for blk in self.down:
            h = blk(F.avg_pool2d(h, 2))
            skips.append(h)
        for blk, s in zip(self.up, reversed(skips[:-1])):
            h = F.interpolate(h, size=s.shape[-2:], mode="bilinear", align_corners=False)
            h = blk(torch.cat([h, s], 1))
        return self.head_v(h), self.head_s(h)


# ----------------------------------------------------------------------------- fields from network outputs
def features(mesh, thick, L, W):
    """thick (B, Nn) node thickness, L, W (B,) -> cell input (B, 10, NX, NY)."""
    B = thick.shape[0]
    G = mesh.G
    tg = thick[:, :G].reshape(B, NX + 1, NY + 1)
    tc = thick[:, G:].reshape(B, NX, NY)
    tref = torch.quantile(thick, 0.9, dim=1)[:, None, None]
    corners = [tg[:, :-1, :-1], tg[:, 1:, :-1], tg[:, 1:, 1:], tg[:, :-1, 1:], tc]
    xi = ((torch.arange(NX, device=thick.device) + 0.5) / NX)[None, :, None].expand(B, NX, NY)
    eta = ((torch.arange(NY, device=thick.device) + 0.5) / NY)[None, None, :].expand(B, NX, NY)
    const = [(L / 250)[:, None, None].expand(B, NX, NY), (W / 80)[:, None, None].expand(B, NX, NY),
             (tref / 12).expand(B, NX, NY)]
    return torch.stack([c / tref for c in corners] + [xi, eta] + const, 1).float()


def velocity(mesh, out_v):
    """out_v (B, 4, NX, NY) -> nodal velocities (B, Nn, 2), exactly (0,0) at x=0 and (1,0) at x=L."""
    B = out_v.shape[0]
    lg = F.pad(out_v[:, 0:1], (1, 1, 0, 0), mode="replicate")[:, 0]      # (B, NX, NY+2)
    lg = 0.5 * (lg[:, :, :-1] + lg[:, :, 1:])                            # node rows: (B, NX, NY+1)
    cs = torch.cumsum(torch.softmax(lg, dim=1), dim=1)
    vx = torch.cat([torch.zeros_like(cs[:, :1]), cs / cs[:, -1:]], 1)    # (B, NX+1, NY+1)
    vy = F.avg_pool2d(F.pad(out_v[:, 1:2], (1, 1, 1, 1), mode="replicate"), 2, stride=1)[:, 0]
    xi = (torch.arange(NX + 1, device=out_v.device, dtype=out_v.dtype) / NX)[None, :, None]
    vy = VY_SCALE * 4 * xi * (1 - xi) * vy                              # exactly 0 at both grips
    grid = torch.stack([vx, vy], -1)                                     # (B, NX+1, NY+1, 2)
    centre = 0.25 * (grid[:, :-1, :-1] + grid[:, 1:, :-1] + grid[:, 1:, 1:] + grid[:, :-1, 1:])
    centre = centre + VY_SCALE * out_v[:, 2:4].permute(0, 2, 3, 1)
    return torch.cat([grid.reshape(B, -1, 2), centre.reshape(B, -1, 2)], 1)


def stress(mesh, fit, tri, out_s, thick, L, W):
    """Target N* = t_mean (1 + sxx, syy, sxy) per triangle -> closest admissible Airy field (B, T, 3)."""
    B = out_s.shape[0]
    s = S_SCALE * out_s.permute(0, 2, 3, 1).reshape(B, -1, 3)            # triangle order 4*(i*NY+j)+k
    s = s + torch.tensor([1.0, 0.0, 0.0], dtype=s.dtype, device=s.device)
    target = thick[:, tri].mean(-1)[..., None] * s
    c = fit.fit(derivs_from_stress(target, L, W, mesh))
    return mesh.stress(c, L, W)


def bounds(mesh, fit, tri, out_v, out_s, thick, L, W, soft=True):
    """Normalised (UB, LB) per plate; LB uses a soft maximum of the yield ratio when soft=True."""
    unit = torch.tensor(mesh.unit, dtype=thick.dtype, device=thick.device)
    nodes = unit[None] * torch.stack([L, W], -1)[:, None, :]
    ub = dissipation(velocity(mesh, out_v), nodes, tri, thick)
    N = stress(mesh, fit, tri, out_s, thick, L, W)
    r = yield_ratio(N, thick[:, tri].min(-1).values)
    if soft:
        m = r.max(-1).values.detach()
        rmax = m * TAU * torch.logsumexp(r / (m[:, None] * TAU), -1)
    else:
        rmax = r.max(-1).values
    lb = resultant(mesh, N, W) / rmax
    norm = torch.quantile(thick, 0.9, dim=1) * W
    return ub / norm, lb / norm


# ----------------------------------------------------------------------------- data
FAMILY = os.environ.get("NLO_FAMILY", "")      # leave-one-bridge-out: train only on plates of this generator family


def _family_filter(out):
    """Keep only the training plates of generator family FAMILY (all plates if FAMILY is empty)."""
    if not FAMILY:
        return out
    d = np.load(os.path.join(HERE, "data", "synthetic_plates.npz"), allow_pickle=True)
    idx = np.load(os.path.join(HERE, "data", "nlo_split.npz"))["idx_train"]
    keep = np.asarray(d["family"])[idx] == FAMILY
    print(f"NLO_FAMILY={FAMILY}: {keep.sum()} of {len(keep)} training plates", flush=True)
    return tuple(a[keep] for a in out)


def load_train(mesh):
    """Node thickness of the training plates on the reference mesh (cached: it is pure preprocessing)."""
    cache = os.path.join(HERE, "data", f"nlo_train_cache_{NX}x{NY}.npz")
    if os.path.exists(cache):
        c = np.load(cache)
        return _family_filter((c["thick"], c["L"], c["W"]))
    d = np.load(os.path.join(HERE, "data", "synthetic_plates.npz"), allow_pickle=True)
    idx = np.load(os.path.join(HERE, "data", "nlo_split.npz"))["idx_train"]
    thick, L, W = [], [], []
    for i in idx:
        p = plate_on_mesh(mesh, d["x_mm"][i], d["y_mm"][i], d["thickness_mm"][i])
        thick.append(p["thick"].astype(np.float32))
        L.append(p["L"])
        W.append(p["W"])
    out = np.stack(thick), np.array(L, np.float32), np.array(W, np.float32)
    np.savez(cache, thick=out[0], L=out[1], W=out[2])
    return _family_filter(out)


def flip(mesh, thick, fx, fy):
    """Mirror node thickness of one plate in x and/or y (the union-jack mesh is symmetric)."""
    G = mesh.G
    tg = thick[:G].reshape(NX + 1, NY + 1)
    tc = thick[G:].reshape(NX, NY)
    dims = [d for d, f in ((0, fx), (1, fy)) if f]
    if dims:
        tg, tc = torch.flip(tg, dims), torch.flip(tc, dims)
    return torch.cat([tg.reshape(-1), tc.reshape(-1)])


# ----------------------------------------------------------------------------- parallel Airy solves (opt-in)
POOL_WORKERS = int(os.environ.get("NLO_POOL_WORKERS", 0))   # e.g. 4 for long runs: +16-24 % steps (loop nlo-net-1107, N1)
_CB = None


def _pool_init(cb):
    global _CB
    _CB = cb


def _pool_band(r):
    from scipy.linalg import cho_solve_banded
    return cho_solve_banded((_CB, False), r)


def parallel_solver(fit, workers):
    """AiryFit.solve_np with the banded solves spread over a forked process pool (scipy keeps the GIL, so
    threads do not help). Must be created before CUDA is initialised. Same linear map, so same results."""
    import multiprocessing as mp
    pool = mp.get_context("fork").Pool(workers, initializer=_pool_init, initargs=(fit.cb,))

    def solve_np(r):
        ri, rt = r[:-4], r[-4:]
        y = np.concatenate(pool.map(_pool_band, [ri[:, k:k + 1] for k in range(ri.shape[1])]), 1)
        zt = fit.Sinv @ (rt - fit.Kit.T @ y)
        return np.concatenate([y - fit.X @ zt, zt])
    return pool, solve_np


# ----------------------------------------------------------------------------- train / load / predict
def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    mesh = RefMesh(NX, NY)
    fit = AiryFit(mesh)
    pool = None
    if POOL_WORKERS > 0:                                                 # before CUDA is initialised
        pool, fit.solve_np = parallel_solver(fit, POOL_WORKERS)
    t_load = time.time()
    thick_np, L_np, W_np = load_train(mesh)
    print(f"loaded {len(L_np)} training plates in {time.time() - t_load:.1f}s", flush=True)
    thick_all = torch.tensor(thick_np, device=DEVICE)
    L_all, W_all = torch.tensor(L_np, device=DEVICE), torch.tensor(W_np, device=DEVICE)
    tri = torch.tensor(mesh.tri, device=DEVICE)
    model = UNet(10).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    print(f"params {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M", flush=True)
    rng = np.random.default_rng(SEED)
    t0 = time.time()
    step, last = 0, t0
    hist = []
    while True:
        frac = step / TRAIN_STEPS if TRAIN_STEPS else (time.time() - t0) / TRAIN_SECONDS
        if frac >= 1:
            break
        for g in opt.param_groups:
            g["lr"] = LR * 0.5 * (1 + math.cos(math.pi * frac))
        ii = rng.integers(0, len(L_np), BATCH)
        th = torch.stack([flip(mesh, thick_all[i], rng.random() < 0.5, rng.random() < 0.5) for i in ii])
        L, W = L_all[ii], W_all[ii]
        out_v, out_s = model(features(mesh, th, L, W))
        ub, lb = bounds(mesh, fit, tri, out_v, out_s, th, L, W)
        loss = (ub - lb).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        hist.append([ub.mean().item(), lb.mean().item()])
        step += 1
        if time.time() - last > (300 if TRAIN_STEPS else 30):
            h = np.array(hist[-50:])
            print(f"step {step:6d}  {time.time() - t0:5.0f}s  ub {h[:, 0].mean():.4f}  lb {h[:, 1].mean():.4f}  "
                  f"gap {1 - h[:, 1].mean() / h[:, 0].mean():.4f}", flush=True)
            last = time.time()
    if pool is not None:
        pool.terminate()
    out = CKPT_OUT or CKPT
    torch.save(dict(state=model.state_dict(), steps=step, seconds=time.time() - t0), out)
    print(f"done: {step} steps in {time.time() - t0:.0f}s, saved {out}", flush=True)


def load(path=CKPT):
    model = UNet(10).to(DEVICE)
    model.load_state_dict(torch.load(path, map_location=DEVICE)["state"])
    model.eval()
    return model


_FIT = {}


@torch.no_grad()
def predict(model, plates, batch=16):
    """plates: list of dict(xs, ys, t, sigma_y). Returns one result dict per plate for the frozen verifier."""
    if "mesh" not in _FIT:
        _FIT["mesh"] = RefMesh(NX, NY)
        _FIT["fit"] = AiryFit(_FIT["mesh"])
    mesh, fit = _FIT["mesh"], _FIT["fit"]
    tri = torch.tensor(mesh.tri)
    out = []
    for k in range(0, len(plates), batch):
        chunk = plates[k:k + batch]
        pls = [plate_on_mesh(mesh, p["xs"], p["ys"], p["t"]) for p in chunk]
        th64 = torch.tensor(np.stack([p["thick"] for p in pls]), dtype=torch.float64)
        L64 = torch.tensor([p["L"] for p in pls], dtype=torch.float64)
        W64 = torch.tensor([p["W"] for p in pls], dtype=torch.float64)
        out_v, out_s = model(features(mesh, th64.to(DEVICE), L64.to(DEVICE), W64.to(DEVICE)))
        out_v, out_s = out_v.double().cpu(), out_s.double().cpu()     # certify in float64
        v = velocity(mesh, out_v)
        N = stress(mesh, fit, tri, out_s, th64, L64, W64)
        for j, (p, pl) in enumerate(zip(chunk, pls)):
            out.append(to_res(mesh, pl, v[j].numpy(), N[j].numpy(), p["sigma_y"]))
    return out


if __name__ == "__main__":
    main()
