"""Measured thickness scans: reading the two face scans of a plate, conversion to a thickness grid, and a
linear-elastic plane-stress membrane solver (T3 elements) used to compare synthetic and scanned plates
(membrane-equivalent thickness, elastic stress concentration). The measured scans are not part of this
repository; they are available from the corresponding author on reasonable request."""
import argparse
import csv
import glob
import os

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

E_REF = 100.0
L_HAT = 5.0
DY_TARGET = 2.0      # mm, target in-plane resolution of the thickness grid
EDGE_ZONE = 0.10     # fraction of length at each end excluded from the SCF (clamp / load singularities)


# ----------------------------------------------------------------------------- I/O
def read_table(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(l for l in f if not l.startswith("#")))


def fnum(s):
    try:
        return float(s)
    except (TypeError, ValueError):
        return np.nan


def load_surface(path):
    with open(path, encoding="utf-8") as f:
        lines = [l for l in f if not l.startswith("#")]
    header = lines[0].strip().split(",")
    a = np.loadtxt(lines[1:], delimiter=",", ndmin=2)
    return {h: a[:, i] for i, h in enumerate(header)}


def surface_to_grid(s):
    """Scattered full-grid CSV -> (xs, ys, dict of (nx, ny) fields) at ~DY_TARGET resolution."""
    xs, ys = np.unique(s["x_mm"]), np.unique(s["y_mm"])
    ix, iy = np.searchsorted(xs, s["x_mm"]), np.searchsorted(ys, s["y_mm"])
    keys = ("thickness_mm", "z_top_mm", "z_bottom_mm")
    g = {}
    for k in keys:
        a = np.full((len(xs), len(ys)), np.nan)
        a[ix, iy] = s[k]
        g[k] = a
    if np.median(np.diff(ys)) < 0.5 * DY_TARGET:
        # Amarube: 0.1 mm across the width -> window-average onto a 2 mm node grid
        nb = max(int(round(ys.max() / DY_TARGET)), 1)
        ynew = np.linspace(0.0, ys.max(), nb + 1)
        for k in keys:
            g[k] = np.stack([np.nanmean(g[k][:, np.abs(ys - yn) <= 0.5 * DY_TARGET + 1e-9], axis=1)
                             for yn in ynew], axis=1)
        ys = ynew
    return xs, ys, g


# ----------------------------------------------------------------------------- mesh
def grid_mesh(xs, ys):
    nx, ny = len(xs), len(ys)
    X, Y = np.meshgrid(xs, ys, indexing="ij")
    xy = np.stack([X.ravel(), Y.ravel()], axis=1)
    idx = np.arange(nx * ny).reshape(nx, ny)
    a, b = idx[:-1, :-1], idx[1:, :-1]
    c, d = idx[1:, 1:], idx[:-1, 1:]
    even = ((np.add.outer(np.arange(nx - 1), np.arange(ny - 1))) % 2 == 0)
    # alternate the diagonal so the mesh has no preferred direction; all triangles CCW
    t1 = np.where(even[..., None], np.stack([a, b, c], -1), np.stack([a, b, d], -1))
    t2 = np.where(even[..., None], np.stack([a, c, d], -1), np.stack([b, c, d], -1))
    tri = np.concatenate([t1.reshape(-1, 3), t2.reshape(-1, 3)])
    left_nodes = idx[0, :].copy()
    right_edges = np.stack([idx[-1, :-1], idx[-1, 1:]], axis=1)
    return xy, tri, left_nodes, right_edges


# ----------------------------------------------------------------------------- linear-elastic membrane FEM (T3)
def b_matrices(xy, tri):
    p = xy[tri]
    x1, y1, x2, y2, x3, y3 = p[:, 0, 0], p[:, 0, 1], p[:, 1, 0], p[:, 1, 1], p[:, 2, 0], p[:, 2, 1]
    det = (x2 - x1) * (y3 - y1) - (y2 - y1) * (x3 - x1)
    A = 0.5 * np.abs(det)
    b = np.stack([y2 - y3, y3 - y1, y1 - y2], 1)
    c = np.stack([x3 - x2, x1 - x3, x2 - x1], 1)
    B = np.zeros((len(tri), 3, 6))
    B[:, 0, 0::2] = b
    B[:, 1, 1::2] = c
    B[:, 2, 0::2] = c
    B[:, 2, 1::2] = b
    return B / (2.0 * A)[:, None, None], A


def d_plane_stress(E, nu):
    E, nu = np.broadcast_arrays(np.asarray(E, float), np.asarray(nu, float))
    D = np.zeros(E.shape + (3, 3))
    f = E / (1.0 - nu ** 2)
    D[..., 0, 0] = D[..., 1, 1] = f
    D[..., 0, 1] = D[..., 1, 0] = f * nu
    D[..., 2, 2] = f * (1.0 - nu) / 2.0
    return D


def solve_membrane(xy, tri, left_nodes, right_edges, E_node, nu_node):
    n = len(xy)
    B, A = b_matrices(xy, tri)
    D = d_plane_stress(E_node[tri].mean(1), nu_node[tri].mean(1))
    Ke = A[:, None, None] * np.einsum("tki,tkl,tlj->tij", B, D, B)
    dofs = np.stack([2 * tri, 2 * tri + 1], axis=2).reshape(-1, 6)
    rows = np.repeat(dofs, 6, axis=1).ravel()
    cols = np.tile(dofs, (1, 6)).ravel()
    K = sp.coo_matrix((Ke.ravel(), (rows, cols)), shape=(2 * n, 2 * n)).tocsr()
    f = np.zeros(2 * n)
    L = np.linalg.norm(xy[right_edges[:, 1]] - xy[right_edges[:, 0]], axis=1)
    np.add.at(f, 2 * right_edges[:, 0], 0.5 * L)   # t_x = 1, linear shape functions
    np.add.at(f, 2 * right_edges[:, 1], 0.5 * L)
    fixed = np.concatenate([2 * left_nodes, 2 * left_nodes + 1])
    free = np.setdiff1d(np.arange(2 * n), fixed)
    U = np.zeros(2 * n)
    U[free] = spla.spsolve(K[free][:, free].tocsc(), f[free])
    return U.reshape(n, 2), f @ U        # displacement, compliance (= 2 x strain energy)


def von_mises_hat(xy, tri, uv, nu):
    """Element von Mises stress in units of q/t0 (i.e. SCF w.r.t. the gross nominal stress)."""
    B, _ = b_matrices(xy, tri)
    ue = uv[tri].reshape(-1, 6)
    eps = np.einsum("tij,tj->ti", B, ue)
    s = eps @ d_plane_stress(E_REF, nu).T
    return np.sqrt(s[:, 0] ** 2 - s[:, 0] * s[:, 1] + s[:, 1] ** 2 + 3 * s[:, 2] ** 2)


def corr_length(g, d):
    """1/e correlation length (mm) of a 2-D field along axis 0, averaged over the other axis."""
    a = g - g.mean()
    n = a.shape[0]
    F = np.fft.rfft(a, n=2 * n, axis=0)
    acf = np.fft.irfft(F * np.conj(F), axis=0)[:n].mean(axis=1)
    acf /= acf[0]
    k = np.argmax(acf < np.exp(-1.0))
    return k * d if k > 0 else np.nan


# ----------------------------------------------------------------------------- main
