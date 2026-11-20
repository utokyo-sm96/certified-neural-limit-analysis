"""Probe: geometric nested-dissection ordering of the FVS normal matrix K^T K (LB ADMM solve of nlo_global.FVSArm).
The free FVS unknowns live on the structured 256 x 80 grid (vertex DOFs at (i, j), H-edge DOFs at (i + 1/2, j), vertical
edge DOFs at (i, j + 1/2), 3 global top parameters). K^T K couples unknowns of the same cell, so the unknowns on a grid
line x = c (or y = c) separate the two sides. Recursive bisection perpendicular to the longer side; separators last.
Usage: python loops/scratch/nd_order.py"""
import sys
import time

import numpy as np

sys.path.insert(0, "cnla")


def nd_order(mesh, leaf=256):
    from nlo_fvs import dof_layout
    NX, NY = mesh.NX, mesh.NY
    Vi, Hi, Vti, n, nfree = dof_layout(mesh)
    X, Y = np.full(nfree, np.nan), np.full(nfree, np.nan)
    ii, jj = np.meshgrid(np.arange(NX + 1), np.arange(NY + 1), indexing="ij")
    for k in range(3):
        m = Vi[..., k] >= 0
        X[Vi[..., k][m]], Y[Vi[..., k][m]] = ii[m], jj[m]
    hi, hj = np.meshgrid(np.arange(NX), np.arange(NY + 1), indexing="ij")
    m = Hi >= 0
    X[Hi[m]], Y[Hi[m]] = hi[m] + 0.5, hj[m]
    vi, vj = np.meshgrid(np.arange(NX + 1), np.arange(NY), indexing="ij")
    X[Vti.ravel()], Y[Vti.ravel()] = vi.ravel(), vj.ravel() + 0.5
    assert not np.isnan(X[:n]).any() and not np.isnan(Y[:n]).any()
    out = []

    def rec(idx):
        x, y = X[idx], Y[idx]
        ex, ey = x.max() - x.min(), y.max() - y.min()
        if len(idx) <= leaf or (ex < 2 and ey < 2):
            out.append(idx)
            return
        c_ax, ax = (x, 0) if ex >= ey else (y, 1)
        lo, hi_ = np.floor(c_ax.min()) + 1, np.ceil(c_ax.max()) - 1
        c = float(np.clip(np.round(np.median(c_ax)), lo, hi_))
        rec(idx[c_ax < c])
        rec(idx[c_ax > c])
        out.append(idx[c_ax == c])

    rec(np.arange(n))
    out.append(np.arange(n, nfree))
    p = np.concatenate(out)
    assert len(p) == nfree and len(np.unique(p)) == nfree
    return p


if __name__ == "__main__":
    import scipy.sparse.linalg as spla
    from nlo_core import RefMesh
    from nlo_fvs import FVS, fvs_stress_matrix
    mesh = RefMesh(256, 80)
    K = fvs_stress_matrix(mesh, FVS(mesh), mesh.NX, mesh.NY).tocsr()
    A = (K.T @ K).tocsc()
    t0 = time.time()
    p = nd_order(mesh)
    print(f"ordering {time.time() - t0:.2f} s")
    Ap = A[p][:, p].tocsc()
    rng = np.random.default_rng(0)
    b = rng.standard_normal(A.shape[0])
    for leaf in (256,):
        t0 = time.time()
        lu = spla.splu(Ap, permc_spec="NATURAL", diag_pivot_thresh=0.0, options=dict(SymmetricMode=True))
        ts = time.time() - t0
        lu.solve(b[p])
        t0 = time.time()
        for _ in range(10):
            y = lu.solve(b[p])
        x = np.empty_like(y)
        x[p] = y
        print(f"ND: factor {ts:.1f} s, nnz L+U {lu.L.nnz + lu.U.nnz:,}, solve {(time.time() - t0) / 10 * 1e3:.1f} ms, "
              f"rel resid {np.linalg.norm(A @ x - b) / np.linalg.norm(b):.1e}")
