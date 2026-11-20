"""A6a prototype: linear-stress LB correction in a window of the reference mesh (outside field fixed, scaled by lam)."""
import numpy as np
import scipy.sparse as sp
import clarabel

SQ3 = np.sqrt(3.0)
_H = {}


def edges(tri):
    if "he" not in _H:
        from limit_analysis import half_edges
        _H["he"] = half_edges(tri)
    return _H["he"]


def refine_lb_linear(N, nodes, tri, th, mesh, L, W, sy, theta=0.9, max_tri=3000):
    """N (T,3) constant per triangle at the yield surface. Returns per-vertex stresses S (T,3,3) at the yield
    surface (or None): window triangles (ratio >= theta) get linear stresses, the rest stays lam * N."""
    from limit_analysis import geometry
    T = len(tri)
    tmin = th[tri].min(1)
    f = np.sqrt(np.maximum(N[:, 0] ** 2 - N[:, 0] * N[:, 1] + N[:, 1] ** 2 + 3 * N[:, 2] ** 2, 0))
    r = f / (sy * tmin)
    hot = np.nonzero(r >= theta * r.max())[0]
    hot = hot[np.argsort(r[hot])[::-1]][:max_tri]
    win = np.zeros(T, bool)
    win[hot] = True
    wid = -np.ones(T, int)
    wid[hot] = np.arange(len(hot))
    nw = len(hot)
    s0 = sy * th.max()
    Nn = N / s0
    area, b, c = geometry(nodes, tri)
    el, la, lb, ga, gb, (h1, h2), hb = edges(tri)
    d = nodes[gb] - nodes[ga]
    ln = np.linalg.norm(d, axis=1)
    nrm = np.stack([d[:, 1], -d[:, 0]], 1) / ln[:, None]
    nv = 1 + 9 * nw
    vid = lambda w, k, comp: 1 + 9 * w + 3 * k + comp
    R, C, V, rhs = [], [], [], []
    nr = [0]

    def row(cols, vals):
        R.append(np.full(len(cols), nr[0])); C.append(np.asarray(cols)); V.append(np.asarray(vals, float))
        nr[0] += 1

    # equilibrium in window elements (rows built vectorised)
    rows_eq = []
    e = hot
    w = wid[e]
    for comp_pair in ((0, 2), (2, 1)):                                  # (Nxx,Nxy) with (b,c); (Nxy,Nyy) with (b,c)
        cols = np.concatenate([vid(w, k, comp_pair[0]) for k in range(3)] + [vid(w, k, comp_pair[1]) for k in range(3)])
        vals = np.concatenate([b[e, k] for k in range(3)] + [c[e, k] for k in range(3)])
        rr = nr[0] + np.tile(np.arange(nw), 6)
        R.append(rr); C.append(cols); V.append(vals)
        nr[0] += nw

    def trac_terms(eidx, k, n, comp):
        """columns/values of traction component comp (0=x,1=y) of element eidx at local vertex k, normal n."""
        if comp == 0:
            return [vid(wid[eidx], k, 0), vid(wid[eidx], k, 2)], [n[0], n[1]]
        return [vid(wid[eidx], k, 1), vid(wid[eidx], k, 2)], [n[1], n[0]]

    def fixed_trac(eidx, n, comp):
        return (Nn[eidx, 0] * n[0] + Nn[eidx, 2] * n[1]) if comp == 0 else (Nn[eidx, 2] * n[0] + Nn[eidx, 1] * n[1])

    # interior edges touching the window: continuity at both endpoints
    touch = win[el[h1]] | win[el[h2]]
    for i in np.nonzero(touch)[0]:
        e1, e2 = el[h1[i]], el[h2[i]]
        n = nrm[h1[i]]
        for node, k1 in ((ga[h1[i]], la[h1[i]]), (gb[h1[i]], lb[h1[i]])):
            k2 = la[h2[i]] if ga[h2[i]] == node else lb[h2[i]]
            for comp in (0, 1):
                cols, vals = [], []
                for ee, kk, sgn in ((e1, k1, 1.0), (e2, k2, -1.0)):
                    if win[ee]:
                        cc, vv = trac_terms(ee, kk, n, comp)
                        cols += cc; vals += [sgn * x for x in vv]
                    else:
                        cols.append(0); vals.append(sgn * fixed_trac(ee, n, comp))
                row(cols, vals)
    # free boundary edges (y = 0, W) of window elements
    mid = 0.5 * (nodes[ga[hb]] + nodes[gb[hb]])
    tol = 1e-7 * max(L, W)
    free = (np.abs(mid[:, 1]) < tol) | (np.abs(mid[:, 1] - W) < tol)
    right = np.abs(mid[:, 0] - L) < tol
    for i in np.nonzero(free & win[el[hb]])[0]:
        h = hb[i]
        for k in (la[h], lb[h]):
            for comp in (0, 1):
                cols, vals = trac_terms(el[h], k, nrm[h], comp)
                row(cols, vals)
    neq = nr[0]
    A_eq = sp.csr_matrix((np.concatenate(V), (np.concatenate(R), np.concatenate(C))), shape=(neq, nv))
    # lam <= 1 / max outside ratio
    out = ~win
    lam_max = 1.0 / r[out].max() if out.any() else 1e6
    A_lam = sp.csr_matrix(([1.0], ([0], [0])), shape=(1, nv))
    # yield cones at the 3 vertices of each window element
    wv = np.repeat(np.arange(nw), 3)
    kv = np.tile(np.arange(3), nw)
    r0 = 4 * np.arange(3 * nw)
    A_soc = sp.coo_matrix((np.concatenate([-np.ones(3 * nw), 0.5 * np.ones(3 * nw), -0.5 * SQ3 * np.ones(3 * nw), -SQ3 * np.ones(3 * nw)]),
                           (np.concatenate([r0 + 1, r0 + 1, r0 + 2, r0 + 3]),
                            np.concatenate([vid(wv, kv, 0), vid(wv, kv, 1), vid(wv, kv, 1), vid(wv, kv, 2)]))), shape=(12 * nw, nv))
    b_soc = np.zeros(12 * nw)
    b_soc[r0] = th[tri[hot[wv], kv]] / th.max()
    # objective: grip resultant = lam * (fixed right-edge part) + window right-edge part
    g = np.zeros(nv)
    hr = hb[right]
    for h in hr:
        e_ = el[h]
        if win[e_]:
            g[vid(wid[e_], la[h], 0)] += 0.5 * ln[h]
            g[vid(wid[e_], lb[h], 0)] += 0.5 * ln[h]
        else:
            g[0] += Nn[e_, 0] * ln[h]
    A = sp.vstack([A_eq, A_lam, A_soc]).tocsc()
    bb = np.concatenate([np.zeros(neq), [lam_max], b_soc])
    cones = [clarabel.ZeroConeT(neq), clarabel.NonnegativeConeT(1)] + [clarabel.SecondOrderConeT(4)] * (3 * nw)
    st = clarabel.DefaultSettings()
    st.verbose = False
    sol = clarabel.DefaultSolver(sp.csc_matrix((nv, nv)), -g, A, bb, cones, st).solve()
    if str(sol.status) not in ("Solved", "AlmostSolved"):
        return None
    x = np.asarray(sol.x)
    S = np.repeat((x[0] * Nn)[:, None, :], 3, 1)
    S[hot] = x[1:].reshape(nw, 3, 3)
    S = S * s0
    fv = np.sqrt(np.maximum(S[..., 0] ** 2 - S[..., 0] * S[..., 1] + S[..., 1] ** 2 + 3 * S[..., 2] ** 2, 0))
    return S / (fv / (sy * th[tri])).max()
