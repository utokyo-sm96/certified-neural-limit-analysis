"""Numerical study of S_3^1 (C1 piecewise cubics) on the union-jack (criss-cross) mesh."""
import itertools, numpy as np
from collections import defaultdict

MON = [(i, j) for d in range(4) for i in range(d, -1, -1) for j in [d - i]]   # 10 monomials x^i y^j, deg <= 3

def mon(x, y): return np.array([x ** i * y ** j for i, j in MON])
def dmx(x, y): return np.array([i * x ** (i - 1) * y ** j if i else 0.0 for i, j in MON])
def dmy(x, y): return np.array([j * x ** i * y ** (j - 1) if j else 0.0 for i, j in MON])

def cell_tris(i, j):
    a, b, c, d, m = (i, j), (i + 1, j), (i + 1, j + 1), (i, j + 1), (i + .5, j + .5)
    return [(a, b, m), (b, c, m), (c, d, m), (d, a, m)]

def patch(P, Q=None):
    Q = Q or P
    T = [((i, j, k), np.array(tv, float)) for i in range(P) for j in range(Q) for k, tv in enumerate(cell_tris(i, j))]
    E = defaultdict(list)
    for t, (key, V) in enumerate(T):
        for s in range(3):
            p, q = tuple(V[s]), tuple(V[(s + 1) % 3]); E[tuple(sorted([p, q]))].append(t)
    nT = len(T); rows = []
    for (p, q), ts in E.items():
        if len(ts) != 2: continue
        t1, t2 = ts; p, q = np.array(p), np.array(q)
        for s in (0, 1 / 3, 2 / 3, 1):                                  # C0: cubic along edge, 4 points
            pt = p + s * (q - p); r = np.zeros(10 * nT); r[10 * t1:10 * t1 + 10] = mon(*pt); r[10 * t2:10 * t2 + 10] = -mon(*pt); rows.append(r)
        for s in (0, 0.5, 1):                                            # C1: gradient (quadratic along edge), 3 points
            pt = p + s * (q - p)
            for f in (dmx, dmy):
                r = np.zeros(10 * nT); r[10 * t1:10 * t1 + 10] = f(*pt); r[10 * t2:10 * t2 + 10] = -f(*pt); rows.append(r)
    return T, np.array(rows)

def null(A, tol=1e-9):
    U, s, Vt = np.linalg.svd(A); r = (s > tol * s[0]).sum(); return Vt[r:]

for P in (2, 3, 4, 5):
    T, A = patch(P); N = null(A)
    print(f"patch {P}x{P}: dim S_3^1 = {len(N)}  (4P^2 = {4*P*P}, (P+1)^2 = {(P+1)**2})", flush=True)

# ---- minimal-support generators: null-space elements supported in a k x k block of a (k+2)x(k+2) patch
def block_space(k):
    P = k + 2
    T, A = patch(P)
    Z = []
    for t, (key, V) in enumerate(T):
        i, j, _ = key
        if not (1 <= i <= k and 1 <= j <= k):
            for cc in range(10):
                r = np.zeros(10 * len(T)); r[10 * t + cc] = 1; Z.append(r)
    N = null(np.vstack([A, np.array(Z)]))
    return T, N
for k in (1, 2, 3):
    T, N = block_space(k)
    print(f"elements supported in a {k}x{k} cell block: {len(N)}", flush=True)
