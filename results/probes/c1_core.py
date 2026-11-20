"""Plan C, C1 core: ADMM in cell-unit derivative space (plate-independent equilibrium solves) + local exact repair.

Derivative space: per constraint j (ZP: triangle; FVS: triangle vertex) d_j = second derivatives of the Airy function
in cell units, rows (ww, uu, s*uw). Physical stresses N = S d with S = diag(1/hy^2, 1/hx^2, 1/(hx hy)) (sign of the
third row chosen by each arm so that N = (Nxx, Nyy, Nxy)). Yield: ||M N|| <= sy t_j  <=>  ||G d|| <= sy t_j, G = M S:
one ellipsoid shape per plate, radius per constraint. Equilibrium solves never see hx, hy: they are plate-independent.

ADMM (scaled):  min -P(x)  s.t.  D(x) = d,  d in ellipsoids
    x <- argmin -P(x) + rho/2 ||D(x) - d + u||^2      (one equilibrium least-squares solve; P = h . D(x) for both arms)
    d <- Proj_ellipsoids(D(x) + u);  u <- u + D(x) - d
"""
import numpy as np

SQ3 = np.sqrt(3.0)
MV = np.array([[1.0, -0.5, 0.0], [0.0, 0.5 * SQ3, 0.0], [0.0, 0.0, SQ3]])


class Ellipsoids:
    """Euclidean projection onto {d: ||G d|| <= rad_j} for many d (rows), one G (3x3)."""

    def __init__(self, G, rad):
        lam, Q = np.linalg.eigh(G.T @ G)
        self.lam, self.Q, self.rad, self.G = lam, Q, np.asarray(rad, float), G

    def ratio(self, d):
        return np.linalg.norm(d @ self.G.T, axis=1) / self.rad

    def project(self, w, newton=30):
        wh = w @ self.Q                                                  # rotated: q(d) = sum lam_i dh_i^2
        lam, r2 = self.lam, self.rad ** 2
        out = np.einsum("ij,ij->i", wh * lam, wh) > r2
        if not out.any():
            return w.copy()
        a = wh[out] ** 2 * lam
        mu = np.zeros(out.sum())
        for _ in range(newton):                                          # phi(mu) = sum lam a_i/(1+mu lam_i)^2 - r^2
            den = 1 + mu[:, None] * lam
            phi = (a / den ** 2).sum(1) - r2[out]
            dphi = (-2 * a * lam / den ** 3).sum(1)
            step = phi / dphi
            mu = mu - step                                               # convex decreasing: Newton from 0 is monotone
            if np.abs(step).max() < 1e-14 * max(1.0, mu.max()):
                break
        res = w.copy()
        res[out] = (wh[out] / (1 + mu[:, None] * lam)) @ self.Q.T
        return res


def test_projection():
    from scipy.optimize import minimize
    rng = np.random.default_rng(0)
    hx, hy = 0.9, 1.1
    G = MV @ np.diag([1 / hy ** 2, 1 / hx ** 2, 1 / (hx * hy)])
    E = Ellipsoids(G, rng.uniform(0.5, 2, 50))
    w = rng.standard_normal((50, 3)) * 3
    p = E.project(w)
    err = 0.0
    for i in range(50):
        cons = {"type": "ineq", "fun": lambda d, i=i: E.rad[i] ** 2 - np.sum((G @ d) ** 2)}
        sol = minimize(lambda d: np.sum((d - w[i]) ** 2), w[i] * 0.1, constraints=[cons], method="SLSQP",
                       options=dict(ftol=1e-14, maxiter=500))
        err = max(err, np.abs(sol.x - p[i]).max())
    print(f"ellipsoid projection vs SLSQP: max |diff| {err:.1e}; max ratio after projection {E.ratio(p).max():.12f}")


if __name__ == "__main__":
    test_projection()
