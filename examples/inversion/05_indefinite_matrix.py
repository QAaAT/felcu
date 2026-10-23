"""
Inversion of an indefinite Hermitian matrix
===========================================

When H has eigenvalues on both sides of 0, in [lambda_min, a] U [b, lambda_max]
with a < 0 < b, felcu fits 1/x on both parts and leaves the gap (a, b) around
the pole free.  The fit is still centred at the midpoint mu of
[lambda_min, lambda_max], which need not be 0:

    f(H) = g(tau (H - mu I))

With a matrix, the gap is read from its eigenvalues; with bounds, pass
gap=(a, b).

Run with:
    python examples/inversion/05_indefinite_matrix.py
"""

import numpy as np

from felcu import evaluate_fourier_series, solve_fe_coefficients

rng = np.random.default_rng(4)
Q, _ = np.linalg.qr(rng.standard_normal((6, 6)))
H = Q @ np.diag([-2.0, -1.2, -0.5, 0.4, 1.6, 3.0]) @ Q.T

r = solve_fe_coefficients("inversion", matrix=H, target_error=1e-6, m_values=range(8, 129, 8))
print(f"spectrum    = [{r.sigma_min:.2f}, {r.gap[0]:.2f}] U [{r.gap[1]:.2f}, {r.sigma_max:.2f}]")
print(f"mu, tau     = {r.mu:.4f}, {r.tau:.4f}")
print(f"fitted on   = [{-r.x_max:.4f}, {r.x_gap[0]:.4f}] U [{r.x_gap[1]:.4f}, {r.x_max:.4f}]")
print(f"eta         = {r.eta:.4f}   (default 1 + (b - a) / (2 delta))")
print(f"m, alpha    = {r.m}, {r.alpha:.4f}")

w, V = np.linalg.eigh(H)
g = evaluate_fourier_series(r.tau * (w - r.mu), r.a0, r.ak, r.bk)
print(f"||approx - H^-1||_2 = {np.linalg.norm(V @ np.diag(g) @ V.T - np.linalg.inv(H), 2):.1e}")

same = solve_fe_coefficients(
    "inversion", sigma_min=-2.0, sigma_max=3.0, gap=(-0.5, 0.4), target_error=1e-6, m_values=range(8, 129, 8)
)
print(f"from bounds: m={same.m}  alpha={same.alpha:.4f}")
