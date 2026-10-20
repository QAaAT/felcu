"""
Shifted inversion of a definite Hermitian matrix
================================================

When H is definite (all eigenvalues the same sign), the shifted
construction fits 1/x on a single interval centred at mu:

    f(H) = g(tau (H - mu I))

This works for positive- and negative-definite H alike.

Run with:
    python examples/inversion/04_shifted_definite.py
"""

import numpy as np

from felcu import evaluate_fourier_series, solve_fe_coefficients

rng = np.random.default_rng(3)
Q, _ = np.linalg.qr(rng.standard_normal((6, 6)))

for label, eigs in [("positive definite", np.linspace(0.5, 2.0, 6)), ("negative definite", np.linspace(-2.0, -0.5, 6))]:
    H = Q @ np.diag(eigs) @ Q.T
    r = solve_fe_coefficients("inversion", shifted=True, matrix=H, target_error=1e-6, m_values=range(4, 65, 4))

    mu = 0.5 * (r.sigma_min + r.sigma_max)
    w, V = np.linalg.eigh(H)
    g = evaluate_fourier_series(r.tau * (w - mu), r.a0, r.ak, r.bk)
    err = np.linalg.norm(V @ np.diag(g) @ V.T - np.linalg.inv(H), 2)

    print(f"{label}:  mu={mu:+.2f}  basis={r.basis_used}  m={r.m}  alpha={r.alpha:.3f}  ||approx - H^-1||_2={err:.1e}")
