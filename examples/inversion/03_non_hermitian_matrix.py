"""
Inverting a non-Hermitian matrix
================================

A has no real eigenvalue interval, so felcu fits its singular values,
through the Hermitian dilation [[0, A], [A^dagger, 0]].  The dilation's
spectrum is symmetric about 0, so mu = 0, and the fit is the odd
g(x) = tau / x on two intervals either side of the pole.  With
A = U diag(s) V^dagger,

    V diag(g(tau s)) U^dagger  ~=  A^{-1}

Run with:
    python examples/inversion/03_non_hermitian_matrix.py
"""

import numpy as np

from felcu import evaluate_fourier_series, solve_fe_coefficients

rng = np.random.default_rng(2)
U0, _ = np.linalg.qr(rng.standard_normal((5, 5)))
V0, _ = np.linalg.qr(rng.standard_normal((5, 5)))
A = U0 @ np.diag([0.4, 0.7, 1.0, 1.5, 2.0]) @ V0.T  # kappa = 5

result = solve_fe_coefficients("inversion", matrix=A, target_error=1e-6, m_values=range(4, 65, 4))
print(f"spectral kind       = {result.spectral_kind_used}  (mu = {result.mu})")
print(f"singular values in  = [{result.sigma_min:.4f}, {result.sigma_max:.4f}]")
print(f"m                   = {result.m}")

U, s, Vh = np.linalg.svd(A)
g = evaluate_fourier_series(result.tau * s, result.a0, result.ak, result.bk)
A_inv_approx = Vh.T @ np.diag(g) @ U.T

A_inv = np.linalg.inv(A)
print(f"||approx - A^-1||_2 = {np.linalg.norm(A_inv_approx - A_inv, 2):.2e}")
print(f"||A approx - I||_2  = {np.linalg.norm(A @ A_inv_approx - np.eye(5), 2):.2e}")
