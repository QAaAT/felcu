"""
Matrix square root from a positive-definite matrix
==================================================

Fit sqrt(H) from H directly, then check that the block-encoded operator
g(tau (H - mu I)) squares back to H.

Run with:
    python examples/square_root/02_matrix_square_root.py
"""

import numpy as np

from felcu import evaluate_fourier_series, solve_fe_coefficients

rng = np.random.default_rng(6)
X = rng.standard_normal((8, 8))
H = X @ X.T / 8 + 0.2 * np.eye(8)  # random symmetric positive definite

result = solve_fe_coefficients("sqrt", matrix=H, target_error=1e-8, m_values=range(8, 129, 8))
print(f"eigenvalues in = [{result.sigma_min:.4f}, {result.sigma_max:.4f}]")
print(f"m              = {result.m}")

mu = 0.5 * (result.sigma_min + result.sigma_max)
w, V = np.linalg.eigh(H)
g = evaluate_fourier_series(result.tau * (w - mu), result.a0, result.ak, result.bk)
sqrt_H = V @ np.diag(g) @ V.T

print(f"||approx^2 - H||_2           = {np.linalg.norm(sqrt_H @ sqrt_H - H, 2):.2e}")
print(f"||approx - V sqrt(L) V^T||_2 = {np.linalg.norm(sqrt_H - V @ np.diag(np.sqrt(w)) @ V.T, 2):.2e}")
