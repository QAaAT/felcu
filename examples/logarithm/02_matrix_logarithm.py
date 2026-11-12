"""
Matrix logarithm from a positive-definite matrix
================================================

Fit log(H) from H directly, then compare the block-encoded operator
g(tau (H - mu I)) with the exact matrix logarithm, and check that its
exponential recovers H.

Run with:
    python examples/logarithm/02_matrix_logarithm.py
"""

import numpy as np

from felcu import solve_fe_coefficients

rng = np.random.default_rng(7)
Q, _ = np.linalg.qr(rng.standard_normal((6, 6)))
lam = np.array([0.5, 0.8, 1.0, 1.7, 2.5, 4.0])
H = Q @ np.diag(lam) @ Q.T

result = solve_fe_coefficients("log", matrix=H, target_error=1e-8, m_values=range(8, 129, 8))
print(f"eigenvalues in = [{result.sigma_min:.4f}, {result.sigma_max:.4f}]")
print(f"m              = {result.m}")

log_H = result.apply(H)  # g(tau (H - mu I)), by eigendecomposition
w, V = np.linalg.eigh(log_H)  # log_H is symmetric, so exponentiate it the same way
exp_log_H = V @ np.diag(np.exp(w)) @ V.T

print(f"||approx - log(H)||_2 = {np.linalg.norm(log_H - Q @ np.diag(np.log(lam)) @ Q.T, 2):.2e}")
print(f"||exp(approx) - H||_2 = {np.linalg.norm(exp_log_H - H, 2):.2e}")
