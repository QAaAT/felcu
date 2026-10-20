"""
Matrix exponential from a Hermitian matrix
==========================================

Fit exp(H) from H directly, then compare g(tau H) with the exact matrix
exponential computed by eigendecomposition.

Run with:
    python examples/exponentiation/04_matrix_exponential.py
"""

import numpy as np
from felcu import evaluate_fourier_series, solve_fe_coefficients

rng = np.random.default_rng(4)
n = 10
X = rng.standard_normal((n, n))
H = (X + X.T) / 6                            # random real symmetric

result = solve_fe_coefficients("exponentiation", matrix=H, target_error=1e-8, m_values=range(4, 33, 2))
print(f"eigenvalues in = [{result.sigma_min:.4f}, {result.sigma_max:.4f}]")
print(f"m              = {result.m}")

w, V = np.linalg.eigh(H)
exp_H = V @ np.diag(np.exp(w)) @ V.T
g = evaluate_fourier_series(result.tau * w, result.a0, result.ak, result.bk)
approx = V @ np.diag(g) @ V.T

print(f"||g(tau H) - exp(H)||_2 = {np.linalg.norm(approx - exp_H, 2):.2e}")
