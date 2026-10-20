"""
Action on a non-Hermitian matrix
================================

For a non-Hermitian A, felcu switches to the singular values automatically.
The odd sine series acts on the singular values (as in QSVT):

    g(tau A) = U diag(g(tau s)) V^dagger  ~=  A

Run with:
    python examples/action/05_non_hermitian_matrix.py
"""

import numpy as np

from felcu import evaluate_fourier_series, solve_fe_coefficients

rng = np.random.default_rng(1)
A = rng.standard_normal((6, 6)) / 3

result = solve_fe_coefficients("action", matrix=A, target_error=1e-8, m_values=range(4, 33, 2))
print(f"hermitian?           = {result.matrix_is_hermitian}")
print(f"spectral kind        = {result.spectral_kind_used}")
print(f"singular values in   = [{result.sigma_min:.4f}, {result.sigma_max:.4f}]")
print(f"m                    = {result.m}")

U, s, Vh = np.linalg.svd(A)
g = evaluate_fourier_series(result.tau * s, result.a0, result.ak, result.bk)
A_approx = U @ np.diag(g) @ Vh

print(f"||g(tau A) - A||_2   = {np.linalg.norm(A_approx - A, 2):.2e}")
