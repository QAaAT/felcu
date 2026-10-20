"""
Action from a Hermitian matrix
==============================

Pass the matrix itself: felcu reads its eigenvalue range.  We then check
the block-encoded operator g(tau H) against H using an eigendecomposition.

Run with:
    python examples/action/04_hermitian_matrix.py
"""

import numpy as np

from felcu import evaluate_fourier_series, solve_fe_coefficients

rng = np.random.default_rng(0)
n = 8
X = rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n))
H = (X + X.conj().T) / 4  # random complex Hermitian

result = solve_fe_coefficients("action", matrix=H, target_error=1e-8, m_values=range(4, 33, 2))
print(f"spectral kind   = {result.spectral_kind_used}")
print(f"eigenvalues in  = [{result.sigma_min:.4f}, {result.sigma_max:.4f}]")
print(f"m               = {result.m}")

w, V = np.linalg.eigh(H)
g = evaluate_fourier_series(result.tau * w, result.a0, result.ak, result.bk)
H_approx = V @ np.diag(g) @ V.conj().T

print(f"||g(tau H) - H||_2 = {np.linalg.norm(H_approx - H, 2):.2e}")
