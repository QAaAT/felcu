"""
Identity from a Hermitian matrix
================================

Pass the matrix itself: felcu reads its eigenvalue range.  We then check
the block-encoded operator g(tau (H - mu I)) against H using an
eigendecomposition, with mu the centre of the eigenvalue range.

Run with:
    python examples/identity/03_hermitian_matrix.py
"""

import numpy as np

from felcu import solve_fe_coefficients

rng = np.random.default_rng(0)
n = 8
X = rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n))
H = (X + X.conj().T) / 4  # random complex Hermitian

result = solve_fe_coefficients("identity", matrix=H, target_error=1e-8, m_values=range(4, 33, 2))
print(f"spectral kind   = {result.spectral_kind_used}")
print(f"eigenvalues in  = [{result.sigma_min:.4f}, {result.sigma_max:.4f}]")
print(f"mu              = {result.mu:.4f}")
print(f"m               = {result.m}")

H_approx = result.apply(H)  # g(tau (H - mu I)), by eigendecomposition

print(f"||g(tau (H - mu I)) - H||_2 = {np.linalg.norm(H_approx - H, 2):.2e}")
