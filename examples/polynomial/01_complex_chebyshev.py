"""
Complex polynomial in the Chebyshev basis
=========================================

Fit f(z) = (1 + 0.5i) T_0(z) - 0.3i T_1(z) + 0.25 T_2(z) + (0.4 - 0.1i) T_3(z)
for eigenvalues in [-0.5, 1.5].  The Chebyshev polynomials are evaluated at
the eigenvalues themselves (no mapping onto [-1, 1]), and the complex
coefficients give a complex-valued f, so the Fourier coefficients are
complex too, with c_{-k} != conj(c_k).

Then check the block-encoded operator against the exact f(H) for a
Hermitian H with that spectrum, and compare the three methods.

Run with:
    python examples/polynomial/01_complex_chebyshev.py
"""

import numpy as np

from felcu import evaluate_fourier_series, solve_fe_coefficients

gamma = [1 + 0.5j, -0.3j, 0.25, 0.4 - 0.1j]
f = np.polynomial.Chebyshev(gamma)
common = dict(polynomial_coefficients=gamma, polynomial_basis="chebyshev", sigma_min=-0.5, sigma_max=1.5)

result = solve_fe_coefficients("polynomial", target_error=1e-10, m_values=range(8, 33, 4), **common)
c, m = result.complex_coefficients, result.m
print(f"m           = {m}")
print(f"Linf error  = {result.achieved_linf_error:.2e}")
print(f"alpha       = {result.alpha:.4f}   (sum of |c_k| over both signed modes)")
print(f"c_1, c_-1   = {c[m + 1]:.4f}, {c[m - 1]:.4f}")

rng = np.random.default_rng(0)
Q, _ = np.linalg.qr(rng.standard_normal((6, 6)))
lam = np.linspace(-0.5, 1.5, 6)
H = Q @ np.diag(lam) @ Q.T
g = evaluate_fourier_series(result.tau * (lam - result.mu), result.a0, result.ak, result.bk)
print(f"||g(tau (H - mu I)) - f(H)||_2 = {np.linalg.norm(Q @ np.diag(g - f(lam)) @ Q.T, 2):.2e}")

print(f"\n{'method':<15} {'eta':>4} {'m':>4}  {'Linf':>8}  {'alpha':>7}")
for method in ("sobolev", "reflected", "arcsine_taylor"):
    r = solve_fe_coefficients("polynomial", method=method, m=64, **common)
    print(f"{method:<15} {r.eta:>4.1f} {r.m:>4}  {r.achieved_linf_error:>8.1e}  {r.alpha:>7.4f}")
