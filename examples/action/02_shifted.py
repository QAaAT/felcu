"""
Shifted action on an off-centre interval
========================================

For eigenvalues in [2, 5], felcu centres the fit at mu = 3.5:

    g(x) = mu + x / tau,   f(H) = g(tau (H - mu I))

so the constant term carries mu and only the half-width needs fitting.

H is positive definite here, so its singular values are its eigenvalues
and the unshifted singular-value construction (shifted=False) also
applies.  That one fits the odd g(x) = x / tau on the whole of [-5, 5],
which costs a larger alpha.

Run with:
    python examples/action/02_shifted.py
"""

from felcu import solve_fe_coefficients

kwargs = dict(sigma_min=2.0, sigma_max=5.0, target_error=1e-8, m_values=range(4, 33, 2))

for shifted in (True, False):
    r = solve_fe_coefficients("action", shifted=shifted, **kwargs)
    print(
        f"shifted={str(shifted):<5}  {r.spectral_kind_used:<15}  basis={r.basis_used:<16}  m={r.m:<4}  "
        f"a0={r.a0:.4f}  alpha={r.alpha:.4f}  Linf={r.achieved_linf_error:.1e}"
    )
