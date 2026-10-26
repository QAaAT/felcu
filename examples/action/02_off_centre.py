"""
Action on an off-centre interval
================================

For eigenvalues in [2, 5], felcu centres the fit at mu = 3.5:

    g(x) = mu + x / tau,   f(H) = g(tau (H - mu I))

so the constant term carries mu and only the half-width needs fitting.

H is positive definite here, so its singular values are its eigenvalues,
and spectral_kind='singular_values' fits them through the Hermitian
dilation instead.  The dilation's spectrum, [-5, -2] U [2, 5], is centred at
0, so mu = 0 and the odd g(x) = x / tau covers the whole of [-5, 5], which
costs a larger alpha.

Run with:
    python examples/action/02_off_centre.py
"""

from felcu import solve_fe_coefficients

kwargs = dict(sigma_min=2.0, sigma_max=5.0, target_error=1e-8, m_values=range(4, 33, 2))

for kind in ("eigenvalues", "singular_values"):
    r = solve_fe_coefficients("action", spectral_kind=kind, **kwargs)
    print(
        f"{kind:<15}  mu={r.mu:<4}  basis={r.basis_used:<16}  m={r.m:<4}  "
        f"a0={r.a0:.4f}  alpha={r.alpha:.4f}  Linf={r.achieved_linf_error:.1e}"
    )
