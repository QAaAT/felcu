"""
Shifted action on an off-centre interval
========================================

For eigenvalues in [2, 5] the unshifted fit has to cover [-5, 5].  The
shifted construction centres the interval at mu = 3.5 instead:

    g(x) = mu + x / tau,   f(H) = g(tau (H - mu I))

so the constant term carries mu and only the half-width needs fitting.

Run with:
    python examples/action/03_shifted.py
"""

from felcu import solve_fe_coefficients

kwargs = dict(sigma_min=2.0, sigma_max=5.0, target_error=1e-8, m_values=range(4, 33, 2))

for shifted in (False, True):
    r = solve_fe_coefficients("action", shifted=shifted, **kwargs)
    print(
        f"shifted={str(shifted):<5}  basis={r.basis_used:<16}  m={r.m:<4}  "
        f"a0={r.a0:.4f}  alpha={r.alpha:.4f}  Linf={r.achieved_linf_error:.1e}"
    )
