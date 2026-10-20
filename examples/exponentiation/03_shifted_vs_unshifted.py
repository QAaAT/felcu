"""
Shifted versus unshifted exponentiation
=======================================

For eigenvalues in [1, 3], the unshifted fit covers [-3, 3].  The shifted
fit centres at mu = 2 and only covers the half-width, using

    exp(H) = g(tau (H - mu I)),   g(x) = exp(mu + x / tau).

Run with:
    python examples/exponentiation/03_shifted_vs_unshifted.py
"""

from felcu import solve_fe_coefficients

for shifted in (False, True):
    r = solve_fe_coefficients(
        "exponentiation",
        shifted=shifted,
        sigma_min=1.0,
        sigma_max=3.0,
        target_error=1e-8,
        m_values=range(4, 33, 2),  # short sweep for speed; default goes to m=256
    )
    print(
        f"shifted={str(shifted):<5}  tau={r.tau:.4f}  m={r.m:<4}  alpha={r.alpha:.4f}  Linf={r.achieved_linf_error:.1e}"
    )
