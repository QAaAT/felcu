"""
Inversion on a definite interval
================================

Fit f(H) = H^{-1} for eigenvalues in [0.5, 3].

The target 1/x has a pole at 0.  The interval is definite, so the pole lies
outside it, and the default eta places it on the periodic boundary,
x = -tau mu = -pi.

Run with:
    python examples/inversion/01_basic_interval.py
"""

from felcu import solve_fe_coefficients

result = solve_fe_coefficients(
    "inversion",
    sigma_min=0.5,
    sigma_max=3.0,
    target_error=1e-6,
    m_values=range(4, 65, 4),  # short sweep for speed; default goes to m=256
)

print(f"converged   = {result.converged}")
print(f"mu, tau     = {result.mu:.4f}, {result.tau:.4f}")
print(f"pole at x   = {-result.tau * result.mu:.4f}")
print(f"basis       = {result.basis_used}")
print(f"eta         = {result.eta:.4f}   (default (kappa + 1) / (kappa - 1))")
print(f"m           = {result.m}")
print(f"Linf error  = {result.achieved_linf_error:.2e}")
print(f"alpha (LCU) = {result.alpha:.4f}")
