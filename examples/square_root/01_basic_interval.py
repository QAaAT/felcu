"""
Square root on a positive-definite interval
===========================================

Fit f(H) = sqrt(H) for eigenvalues in [0.15, 1], the square-root example of
the paper.  sqrt has a branch point at 0, and the default
eta = (kappa + 1) / (kappa - 1) places it on the periodic boundary.  The Sobolev scale defaults to
r = kappa^(-1/2) for w = 1.

Run with:
    python examples/square_root/01_basic_interval.py
"""

from felcu import solve_fe_coefficients

result = solve_fe_coefficients(
    "square_root",
    sigma_min=0.15,
    sigma_max=1.0,
    target_error=1e-8,
    m_values=range(8, 129, 8),  # short sweep for speed; default goes to m=256
)

print(f"converged   = {result.converged}")
print(f"eta         = {result.eta:.4f}   (default (kappa + 1) / (kappa - 1))")
print(f"sobolev_r   = {result.sobolev_r:.4f}   (default kappa^(-1/2))")
print(f"m           = {result.m}")
print(f"Linf error  = {result.achieved_linf_error:.2e}")
print(f"alpha       = {result.alpha:.4f}   (lower bound ||sqrt(H)|| = 1)")
