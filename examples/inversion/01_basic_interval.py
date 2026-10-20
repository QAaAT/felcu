"""
Inversion on a singular-value interval
======================================

Fit f(A) = A^{-1} for singular values in [0.5, 3].

The target 1/x has a pole at 0, so felcu fits on two intervals,
[-x_max, -x_min] and [x_min, x_max], and leaves the gap around the pole
unconstrained.

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
print(f"domain      = {result.domain}")
print(f"fitted on   = [-{result.x_max:.4f}, -{result.x_min:.4f}] U [{result.x_min:.4f}, {result.x_max:.4f}]")
print(f"basis       = {result.basis_used}")
print(f"eta         = {result.eta:.4f}   (default 1 + 1/kappa)")
print(f"m           = {result.m}")
print(f"Linf error  = {result.achieved_linf_error:.2e}")
print(f"alpha (LCU) = {result.alpha:.4f}")
