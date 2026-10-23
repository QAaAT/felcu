"""
Action on a symmetric interval
==============================

Fit f(H) = H for a Hermitian H with eigenvalues in [-1, 1].

felcu centres the fit at mu = 0, the middle of the interval, and fits
g(x) = x / tau.  Because that is odd, every cosine coefficient (bk) and
the constant term (a0) vanish.

Run with:
    python examples/action/01_basic_interval.py
"""

import numpy as np

from felcu import solve_fe_coefficients

result = solve_fe_coefficients(
    "action",
    sigma_min=-1.0,
    sigma_max=1.0,
    target_error=1e-8,
    m_values=range(4, 33, 2),  # short sweep for speed; default goes to m=256
)

print(f"converged      = {result.converged}")
print(f"basis          = {result.basis_used}")
print(f"m              = {result.m}")
print(f"tau            = {result.tau:.4f}   (fit g(x) = x / tau, then f(H) = g(tau H))")
print(f"Linf error     = {result.achieved_linf_error:.2e}")
print(f"alpha (LCU)    = {result.alpha:.4f}")
print(f"|a0|           = {abs(result.a0):.1e}")
print(f"max |bk|       = {np.max(np.abs(result.bk)):.1e}")
print(f"first 5 ak     = {np.array2string(result.ak[:5], precision=4)}")
