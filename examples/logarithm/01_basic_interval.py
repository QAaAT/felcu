"""
Logarithm on a positive-definite interval
=========================================

Fit f(H) = log(H) for eigenvalues in [0.2, 5].  Like the square root, log
has a branch point at 0, so it uses the shifted construction with
eta = (kappa + 1) / (kappa - 1).  The Sobolev scale defaults to the gap to
the branch point, r = min(1, 2 pi / (kappa + 1)).

The spectrum straddles 1, so log(H) has eigenvalues of both signs and
||log(H)|| = max(|log 0.2|, |log 5|).

Run with:
    python examples/logarithm/01_basic_interval.py
"""

import numpy as np

from felcu import solve_fe_coefficients

result = solve_fe_coefficients(
    "logarithm",
    shifted=True,
    sigma_min=0.2,
    sigma_max=5.0,
    target_error=1e-8,
    m_values=range(8, 129, 8),  # short sweep for speed; default goes to m=256
)

norm = max(abs(np.log(0.2)), abs(np.log(5.0)))
print(f"converged   = {result.converged}")
print(f"eta         = {result.eta:.4f}")
print(f"sobolev_r   = {result.sobolev_r:.4f}")
print(f"m           = {result.m}")
print(f"Linf error  = {result.achieved_linf_error:.2e}")
print(f"alpha       = {result.alpha:.4f}   (lower bound ||log(H)|| = {norm:.4f})")
