"""
Logarithm on a positive-definite interval
=========================================

Fit f(H) = log(H) for eigenvalues in [0.2, 5].  Like the square root, log
has a branch point at 0, and the default eta = (kappa + 1) / (kappa - 1)
places it on the periodic boundary.  The Sobolev scale defaults to
r = kappa^(-1/2) / (1 + ln kappa) for w = 1.

The spectrum straddles 1, so log(H) has eigenvalues of both signs and
||log(H)|| = max(|log 0.2|, |log 5|).

The explicit methods work too, shown at fixed orders on [0.5, 2]: the
reflected extension converges like O(1/m), and arcsine-Taylor
exponentially with factor sin(pi / eta).  On [0.2, 5] the latter would be
slow, since 2 < eta < 2 mu / delta = 2.17 leaves eta close to 2.

Run with:
    python examples/logarithm/01_basic_interval.py
"""

import numpy as np

from felcu import solve_fe_coefficients

common = dict(sigma_min=0.2, sigma_max=5.0)
result = solve_fe_coefficients(
    "logarithm",
    target_error=1e-8,
    m_values=range(8, 129, 8),  # short sweep for speed; default goes to m=256
    **common,
)

norm = max(abs(np.log(0.2)), abs(np.log(5.0)))
print(f"converged   = {result.converged}")
print(f"eta         = {result.eta:.4f}   (default (kappa + 1) / (kappa - 1))")
print(f"sobolev_r   = {result.sobolev_r:.4f}   (default kappa^(-1/2) / (1 + ln kappa))")
print(f"m           = {result.m}")
print(f"Linf error  = {result.achieved_linf_error:.2e}")
print(f"alpha       = {result.alpha:.4f}   (lower bound ||log(H)|| = {norm:.4f})")

print(f"\nOn [0.5, 2]:\n{'method':<15} {'eta':>6} {'m':>4}  {'Linf':>8}  {'alpha':>7}")
for method in ("sobolev", "reflected", "arcsine_taylor"):
    r = solve_fe_coefficients("logarithm", method=method, sigma_min=0.5, sigma_max=2.0, m=64)
    print(f"{method:<15} {r.eta:>6.3f} {r.m:>4}  {r.achieved_linf_error:>8.1e}  {r.alpha:>7.4f}")
