"""
Comparing the three coefficient-selection methods
=================================================

Fit exp(H) for eigenvalues in [0.1, 2] with each method at the same orders:

- 'reflected' has the optimal subnormalisation alpha_m = ||f(H)|| - error,
  but its error only falls as O(1/m);
- 'arcsine_taylor' converges exponentially, with alpha approaching
  exp(mu + eta delta / 2), f at the enlarged endpoint;
- 'sobolev' (the default) converges down to the target error with alpha
  close to optimal.

Run with:
    python examples/methods/01_compare_methods.py
"""

import math

from felcu import solve_fe_coefficients

common = dict(shifted=True, sigma_min=0.1, sigma_max=2.0, target_error=1e-10)
print(f"||f(H)|| = exp(2) = {math.exp(2.0):.6f}\n")
print(f"{'m':>4}  {'method':<15} {'eta':>5}  {'Linf':>9}  {'alpha':>9}")
for m in (16, 32, 64):
    for method in ("reflected", "arcsine_taylor", "sobolev"):
        r = solve_fe_coefficients("exponentiation", method=method, m=m, **common)
        print(f"{m:>4}  {method:<15} {r.eta:>5.2f}  {r.achieved_linf_error:>9.1e}  {r.alpha:>9.6f}")
    print()
