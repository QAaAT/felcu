"""
Effect of the Sobolev regulariser
=================================

At fixed m, felcu adds a penalty lambda * (1 + k^2)^(k_order / 2) on the
k-th harmonic and picks the largest lambda that still meets target_error.
Compare Sobolev orders 0, 1 and 2 and see how lambda_used and the LCU
normalisation alpha change.

Run with:
    python examples/exponentiation/06_sobolev_regularisation.py
"""

from felcu import solve_fe_coefficients

common = dict(sigma_min=0.1, sigma_max=2.0, target_error=1e-6, m=40)

for k_order in (0, 1, 2):
    r = solve_fe_coefficients("exponentiation", k_order=k_order, **common)
    print(f"{'k_order=' + str(k_order):<10}  lambda={r.lambda_used:.1e}  alpha={r.alpha:.6f}  Linf={r.achieved_linf_error:.1e}")
