"""
Effect of the Sobolev regulariser
=================================

At fixed m, felcu regularises with the H_r^w norm, weighting the k-th
harmonic by (1 + r|k|)^(2w), and picks the largest lambda that still
meets target_error.  Vary the order w and the scale r and see how
lambda_used and the LCU normalisation alpha change.

Run with:
    python examples/exponentiation/06_sobolev_regularisation.py
"""

from felcu import solve_fe_coefficients

common = dict(sigma_min=0.1, sigma_max=2.0, target_error=1e-6, m=40)

print("Varying the order w (r = 1):")
for w in (0.75, 1.0, 1.5, 2.0):
    r = solve_fe_coefficients("exponentiation", sobolev_w=w, sobolev_r=1.0, **common)
    print(f"  w={w:<4}  lambda={r.lambda_used:.1e}  alpha={r.alpha:.6f}  Linf={r.achieved_linf_error:.1e}")

print("\nVarying the scale r (w = 1):")
for scale in (0.1, 0.25, 0.5, 1.0):
    r = solve_fe_coefficients("exponentiation", sobolev_w=1.0, sobolev_r=scale, **common)
    print(f"  r={scale:<4}  lambda={r.lambda_used:.1e}  alpha={r.alpha:.6f}  Linf={r.achieved_linf_error:.1e}")
