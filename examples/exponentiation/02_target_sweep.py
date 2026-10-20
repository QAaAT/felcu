"""
Truncation order versus target error
====================================

Tighten target_error and watch how the selected order m grows.  For a
smooth target like exp, m grows roughly logarithmically in 1/error.

Run with:
    python examples/exponentiation/02_target_sweep.py
"""

from felcu import solve_fe_coefficients

print(f"{'target':>8}  {'m':>4}  {'Linf':>9}  {'alpha':>8}")
for target in [1e-2, 1e-4, 1e-6, 1e-8, 1e-10, 1e-12]:
    r = solve_fe_coefficients(
        "exponentiation",
        sigma_min=0.1,
        sigma_max=2.0,
        target_error=target,
        m_values=range(4, 41, 2),  # short sweep for speed; default goes to m=256
    )
    print(f"{target:>8.0e}  {r.m:>4}  {r.achieved_linf_error:>9.1e}  {r.alpha:>8.4f}")
