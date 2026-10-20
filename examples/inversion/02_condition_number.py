"""
Cost of inversion versus condition number
=========================================

Fix sigma_max = 1 and shrink sigma_min, so kappa = 1 / sigma_min grows.
Both the truncation order m and the subnormalisation alpha increase.

Run with:
    python examples/inversion/02_condition_number.py
"""

from felcu import solve_fe_coefficients

print(f"{'kappa':>6}  {'m':>4}  {'alpha':>10}  {'Linf':>9}  converged")
for kappa in [2, 5, 10, 20]:
    r = solve_fe_coefficients(
        "inversion",
        sigma_min=1.0 / kappa,
        sigma_max=1.0,
        target_error=1e-4,
        m_values=range(4, 101, 4),  # short sweep for speed; default goes to m=256
        selection="smallest_m",  # report the shortest series that meets the target
    )
    print(f"{kappa:>6}  {r.m:>4}  {r.alpha:>10.3f}  {r.achieved_linf_error:>9.1e}  {r.converged}")
