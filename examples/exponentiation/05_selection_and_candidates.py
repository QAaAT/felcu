"""
Choosing between candidate truncation orders
============================================

With the default selection='smallest_alpha', felcu solves for every m in
m_values, keeps all candidates in result.all_results, and picks the one
with the smallest subnormalisation.  'smallest_m' picks the shortest
series that meets the target instead, and stops the sweep there.

Run with:
    python examples/exponentiation/05_selection_and_candidates.py
"""

from felcu import solve_fe_coefficients

common = dict(sigma_min=0.1, sigma_max=2.0, target_error=1e-8, m_values=range(8, 41, 4))

r = solve_fe_coefficients("exponentiation", **common)
print(f"{'m':>4}  {'Linf':>9}  {'alpha':>9}  converged")
for c in r.all_results:
    print(f"{c.m:>4}  {c.achieved_linf_error:>9.1e}  {c.alpha:>9.6f}  {c.converged}")

for selection in ("smallest_m", "smallest_alpha"):
    s = solve_fe_coefficients("exponentiation", selection=selection, **common)
    print(f"\nselection={selection!r:<17} -> m={s.m}, alpha={s.alpha:.6f}")
