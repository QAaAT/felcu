"""
Choosing between candidate truncation orders
============================================

With the default selection='smallest_norm', felcu solves for every m in
m_values, keeps all candidates in result.all_results, and picks the one
with the smallest selection score J_p.  For the default w = 1 fit p = 0
and J_0 is the subnormalisation alpha.  'smallest_m' picks the shortest
series that meets the target instead, and stops the sweep there.

With sobolev_w = 2 the score is J_1 = sum_{k != 0} |k| |c_k|, which can
pick a different order from 'smallest_alpha'; both are reported.

Run with:
    python examples/exponentiation/05_selection_and_candidates.py
"""

from felcu import solve_fe_coefficients

common = dict(sigma_min=0.1, sigma_max=2.0, target_error=1e-8, m_values=range(8, 41, 4))

r = solve_fe_coefficients("exponentiation", **common)
print(f"{'m':>4}  {'Linf':>9}  {'alpha':>9}  converged")
for c in r.all_results:
    print(f"{c.m:>4}  {c.achieved_linf_error:>9.1e}  {c.alpha:>9.6f}  {c.converged}")

for selection in ("smallest_m", "smallest_norm"):
    s = solve_fe_coefficients("exponentiation", selection=selection, **common)
    print(f"\nselection={selection!r:<17} -> m={s.m}, alpha={s.alpha:.6f}")

# An order-aware score: p = sobolev_w - 1 = 1 for this square-root fit.
sqrt = dict(sigma_min=0.1, sigma_max=1.0, sobolev_w=2.0, target_error=1e-6, m_values=range(8, 57, 8))
for selection in ("smallest_norm", "smallest_alpha"):
    s = solve_fe_coefficients("square_root", selection=selection, **sqrt)
    print(
        f"\nsqrt, selection={selection!r:<16} -> m={s.m}, alpha={s.alpha:.6f}, "
        f"J_{s.selection_order:g}={s.selection_score:.6f}, readout norm={s.readout_norm:.4f}"
    )
