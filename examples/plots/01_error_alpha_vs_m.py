"""
Error, alpha and lambda versus truncation order m
=================================================

Sweep m for each problem at the same target and plot, per m:

- the achieved error eps_m = ||g - g_m||_inf on the fitted set (falls until
  it reaches the target eps_t, then stays there, because the
  regularisation absorbs the remaining slack);
- the subnormalisation alpha_m of each feasible order, against the lower
  bound alpha* = ||g||_inf on the fitted set (dotted), which no fit can beat;
- the regularisation strength lambda = sqrt(2 pi gamma) chosen for it.

The ringed point is the order picked by the default selection
('smallest_alpha').  Needs matplotlib.

Run with:
    python examples/plots/01_error_alpha_vs_m.py [OUTPUT_DIR]
"""

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

import _style
from felcu import solve_fe_coefficients

TARGET = 1e-5
M_VALUES = range(4, 129, 4)
# Vertical label nudges (points) for the action and square-root lines, which end close together.
LABEL_DY = {"action": -5, "square_root": 5}
PROBLEMS = [
    ("action", dict(sigma_min=-1.0, sigma_max=1.0)),
    ("inversion", dict(sigma_min=0.5, sigma_max=3.0)),
    ("exponentiation", dict(sigma_min=0.1, sigma_max=2.0)),
    ("square_root", dict(sigma_min=0.15, sigma_max=1.0)),
]

_style.apply()
fig, (ax_err, ax_alpha, ax_lam) = plt.subplots(3, 1, sharex=True, figsize=(7, 8.5))

for problem, kwargs in PROBLEMS:
    result = solve_fe_coefficients(problem, target_error=TARGET, m_values=M_VALUES, **kwargs)
    color = _style.PROBLEM_COLORS[problem]
    cands = result.all_results
    feasible = [c for c in cands if c.converged and c.lambda_used > 0]
    alpha_star = _style.alpha_floor(result)

    ax_err.semilogy(
        [c.m for c in cands],
        [c.achieved_linf_error for c in cands],
        marker="o",
        color=color,
        label=_style.PROBLEM_LABELS.get(problem, problem),
    )
    ms, alphas = [c.m for c in feasible], [c.alpha for c in feasible]
    _style.ref_line(ax_alpha, alpha_star, color=color)
    ax_alpha.semilogy(ms, alphas, marker="o", color=color)
    _style.label_end(
        ax_alpha, ms, alphas, _style.PROBLEM_LABELS.get(problem, problem), gap=11, dy=LABEL_DY.get(problem, 0)
    )
    ax_alpha.plot(
        result.m,
        result.alpha,
        marker="o",
        markersize=11,
        markerfacecolor="none",
        markeredgecolor=_style.INK,
        markeredgewidth=1.2,
    )
    ax_lam.semilogy(ms, [c.lambda_used for c in feasible], marker="o", color=color)

    print(f"\n{problem}: selected m={result.m}, alpha={result.alpha:.4f}, alpha* = ||g||_inf = {alpha_star:.4f}")
    print(f"  {'m':>4}  {'Linf':>9}  {'alpha':>9}  {'lambda':>9}")
    for c in cands:
        print(f"  {c.m:>4}  {c.achieved_linf_error:>9.2e}  {c.alpha:>9.4f}  {c.lambda_used:>9.2e}")

_style.target_line(ax_err, TARGET)
ax_err.set_title(r"Achieved error $\epsilon_m = \|g - g_m\|_{L^\infty(\Omega)}$")
ax_err.legend(loc="upper right", bbox_to_anchor=(1, 0.9))
ax_alpha.set_title(r"Subnormalisation $\alpha_m$ (feasible orders; ring = selected)")
ax_alpha.legend(
    handles=[Line2D([], [], color=_style.INK_2, linewidth=1, linestyle=":", label=_style.ALPHA_FLOOR_LABEL)],
    loc="upper right",
)
ax_lam.set_title(r"Regularisation strength $\sqrt{2\pi\gamma}$ chosen")
ax_lam.set_xlabel(r"truncation order $m$")
fig.tight_layout()
_style.finish(fig, "01_error_alpha_vs_m")
