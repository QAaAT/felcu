"""
Cost of inversion versus condition number
=========================================

For singular values in [1/kappa, 1], plot the shortest series that meets
the target (selection='smallest_m') and its subnormalisation alpha_m as the
condition number kappa grows.  The lower bound alpha* = ||g||_inf on the
fitted set (dotted) equals ||A^-1|| = kappa here, so it is a line rather
than a constant.
Needs matplotlib.

Run with:
    python examples/plots/04_inversion_vs_kappa.py [OUTPUT_DIR]
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import NullFormatter

import _style
from felcu import solve_fe_coefficients

TARGET = 1e-4
KAPPAS = np.geomspace(1.5, 50, 12)
color = _style.PROBLEM_COLORS["inversion"]

rows = []
for kappa in KAPPAS:
    r = solve_fe_coefficients(
        "inversion", sigma_min=1.0 / kappa, sigma_max=1.0, target_error=TARGET, selection="smallest_m"
    )
    rows.append((kappa, r.m, r.alpha, _style.alpha_floor(r), r.converged))

kappa, m, alpha, alpha_star, conv = map(np.array, zip(*rows))
ok = conv.astype(bool)

_style.apply()
fig, (ax_m, ax_alpha) = plt.subplots(1, 2, figsize=(10, 4))
ax_m.loglog(kappa[ok], m[ok], marker="o", color=color)
ax_m.set_title(r"Shortest feasible order $m$  ($\epsilon_{\mathrm{t}} = 10^{-4}$)")
ax_alpha.loglog(kappa, alpha_star, color=_style.INK_2, linewidth=1, linestyle=":", label=_style.ALPHA_FLOOR_LABEL)
ax_alpha.loglog(kappa[ok], alpha[ok], marker="o", color=color, label=r"$\alpha_m$")
ax_alpha.legend(loc="upper left")
ax_alpha.set_title(r"Subnormalisation $\alpha_m$")
for ax in (ax_m, ax_alpha):
    ax.set_xlabel(r"condition number $\kappa$")
    ax.set_xticks([2, 5, 10, 20, 50], labels=["$2$", "$5$", "$10$", "$20$", "$50$"])
    ax.xaxis.set_minor_formatter(NullFormatter())
fig.tight_layout()

print(f"{'kappa':>7}  {'m':>4}  {'alpha':>9}  {'alpha*':>9}  converged")
for row in rows:
    print(f"{row[0]:>7.2f}  {row[1]:>4}  {row[2]:>9.3f}  {row[3]:>9.3f}  {row[4]}")
if not ok.all():
    print("\n(non-converged points are left out of the plot: increase m_values to reach them)")

_style.finish(fig, "04_inversion_vs_kappa")
