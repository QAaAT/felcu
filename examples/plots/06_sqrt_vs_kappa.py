"""
Cost of the square root versus condition number
===============================================

For eigenvalues in [1/kappa, 1], fit sqrt(H) (shifted, with the default
eta_def and r_sqrt = kappa^(-1/2)) and plot the shortest series that meets
the target and the smallest subnormalisation alpha_m over the sweep as
kappa grows.  ||sqrt(H)|| = 1 throughout, so the
lower bound alpha* = 1 (dotted) is constant.  Unlike inversion, sqrt is
concave and the periodic closure narrows with kappa, so the paper predicts
alpha_m = O(kappa^(1/4)) for w = 1 (dashed, through the first point).
Needs matplotlib.

Run with:
    python examples/plots/06_sqrt_vs_kappa.py [OUTPUT_DIR]
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import NullFormatter

import _style
from felcu import solve_fe_coefficients

TARGET = 1e-6
KAPPAS = np.geomspace(1.5, 50, 10)
color = _style.PROBLEM_COLORS["square_root"]

rows = []
for kappa in KAPPAS:
    r = solve_fe_coefficients(
        "square_root",
        sigma_min=1.0 / kappa,
        sigma_max=1.0,
        target_error=TARGET,
        m_values=range(8, 257, 8),
    )
    m_min = min((c.m for c in r.all_results if c.converged), default=0)
    rows.append((kappa, m_min, r.alpha, _style.alpha_floor(r), r.converged))

kappa, m, alpha, alpha_star, conv = map(np.array, zip(*rows))
ok = conv.astype(bool)

_style.apply()
fig, (ax_m, ax_alpha) = plt.subplots(1, 2, figsize=(10, 4))
ax_m.loglog(kappa[ok], m[ok], marker="o", color=color)
_style.label_end(ax_m, kappa[ok], m[ok], r"$\sqrt{\mathcal{H}}$")
ax_m.set_title(r"Shortest feasible order $m$  ($\epsilon_{\mathrm{t}} = 10^{-6}$)")
ax_alpha.loglog(kappa, alpha_star, color=_style.INK_2, linewidth=1, linestyle=":", label=_style.ALPHA_FLOOR_LABEL)
first = np.argmax(ok)
ax_alpha.loglog(
    kappa,
    alpha[first] * (kappa / kappa[first]) ** 0.25,
    color=_style.INK_2,
    linewidth=1,
    linestyle="--",
    label=r"$\propto \kappa^{1/4}$",
)
ax_alpha.loglog(kappa[ok], alpha[ok], marker="o", color=color, label=r"$\alpha_m$")
_style.label_end(ax_alpha, kappa[ok], alpha[ok], r"$\sqrt{\mathcal{H}}$")
ax_alpha.legend(loc="upper left")
ax_alpha.set_title(r"Subnormalisation $\alpha_m$")
for ax in (ax_m, ax_alpha):
    ax.set_xlabel(r"condition number $\kappa$")
    ax.set_xticks([2, 5, 10, 20, 50], labels=["$2$", "$5$", "$10$", "$20$", "$50$"])
    ax.xaxis.set_minor_formatter(NullFormatter())
fig.tight_layout()

print(f"{'kappa':>7}  {'m_min':>5}  {'alpha':>9}  {'alpha*':>9}  converged")
for row in rows:
    print(f"{row[0]:>7.2f}  {row[1]:>5}  {row[2]:>9.3f}  {row[3]:>9.3f}  {row[4]}")
if not ok.all():
    print("\n(non-converged points are left out of the plot: increase m_values to reach them)")

_style.finish(fig, "06_sqrt_vs_kappa")
