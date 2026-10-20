"""
Cost of inversion versus condition number
=========================================

For singular values in [1/kappa, 1], plot the shortest series that meets
the target (selection='smallest_m') and its LCU normalisation alpha as the
condition number kappa grows.  Needs matplotlib.

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
    r = solve_fe_coefficients("inversion", sigma_min=1.0 / kappa, sigma_max=1.0,
                              target_error=TARGET, selection="smallest_m")
    rows.append((kappa, r.m, r.alpha, r.converged))

kappa, m, alpha, conv = map(np.array, zip(*rows))
ok = conv.astype(bool)

_style.apply()
fig, (ax_m, ax_alpha) = plt.subplots(1, 2, figsize=(10, 4))
ax_m.loglog(kappa[ok], m[ok], marker="o", color=color)
ax_m.set_title("Shortest feasible order m  (target 1e-4)")
ax_alpha.loglog(kappa[ok], alpha[ok], marker="o", color=color)
ax_alpha.set_title("LCU normalisation alpha")
for ax in (ax_m, ax_alpha):
    ax.set_xlabel("condition number kappa")
    ax.set_xticks([2, 5, 10, 20, 50], labels=["2", "5", "10", "20", "50"])
    ax.xaxis.set_minor_formatter(NullFormatter())
fig.tight_layout()

print(f"{'kappa':>7}  {'m':>4}  {'alpha':>9}  converged")
for row in rows:
    print(f"{row[0]:>7.2f}  {row[1]:>4}  {row[2]:>9.3f}  {row[3]}")
if not ok.all():
    print("\n(non-converged points are left out of the plot: increase m_values to reach them)")

_style.finish(fig, "04_inversion_vs_kappa")
