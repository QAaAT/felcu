"""
The error-alpha trade-off controlled by lambda
==============================================

At fixed m, felcu chooses the largest lambda that still meets the target,
so sweeping target_error traces out how error and alpha depend on lambda:
more regularisation means a larger error but a smaller alpha, which
approaches the optimal alpha* = max |f| on the fitted domain (dotted).
Repeated for three orders m.  Needs matplotlib.

Run with:
    python examples/plots/02_error_alpha_vs_lambda.py [OUTPUT_DIR]
"""

import matplotlib.pyplot as plt
import numpy as np

import _style
from felcu import solve_fe_coefficients

M_ORDERS = (20, 40, 80)
TARGETS = np.geomspace(1e-12, 1e-3, 19)

_style.apply()
fig, (ax_err, ax_alpha) = plt.subplots(2, 1, sharex=True, figsize=(7, 6.5))

for m, color in zip(M_ORDERS, _style.ramp(len(M_ORDERS))):
    rows = []
    for target in TARGETS:
        r = solve_fe_coefficients("exponentiation", sigma_min=0.1, sigma_max=2.0, target_error=target, m=m)
        if r.converged and r.lambda_used > 0:
            rows.append((r.lambda_used, r.achieved_linf_error, r.alpha))
    lam, err, alpha = map(np.array, zip(*rows))

    ax_err.loglog(lam, err, marker="o", color=color, label=f"$m = {m}$")
    ax_alpha.semilogx(lam, alpha, marker="o", color=color)
    _style.label_end(ax_alpha, lam, alpha, f"$m = {m}$", at="start")

    print(f"\nm = {m}")
    print(f"  {'lambda':>9}  {'Linf':>9}  {'alpha':>9}")
    for row in rows:
        print(f"  {row[0]:>9.2e}  {row[1]:>9.2e}  {row[2]:>9.4f}")

alpha_star = _style.alpha_floor(r)
_style.ref_line(ax_alpha, alpha_star, _style.ALPHA_FLOOR_LABEL + f" $= {alpha_star:.3f}$", side="left")
print(f"\nalpha* = max|f| = {alpha_star:.4f}")

ax_alpha.set_xlim(left=ax_alpha.get_xlim()[0] / 10)  # room for the start labels
ax_err.set_title(r"Achieved $L^\infty$ error  ($e^{H}$, eigenvalues in $[0.1, 2]$)")
ax_err.legend(loc="upper left")
ax_alpha.set_title(r"LCU normalisation $\alpha$")
ax_alpha.set_xlabel(r"regularisation strength $\lambda$")
fig.tight_layout()
_style.finish(fig, "02_error_alpha_vs_lambda")
