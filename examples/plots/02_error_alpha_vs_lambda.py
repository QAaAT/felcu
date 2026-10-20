"""
The error-alpha trade-off controlled by lambda
==============================================

At fixed m, felcu chooses the largest regularisation weight gamma that
still meets the target eps_t, so sweeping target_error traces out how the
error eps_m and subnormalisation alpha_m depend on lambda = sqrt(2 pi gamma):
more regularisation means a larger error but a smaller alpha_m, which
approaches the lower bound alpha* = ||g||_inf on the fitted set (dotted).
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
print(f"\nalpha* = ||g||_inf = {alpha_star:.4f}")

ax_alpha.set_xlim(left=ax_alpha.get_xlim()[0] / 10)  # room for the start labels
ax_err.set_title(r"Achieved error $\epsilon_m$  ($e^{\mathcal{H}}$, $\mathrm{spec}(\mathcal{H}) \subseteq [0.1, 2]$)")
ax_err.legend(loc="upper left")
ax_alpha.set_title(r"Subnormalisation $\alpha_m$")
ax_alpha.set_xlabel(r"regularisation strength $\sqrt{2\pi\gamma}$")
fig.tight_layout()
_style.finish(fig, "02_error_alpha_vs_lambda")
