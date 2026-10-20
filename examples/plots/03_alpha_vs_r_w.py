"""
Alpha over the Sobolev parameters r and w
=========================================

The H_r^w regulariser weights the k-th Fourier mode by 1 + (r|k|)^(2w).
At fixed m and target, sweep 0 < r <= 1 and w > 0.5 and plot the resulting
subnormalisation alpha_m, as a heatmap and as slices at four values of r,
against the lower bound alpha* = ||g||_inf on the fitted set (dotted).
Every fit here meets the target, so only alpha_m changes.  Needs matplotlib.

Run with:
    python examples/plots/03_alpha_vs_r_w.py [OUTPUT_DIR]
"""

import matplotlib.pyplot as plt
import numpy as np

import _style
from felcu import solve_fe_coefficients

R = np.linspace(0.05, 1.0, 16)
W = np.linspace(0.55, 3.0, 16)
R_SLICES = (0.1, 0.25, 0.5, 1.0)
common = dict(sigma_min=0.1, sigma_max=2.0, target_error=1e-8, m=40)


def solve(r: float, w: float):
    return solve_fe_coefficients("exponentiation", sobolev_r=r, sobolev_w=w, **common)


grid = np.array([[solve(r, w).alpha for r in R] for w in W])
i, j = np.unravel_index(np.argmin(grid), grid.shape)
default = solve_fe_coefficients("exponentiation", **common)  # r from default_sobolev_r, w = 1
alpha_star = _style.alpha_floor(default)

_style.apply()
fig, (ax_map, ax_cut) = plt.subplots(1, 2, figsize=(11, 4.5))

mesh = ax_map.pcolormesh(R, W, grid, cmap=_style.SEQUENTIAL, shading="nearest")
fig.colorbar(mesh, ax=ax_map, label=r"$\alpha_m$")
ax_map.grid(False)
ax_map.plot(
    R[j],
    W[i],
    marker="o",
    markersize=10,
    markerfacecolor="none",
    markeredgecolor=_style.INK,
    markeredgewidth=1.4,
    linestyle="none",
    label=f"minimum $({R[j]:.2f}, {W[i]:.2f})$",
)
ax_map.plot(
    default.sobolev_r,
    default.sobolev_w,
    marker="s",
    markersize=9,
    markerfacecolor="none",
    markeredgecolor=_style.INK,
    markeredgewidth=1.4,
    linestyle="none",
    label=f"default $({default.sobolev_r:.2f}, {default.sobolev_w:.0f})$",
)
ax_map.legend(loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=2)
ax_map.set_xlabel(r"scale $r$")
ax_map.set_ylabel(r"order $w$")
ax_map.set_title(r"$\alpha_m$  ($e^{\mathcal{H}}$, $m = 40$, $\epsilon_{\mathrm{t}} = 10^{-8}$)")

for r, color in zip(R_SLICES, _style.ramp(len(R_SLICES))):
    a = [solve(r, w).alpha for w in W]
    ax_cut.plot(W, a, color=color, label=f"$r = {r}$")
_style.ref_line(ax_cut, alpha_star, _style.ALPHA_FLOOR_LABEL + f" $= {alpha_star:.3f}$")
ax_cut.set_xlabel(r"order $w$")
ax_cut.set_title(r"$\alpha_m$ versus $w$ at fixed $r$")
ax_cut.legend(loc="upper left")

fig.tight_layout()

print(f"minimum alpha {grid[i, j]:.4f} at r={R[j]:.3f}, w={W[i]:.3f}")
print(f"default (r={default.sobolev_r:.3f}, w=1) alpha {default.alpha:.4f}")
print(f"alpha* = ||g||_inf = {alpha_star:.4f}")
print("\nalpha  (rows: w, columns: r)")
print("   w \\ r " + " ".join(f"{r:>6.2f}" for r in R[::3]))
for w, row in zip(W[::3], grid[::3]):
    print(f"  {w:>5.2f}  " + " ".join(f"{a:>6.2f}" for a in row[::3]))

_style.finish(fig, "03_alpha_vs_r_w")
