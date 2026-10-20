"""
Alpha over the Sobolev parameters r and w
=========================================

The H_r^w regulariser weights the k-th harmonic by (1 + r|k|)^(2w).  At
fixed m and target, sweep 0 < r <= 1 and w > 0.5 and plot the resulting
LCU normalisation alpha, as a heatmap and as slices at four values of r.
Every fit here meets the target, so only alpha changes.  Needs matplotlib.

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


def alpha(r: float, w: float) -> float:
    return solve_fe_coefficients("exponentiation", sobolev_r=r, sobolev_w=w, **common).alpha


grid = np.array([[alpha(r, w) for r in R] for w in W])
i, j = np.unravel_index(np.argmin(grid), grid.shape)

_style.apply()
fig, (ax_map, ax_cut) = plt.subplots(1, 2, figsize=(11, 4.5))

mesh = ax_map.pcolormesh(R, W, grid, cmap=_style.SEQUENTIAL, shading="nearest")
fig.colorbar(mesh, ax=ax_map, label="alpha")
ax_map.grid(False)
ax_map.plot(R[j], W[i], marker="o", markersize=10, markerfacecolor="none",
            markeredgecolor=_style.INK, markeredgewidth=1.4, linestyle="none",
            label=f"minimum ({R[j]:.2f}, {W[i]:.2f})")
ax_map.plot(1.0, 1.0, marker="s", markersize=9, markerfacecolor="none",
            markeredgecolor=_style.INK, markeredgewidth=1.4, linestyle="none",
            label="default (1, 1)")
ax_map.legend(loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=2)
ax_map.set_xlabel("scale r")
ax_map.set_ylabel("order w")
ax_map.set_title("alpha  (exp, m = 40, target 1e-8)")

for r, color in zip(R_SLICES, _style.ramp(len(R_SLICES))):
    a = [alpha(r, w) for w in W]
    ax_cut.plot(W, a, color=color, label=f"r = {r}")
    _style.label_end(ax_cut, W, a, f"r = {r}")
ax_cut.set_xlabel("order w")
ax_cut.set_title("alpha versus w at fixed r")
ax_cut.legend(loc="upper left")

fig.tight_layout()

print(f"minimum alpha {grid[i, j]:.4f} at r={R[j]:.3f}, w={W[i]:.3f}")
print(f"default (r=1, w=1) alpha {alpha(1.0, 1.0):.4f}")
print("\nalpha  (rows: w, columns: r)")
print("   w \\ r " + " ".join(f"{r:>6.2f}" for r in R[::3]))
for w, row in zip(W[::3], grid[::3]):
    print(f"  {w:>5.2f}  " + " ".join(f"{a:>6.2f}" for a in row[::3]))

_style.finish(fig, "03_alpha_vs_r_w")
