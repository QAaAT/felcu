"""
The fitted series over a whole period, and its pointwise error
==============================================================

Top: the series g over the full period [-pi, pi] against the target f on
the fitted domain.  The shaded regions are not fitted: the extension
region, and for unshifted inversion also the gap around the pole, where g
stays bounded.  Bottom: |g - f| on the fitted domain.  Needs matplotlib.

Run with:
    python examples/plots/05_pointwise_error.py [OUTPUT_DIR]
"""

import matplotlib.pyplot as plt
import numpy as np

import _style
from felcu import (
    evaluate_fourier_series,
    evaluate_target_function,
    fitted_domain_segments,
    solve_fe_coefficients,
)

TARGET = 1e-8
CASES = [
    ("exponentiation", dict(sigma_min=0.1, sigma_max=2.0, m_values=range(4, 33, 2))),
    ("inversion", dict(sigma_min=0.5, sigma_max=3.0, m_values=range(4, 65, 4))),
]

_style.apply()
fig, axes = plt.subplots(2, 2, figsize=(11, 6.5), sharex=True)
x_full = np.linspace(-np.pi, np.pi, 2001)

for col, (problem, kwargs) in enumerate(CASES):
    r = solve_fe_coefficients(problem, target_error=TARGET, **kwargs)
    color = _style.PROBLEM_COLORS[problem]
    ax_fit, ax_err = axes[0, col], axes[1, col]
    segments = fitted_domain_segments(r)

    # Shade everything outside the fitted segments
    edges = [-np.pi] + [v for s in segments for v in (s[0], s[-1])] + [np.pi]
    for ax in (ax_fit, ax_err):
        for left, right in zip(edges[::2], edges[1::2]):
            ax.axvspan(left, right, color=_style.SHADE, linewidth=0)

    g = evaluate_fourier_series(x_full, r.a0, r.ak, r.bk)
    ax_fit.plot(x_full, g, color=color, label="series $g$")
    worst = 0.0
    for i, xs in enumerate(segments):
        f = evaluate_target_function(xs, r)
        err = np.abs(evaluate_fourier_series(xs, r.a0, r.ak, r.bk) - f)
        worst = max(worst, err.max())
        ax_fit.plot(xs, f, color=_style.INK, linewidth=0.9, label="target $f$" if i == 0 else None)
        ax_err.semilogy(xs, err, color=color)

    _style.target_line(ax_err, TARGET)
    ax_fit.set_title(f"{problem}  ($m = {r.m}$; shaded = not fitted)")
    ax_fit.legend(loc="upper left")
    ax_err.set_title(r"pointwise error $|g(x) - f(x)|$ on the fitted domain")
    ax_err.set_xlabel(r"scaled variable $x$")
    print(f"{problem}: m={r.m}  domain={r.domain}  max |g - f| = {worst:.2e}  (target {TARGET:g})")

fig.tight_layout()
_style.finish(fig, "05_pointwise_error")
