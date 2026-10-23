"""
Comparing the three coefficient-selection methods
=================================================

The paper's numerical comparison (its Fig. 1): the identity, exponential,
square root and inverse, each fitted by the reflected extension ("refl"),
arcsine-Taylor ("asin") and the Sobolev-regularised extension ("reg").

Top: each series g_m for m = 128 over one whole period, mapped back to the
eigenvalue lambda = mu + x / tau so that all three share the target f, which
is drawn on the spectral interval (unshaded).  The methods agree there but
extend f differently outside it, over periods set by their own eta.
Middle: the error eps_m against m.  Bottom: alpha_m against m, with the
lower bound alpha* = ||f(H)|| (dotted) and, for the convex functions, the
arcsine-Taylor limit |f(mu +- eta delta / 2)| (dashed).  The reflected
extension sits just below alpha* but converges as O(1/m); arcsine-Taylor
converges exponentially towards a larger alpha; the regularised fit reaches
the target with alpha close to alpha*.  Sobolev alpha is shown for the
orders that meet the target.  Needs matplotlib.

Run with:
    python examples/plots/07_method_comparison.py [OUTPUT_DIR]
"""

import matplotlib.pyplot as plt
import numpy as np

import _style
from felcu import evaluate_fourier_series, solve_fe_coefficients

TARGET = 1e-10
M_VALUES = range(8, 257, 8)
M_SHOW = 128

# (problem, f, spectral interval, arcsine-Taylor eta, f convex).  eta = 3 for
# the entire functions; the others sit below their analyticity bounds
# 2 mu / delta, 46/17 ~ 2.71 for the square root and 3 for the inverse, as in
# the paper.  The arcsine-Taylor limit of alpha only holds for convex f.
CASES = [
    ("action", r"$\mathcal{H}$", lambda t: t, (-1.0, 2.0), 3.0, True),
    ("exponentiation", r"$e^{\mathcal{H}}$", np.exp, (-1.0, 1.0), 3.0, True),
    ("square_root", r"$\sqrt{\mathcal{H}}$", np.sqrt, (0.15, 1.0), 2.7, False),
    ("inversion", r"$\mathcal{H}^{-1}$", lambda t: 1.0 / t, (0.2, 1.0), 2.5, True),
]
METHODS = {
    "reflected": ("refl", "--"),
    "arcsine_taylor": ("asin", "-."),
    "sobolev": ("reg", "-"),
}

_style.apply()
fig, axes = plt.subplots(3, len(CASES), figsize=(4.2 * len(CASES), 9.5))
x_full = np.linspace(-np.pi, np.pi, 2001)

print(f"{'problem':<15} {'method':<15} {'eta':>5}  {'eps_128':>9}  {'alpha_128':>9}  {'alpha*':>9}")
for col, (problem, name, f, (lo, hi), eta_asin, convex) in enumerate(CASES):
    color = _style.PROBLEM_COLORS[problem]
    ax_fit, ax_err, ax_alpha = axes[:, col]
    mu, delta = (lo + hi) / 2, (hi - lo) / 2
    common = dict(shifted=True, sigma_min=lo, sigma_max=hi, target_error=TARGET, m_values=M_VALUES)

    for method, (short, linestyle) in METHODS.items():
        kwargs = common | ({"eta": eta_asin} if method == "arcsine_taylor" else {})
        r = solve_fe_coefficients(problem, method=method, **kwargs)
        sweep = r.all_results
        shown = next(c for c in sweep if c.m == M_SHOW)
        style = dict(color=color, linestyle=linestyle, label=f"{short} ($\\eta = {r.eta:.3g}$)")

        # Each method scales the spectrum by its own tau, so compare in lambda.
        lam = mu + x_full / shown.tau
        ax_fit.plot(lam, evaluate_fourier_series(x_full, shown.a0, shown.ak, shown.bk), **style)
        ax_err.semilogy([c.m for c in sweep], [c.achieved_linf_error for c in sweep], **style)
        keep = [c for c in sweep if method != "sobolev" or c.converged]
        ax_alpha.plot([c.m for c in keep], [c.alpha for c in keep], **style)
        print(
            f"{problem:<15} {method:<15} {r.eta:>5.3g}  {shown.achieved_linf_error:>9.1e}  "
            f"{shown.alpha:>9.4f}  {_style.alpha_floor(r):>9.4f}"
        )

    left, right = ax_fit.get_xlim()
    ax_fit.axvspan(left, lo, color=_style.SHADE, linewidth=0, zorder=0)
    ax_fit.axvspan(hi, right, color=_style.SHADE, linewidth=0, zorder=0)
    ax_fit.set_xlim(left, right)
    spectrum = np.linspace(lo, hi, 201)
    ax_fit.plot(spectrum, f(spectrum), color=_style.INK, linewidth=0.9, label="target $f$")

    _style.target_line(ax_err, TARGET)
    _style.ref_line(ax_alpha, _style.alpha_floor(r), _style.ALPHA_FLOOR_LABEL, side="left")
    if convex:
        asin_limit = max(abs(f(mu - eta_asin * delta / 2)), abs(f(mu + eta_asin * delta / 2)))
        _style.ref_line(ax_alpha, asin_limit, "asin limit", linestyle="--")

    ax_fit.set_title(f"{name}, $[{lo:g}, {hi:g}]$  ($m = {M_SHOW}$)")
    ax_fit.set_xlabel(r"eigenvalue $\lambda = \mu + x/\tau$")
    ax_err.set_title(r"error $\epsilon_m = \|g - g_m\|_{L^\infty(\Omega)}$")
    ax_err.set_ylim(1e-16, None)
    ax_alpha.set_title(r"subnormalisation $\alpha_m$")
    for ax in (ax_err, ax_alpha):
        ax.set_xlabel("truncation order $m$")
    if col == 0:
        ax_fit.legend(loc="upper left", fontsize=8)

fig.tight_layout()
_style.finish(fig, "07_method_comparison")
