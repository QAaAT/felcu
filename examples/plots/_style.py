"""Shared figure style for the plotting examples.

Each script saves its figure as ``<name>.png`` in the directory given as its
first argument (default: the current directory), and also shows it when an
interactive matplotlib backend is available.
"""

import shutil
import sys
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

from felcu import evaluate_target_function, fitted_domain_segments

# Render text with LaTeX when it is installed; otherwise fall back to
# matplotlib's built-in Computer Modern mathtext, which looks much the same.
USETEX = shutil.which("latex") is not None

# Surfaces and ink
SURFACE = "#fcfcfb"
SHADE = "#f0efec"   # unfitted regions of the period
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"

# One fixed colour per problem, the same in every figure
PROBLEM_COLORS = {
    "action": "#2a78d6",
    "inversion": "#eb6834",
    "exponentiation": "#1baf7a",
}

# Single-hue blue steps, light to dark
_BLUE = ["#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7", "#3987e5",
         "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b"]
SEQUENTIAL = LinearSegmentedColormap.from_list("felcu_blue", _BLUE)


def ramp(n: int) -> list[str]:
    """n ordered colours for ordered parameter values (steps 250 to 600)."""
    steps = _BLUE[3:11]
    return [steps[i] for i in np.linspace(0, len(steps) - 1, n).round().astype(int)]


def apply() -> None:
    plt.rcParams.update({
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "axes.edgecolor": AXIS,
        "axes.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "axes.axisbelow": True,
        "axes.labelcolor": INK_2,
        "axes.titlecolor": INK,
        "axes.titlesize": 11,
        "axes.titlelocation": "left",
        "grid.color": GRID,
        "grid.linewidth": 0.6,
        "grid.linestyle": "-",
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "xtick.labelcolor": INK_2,
        "ytick.labelcolor": INK_2,
        "text.color": INK,
        "lines.linewidth": 1.6,
        "lines.markersize": 4.5,
        "legend.frameon": False,
        "legend.labelcolor": INK_2,
        "font.size": 10,
        "text.usetex": USETEX,
        "text.latex.preamble": r"\usepackage{amsmath}",
        "font.family": "serif",
        "font.serif": ["Computer Modern Roman", "CMU Serif", "DejaVu Serif"],
        "mathtext.fontset": "cm",
        "axes.formatter.use_mathtext": True,
    })


def alpha_floor(result) -> float:
    """max |f| over the fitted domain: the smallest LCU normalisation possible.

    Every series satisfies |g(x)| <= alpha, so a fit with error eps needs
    alpha >= max |f| - eps.
    """
    return max(float(np.max(np.abs(evaluate_target_function(xs, result))))
               for xs in fitted_domain_segments(result))


ALPHA_FLOOR_LABEL = r"$\alpha^\ast = \max_{x \in D} |f(x)|$"


def label_end(ax, x, y, text: str, at: str = "end", gap: float = 5) -> None:
    """Direct-label a line at its last (or first) point, in secondary ink."""
    i, dx, ha = (-1, gap, "left") if at == "end" else (0, -gap, "right")
    ax.annotate(text, (x[i], y[i]), xytext=(dx, 0), textcoords="offset points",
                color=INK_2, va="center", ha=ha, fontsize=9)


def ref_line(ax, y: float, text: str = "", color: str = INK_2, linestyle: str = ":",
             side: str = "right") -> None:
    """A thin horizontal reference line, optionally labelled at one end."""
    ax.axhline(y, color=color, linewidth=1, linestyle=linestyle)
    if text:
        x, dx = (1, -2) if side == "right" else (0, 2)
        ax.annotate(text, (x, y), xycoords=("axes fraction", "data"),
                    xytext=(dx, 4), textcoords="offset points", ha=side,
                    color=INK_2, fontsize=9)


def target_line(ax, target: float) -> None:
    ref_line(ax, target, "target", linestyle="--")


def finish(fig, name: str) -> None:
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(".")
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{name}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"\nsaved {path}")
    if matplotlib.get_backend().lower() != "agg":
        plt.show()
