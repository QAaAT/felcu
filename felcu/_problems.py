"""The problems felcu fits, one record each.

Everything that depends on which function f is being fitted is defined here:
f and its derivatives, its power series for the arcsine-Taylor construction,
its names, where it is defined, and its default Sobolev scale.  Adding a
function means adding a record to :data:`PROBLEMS`.  The design assumes f is
entire or has its one singularity, a pole or a branch point, at 0.

A leaf module: it imports nothing from felcu, so every other module can use it.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

import numpy as np

# ---------------------------------------------------------------------------
# Power series, for the arcsine-Taylor construction
# ---------------------------------------------------------------------------
#
# Each maps the Taylor coefficients of u(z) to those of F(u(z)), truncated
# to the same order.


def _series_exp(u: np.ndarray) -> np.ndarray:
    """exp(u) as a power series, from y' = u' y."""
    y = np.zeros_like(u)
    y[0] = math.exp(u[0])
    ju = np.arange(u.size) * u
    for k in range(1, u.size):
        y[k] = (ju[1 : k + 1] @ y[k - 1 :: -1]) / k
    return y


def _series_reciprocal(u: np.ndarray) -> np.ndarray:
    """1 / u as a power series, from u y = 1."""
    y = np.zeros_like(u)
    y[0] = 1.0 / u[0]
    for k in range(1, u.size):
        y[k] = -(u[1 : k + 1] @ y[k - 1 :: -1]) / u[0]
    return y


def _series_sqrt(u: np.ndarray) -> np.ndarray:
    """sqrt(u) as a power series, from y^2 = u."""
    y = np.zeros_like(u)
    y[0] = math.sqrt(u[0])
    for k in range(1, u.size):
        y[k] = (u[k] - y[1:k] @ y[k - 1 : 0 : -1]) / (2.0 * y[0])
    return y


def _series_log(u: np.ndarray) -> np.ndarray:
    """log(u) as a power series, from u y' = u'."""
    y = np.zeros_like(u)
    y[0] = math.log(u[0])
    j = np.arange(u.size)
    for k in range(1, u.size):
        y[k] = (k * u[k] - (j[1:k] * y[1:k]) @ u[k - 1 : 0 : -1]) / (k * u[0])
    return y


# ---------------------------------------------------------------------------
# Derivatives F^(d)(t)
# ---------------------------------------------------------------------------


def _identity_d(t: np.ndarray, d: int) -> np.ndarray:
    return t if d == 0 else np.full_like(t, 1.0 if d == 1 else 0.0)


def _reciprocal_d(t: np.ndarray, d: int) -> np.ndarray:
    return (-1.0) ** d * math.factorial(d) / t ** (d + 1)


def _exp_d(t: np.ndarray, d: int) -> np.ndarray:
    return np.exp(t)


def _sqrt_d(t: np.ndarray, d: int) -> np.ndarray:
    # (1/2)(1/2 - 1)...(1/2 - d + 1) t^(1/2 - d)
    return math.prod(0.5 - j for j in range(d)) * t ** (0.5 - d)


def _log_d(t: np.ndarray, d: int) -> np.ndarray:
    # (-1)^(d-1) (d-1)! t^(-d) for d >= 1
    return np.log(t) if d == 0 else (-1.0) ** (d - 1) * math.factorial(d - 1) / t**d


def _reciprocal_target(x: np.ndarray, tau: float, mu: float, d: int) -> np.ndarray:
    """g^(d)(x) for 1/t, written without dividing x by tau, which rounds near the pole."""
    return tau * (-1.0) ** d * math.factorial(d) / (x + tau * mu) ** (d + 1)


# ---------------------------------------------------------------------------
# Default Sobolev scales r (Sec. VI of the paper); see default_sobolev_r
# ---------------------------------------------------------------------------


def _exp_r(spectral: dict, tau: float, w: float) -> float:
    # g' = g / tau
    return 1.0 / (1.0 + 1.0 / tau)


def _sqrt_r(spectral: dict, tau: float, w: float) -> float:
    kappa = spectral["sigma_max"] / spectral["sigma_min"]
    return kappa ** (-(2.0 * w - 1.0) / (2.0 * w))


def _log_r(spectral: dict, tau: float, w: float) -> float:
    kappa = spectral["sigma_max"] / spectral["sigma_min"]
    return _sqrt_r(spectral, tau, w) * (1.0 + math.log(kappa)) ** (-1.0 / w)


def _reciprocal_r(spectral: dict, tau: float, w: float) -> float:
    # The distance from the fitted set to the pole at 0.
    lo, hi, gap = spectral["fit_lo"], spectral["fit_hi"], spectral["fit_gap"]
    nearest = min(abs(lo), abs(hi)) if gap is None else min(-gap[0], gap[1])
    return min(1.0, tau * nearest)


# ---------------------------------------------------------------------------
# The problems
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Problem:
    """One function f that felcu can fit."""

    name: str
    aliases: tuple[str, ...]
    formula: str
    """f written out, for messages."""
    derivative: Callable[[np.ndarray, int], np.ndarray]
    """``F^(d)(t)``."""
    series: Callable[[np.ndarray], np.ndarray]
    """The power series of ``F(u(z))`` from that of ``u(z)``."""
    sobolev_r: float | Callable[[dict, float, float], float]
    """The default Sobolev scale, or its rule ``(spectral, tau, w) -> r``."""
    singularity: Literal["none", "pole", "branch"] = "none"
    """f is entire, or has a pole or a branch point at 0.

    A pole needs a nonsingular matrix, and an indefinite spectrum is fitted on
    two intervals; a branch point restricts f to a positive-definite spectrum.
    """
    odd: bool = False
    """Odd, so it can act on singular values through the Hermitian dilation."""
    affine: bool = False
    """``g - g(0)`` is odd, so the series needs no cosines."""
    target: Callable[[np.ndarray, float, float, int], np.ndarray] | None = None
    """``g^(d)(x)`` for finite tau, when it is more accurate than scaling :attr:`derivative`."""

    @property
    def entire(self) -> bool:
        return self.singularity == "none"

    @property
    def pole_at_zero(self) -> bool:
        return self.singularity == "pole"

    @property
    def positive_definite(self) -> bool:
        return self.singularity == "branch"

    def default_sobolev_r(self, spectral: dict, tau: float, sobolev_w: float) -> float:
        """The default Sobolev scale r for a fit of *spectral* with scale *tau*."""
        rule = self.sobolev_r
        return rule(spectral, tau, sobolev_w) if callable(rule) else rule


PROBLEMS: dict[str, Problem] = {
    p.name: p
    for p in (
        Problem(
            "identity",
            ("action", "matrix-action"),
            "x",
            _identity_d,
            lambda u: u,
            sobolev_r=1.0,  # g is linear
            odd=True,
            affine=True,
        ),
        Problem(
            "inversion",
            ("inverse",),
            "1/x",
            _reciprocal_d,
            _series_reciprocal,
            sobolev_r=_reciprocal_r,
            singularity="pole",
            odd=True,
            target=_reciprocal_target,
        ),
        Problem(
            "exponentiation",
            ("exp", "exponential"),
            "exp(x)",
            _exp_d,
            _series_exp,
            sobolev_r=_exp_r,
        ),
        Problem(
            "square_root",
            ("square-root", "sqrt"),
            "sqrt(x)",
            _sqrt_d,
            _series_sqrt,
            sobolev_r=_sqrt_r,
            singularity="branch",
        ),
        Problem(
            "logarithm",
            ("log", "ln"),
            "log(x)",
            _log_d,
            _series_log,
            sobolev_r=_log_r,
            singularity="branch",
        ),
    )
}


# ---------------------------------------------------------------------------
# Names, aliases, and name lists for messages
# ---------------------------------------------------------------------------

_PROBLEM_ALIASES = {alias: p.name for p in PROBLEMS.values() for alias in (p.name, *p.aliases)}


def _listing(names, conjunction: str = "or", fmt: str = "'{}'") -> str:
    """``'a'``, ``'a' or 'b'``, or ``'a', 'b', or 'c'``, with each name written as *fmt*."""
    quoted = [fmt.format(n) for n in names]
    if len(quoted) <= 2:
        return f" {conjunction} ".join(quoted)
    return ", ".join(quoted[:-1]) + f", {conjunction} " + quoted[-1]


def _canonical(name: str, aliases: dict[str, str], what: str) -> str:
    """The canonical name for *name* in *aliases*, ignoring case and surrounding whitespace."""
    key = name.strip().lower()
    if key not in aliases:
        raise ValueError(f"{what} must be {_listing(dict.fromkeys(aliases.values()))}.")
    return aliases[key]


def canonical_problem(name: str) -> str:
    return _canonical(name, _PROBLEM_ALIASES, "problem")
