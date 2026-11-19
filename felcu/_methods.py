"""The coefficient-selection methods, one record each.

Everything that depends on the method is defined here: the options it takes
and their defaults, its eta rules, whether it can fit two intervals, and how
it builds the candidate series of each order.  Adding a method means adding
a record to :data:`METHODS`.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from functools import partial
from types import MappingProxyType

import numpy as np

from felcu._explicit import _arcsine_taylor_coefficients, _explicit_candidate, _reflected_coefficients
from felcu._fitted import _centre_and_halfwidth
from felcu._problems import PROBLEMS, Problem, _canonical, _listing
from felcu._solver import _candidate_for_m

# Default eta of the arcsine-Taylor construction for an entire f (Sec. VI).
ARCSINE_TAYLOR_ETA = 3.0


@dataclass(frozen=True)
class Method:
    """One way of choosing the coefficients of each order."""

    name: str
    aliases: tuple[str, ...]
    candidate: Callable[[dict, int, float, str, int], dict]
    """``(spec, m, target, metric, val_n) -> {"m", "solve", "fields"?}``, the series of order *m*.

    ``fields`` holds the ``method_details`` that vary with the order, such as
    the lambda a regularised fit used.
    """
    default_eta: Callable[[Problem, dict | None], float]
    """The default eta for a problem on a spectrum (``None`` only for an entire f)."""
    options: Mapping[str, object] = field(default_factory=dict)
    """The options it accepts, beyond the common ones, and their defaults."""
    prepare: Callable[[dict, dict], dict] = lambda spec, options: {}
    """Validates the options and returns what to add to the spec, ``(spec, options) -> dict``."""
    result_fields: Callable[[dict], dict[str, float]] = lambda spec: {}
    """``method_details`` of the result fixed for the whole fit, from the prepared spec."""
    constant_fields: Mapping[str, float] = field(default_factory=dict)
    """``method_details`` of the exact constant series of a point spectrum, beyond :attr:`result_fields`."""
    eta_min: tuple[float, bool] | None = None
    """``(bound, inclusive)``: a lower bound on eta beyond eta > 1."""
    explicit: bool = False
    """It builds g on [-pi/2, pi/2] in closed form.  That must stay clear of the
    singularity of f, so it cannot fit two intervals either side of a pole."""


# ---------------------------------------------------------------------------
# Sobolev-regularised least squares
# ---------------------------------------------------------------------------


def _sobolev_eta(problem: Problem, spectral: dict | None) -> float:
    if problem.entire:
        return 2.0
    mu, delta = _centre_and_halfwidth(spectral)
    gap = spectral["fit_gap"]
    if gap is not None:
        # The gap around the pole and the free part of the period are equally wide.
        return 1.0 + (gap[1] - gap[0]) / (2.0 * delta)
    # The pole or branch point at 0 lies at x = -tau mu, on the periodic
    # boundary when eta = |mu| / delta.
    return 2.0 if delta == 0 else abs(mu) / delta


def default_lambda_grid(
    lambda_max: float = 1.0,
    lambda_min: float = 1e-16,
    count: int = 12,
) -> np.ndarray:
    """Return a log-spaced grid of regularisation parameters.

    This is the default ``lambdas`` for
    :func:`~felcu.solve_fe_coefficients`.  The solver sorts the grid
    itself and searches beyond it, so the grid only needs to bracket the
    answer roughly.

    Parameters
    ----------
    lambda_max, lambda_min:
        Largest and smallest value.
    count:
        Number of values.

    Returns
    -------
    numpy.ndarray
        *count* geometrically spaced values from *lambda_max* down to
        *lambda_min*.

    Raises
    ------
    ValueError
        Unless ``0 < lambda_min <= lambda_max`` and ``count >= 2``.
    """
    if lambda_min <= 0 or lambda_max <= 0 or lambda_min > lambda_max or count < 2:
        raise ValueError("Require 0 < lambda_min <= lambda_max and count >= 2.")
    return np.geomspace(lambda_max, lambda_min, count)


def _lambda_grid(lambdas) -> np.ndarray:
    if lambdas is None:
        return default_lambda_grid()
    grid = np.asarray(lambdas, dtype=float)
    if grid.ndim != 1 or grid.size == 0 or not np.all(grid >= 0):
        raise ValueError("lambdas must be a non-empty sequence of non-negative values.")
    return grid


def _sobolev_prepare(spec: dict, options: dict) -> dict:
    w, order_fit, fit_oversample = options["sobolev_w"], options["order_fit"], options["fit_oversample"]
    if not (order_fit >= 0 and float(order_fit).is_integer()) or not fit_oversample > 0:
        raise ValueError("Require order_fit to be a non-negative integer and fit_oversample > 0.")
    if not w > 0.5:
        raise ValueError("sobolev_w must exceed 0.5.")
    r = options["sobolev_r"]
    if r is None:
        r = PROBLEMS[spec["problem"]].option_default("sobolev_r", spec["spectral"], spec["tau"], options)
    if not 0 < r <= 1:
        raise ValueError("sobolev_r must satisfy 0 < sobolev_r <= 1.")
    return {
        "sobolev_r": float(r),
        "sobolev_w": float(w),
        "order_fit": int(order_fit),
        "fit_oversample": fit_oversample,
        "lambdas": _lambda_grid(options["lambdas"]),
    }


# ---------------------------------------------------------------------------
# Explicit extensions
# ---------------------------------------------------------------------------


def _arcsine_taylor_eta(problem: Problem, spectral: dict | None) -> float:
    if problem.entire:
        return ARCSINE_TAYLOR_ETA
    mu, delta = _centre_and_halfwidth(spectral)
    return ARCSINE_TAYLOR_ETA if delta == 0 else min(ARCSINE_TAYLOR_ETA, 1.0 + abs(mu) / delta)


METHODS: dict[str, Method] = {
    m.name: m
    for m in (
        Method(
            "sobolev",
            ("regularised", "regularized", "reg"),
            _candidate_for_m,
            _sobolev_eta,
            options=MappingProxyType(
                {"lambdas": None, "sobolev_w": 1.0, "sobolev_r": None, "order_fit": 1, "fit_oversample": 8}
            ),
            prepare=_sobolev_prepare,
            result_fields=lambda spec: {"sobolev_r": spec["sobolev_r"], "sobolev_w": spec["sobolev_w"]},
            constant_fields=MappingProxyType({"lambda_used": 0.0}),
        ),
        Method(
            "reflected",
            ("refl", "reflected_extension", "reflected-extension"),
            partial(_explicit_candidate, _reflected_coefficients),
            lambda problem, spectral: 2.0,
            eta_min=(2.0, True),
            explicit=True,
        ),
        Method(
            "arcsine_taylor",
            ("arcsine-taylor", "arcsine", "asin"),
            partial(_explicit_candidate, _arcsine_taylor_coefficients),
            _arcsine_taylor_eta,
            eta_min=(2.0, False),
            explicit=True,
        ),
    )
}


# Every method option, each a keyword argument of solve_fe_coefficients.
OPTION_NAMES = tuple(dict.fromkeys(name for m in METHODS.values() for name in m.options))

_METHOD_ALIASES = {alias: m.name for m in METHODS.values() for alias in (m.name, *m.aliases)}


def canonical_method(name: str) -> str:
    return _canonical(name, _METHOD_ALIASES, "method")


# ---------------------------------------------------------------------------
# Checks shared by every method
# ---------------------------------------------------------------------------


def _check_supports(method: str, problem: str, spectral: dict) -> None:
    if METHODS[method].explicit and spectral["fit_gap"] is not None:
        able = _listing((m.name for m in METHODS.values() if not m.explicit), fmt="method='{}'")
        raise ValueError(
            f"method='{method}' does not support a two-interval fit ({problem} on an indefinite interval or on "
            f"singular values): the pole lies between the intervals.  Use {able}."
        )


def _check_eta(spec: dict) -> None:
    """Check that *eta* suits the method of *spec*.

    A method that evaluates g on [-pi/2, pi/2], i.e. f on
    [mu - eta delta/2, mu + eta delta/2], must keep that clear of the pole or
    branch point at 0.
    """
    method, eta = METHODS[spec["method"]], spec["eta"]
    if method.eta_min is not None:
        bound, inclusive = method.eta_min
        if eta < bound or (eta == bound and not inclusive):
            raise ValueError(f"method='{method.name}' requires eta {'>=' if inclusive else '>'} {bound:g}.")
    if not method.explicit or PROBLEMS[spec["problem"]].entire or not math.isfinite(spec["tau"]):
        return
    # Compare eta delta with 2 |mu| directly: recovering delta from tau, or
    # dividing by it, rounds and can admit eta = 2 |mu| / delta itself.
    mu, delta = _centre_and_halfwidth(spec["spectral"])
    if eta * delta >= 2.0 * abs(mu):
        raise ValueError(
            f"method='{method.name}' with eta={eta:.6g} extends {spec['problem']} past its singularity at 0; "
            f"eta must be below 2|mu|/delta = {2.0 * abs(mu) / delta:.6g}."
        )


def _method_options(method: str, given: dict) -> dict:
    """The options of *method*, with defaults for those not *given*; options of other methods are rejected."""
    accepted = METHODS[method].options
    foreign = [name for name, value in given.items() if value is not None and name not in accepted]
    if foreign:
        owners = [m.name for m in METHODS.values() if all(name in m.options for name in foreign)]
        owned = _listing(owners, fmt="method='{}'") or "no method"
        raise ValueError(f"{', '.join(foreign)} only apply to {owned}, not method='{method}'.")
    return {name: default if given.get(name) is None else given[name] for name, default in accepted.items()}
