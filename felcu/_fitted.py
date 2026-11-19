"""The fitted set, and how a candidate series is scored on it.

Shared by every method: the parts of the set in the scaled variable, the
validation grids, the error and subnormalisation of a series, and whether it
meets the target.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from felcu._coefficients import _basis_matrix, _spec_target, _unpack_coefficients
from felcu._linalg import _alpha, _validation_errors
from felcu._logging import logger
from felcu._problems import _canonical

_METRIC_ALIASES = {
    "l2": "l2",
    "2": "l2",
    "linf": "linf",
    "l_inf": "linf",
    "l-infinity": "linf",
    "infinity": "linf",
    "inf": "linf",
    "max": "linf",
}


def canonical_metric(name: str) -> str:
    """'l2' or 'linf', the error metrics a solve records."""
    return _canonical(name, _METRIC_ALIASES, "error_metric")


def _centre_and_halfwidth(spectral: dict) -> tuple[float, float]:
    """Centre mu and half-width delta of the fitted set in the spectral variable."""
    lo, hi = spectral["fit_lo"], spectral["fit_hi"]
    return 0.5 * (lo + hi), 0.5 * (hi - lo)


def _tolerance(target: float) -> float:
    """Slack allowed when comparing an achieved error with *target*."""
    return max(1e-15, abs(target) * 1e-12)


def _meets_target(solve: dict, metric: str, target: float) -> bool:
    """Whether a solve's error in *metric* ('l2' or 'linf') meets *target*."""
    return solve[metric] <= target + _tolerance(target)


def _validation_data(spec: dict, m: int, val_n: int) -> list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    """Validation grids on the fitted set: ``(x, basis matrix, target, L2 weights)`` per part.

    Each interval gets a uniform grid with trapezoid weights, and a point one
    node with the weight of :func:`_point_weight`.
    """

    def trapezoid(lo: float, hi: float, share: float) -> tuple[np.ndarray, np.ndarray]:
        xv = np.linspace(lo, hi, max(int(val_n * share), 201))
        wv = np.full(xv.size, xv[1] - xv[0])
        wv[[0, -1]] /= 2
        return xv, wv

    return [
        (xv, _basis_matrix(xv, m, spec["basis"]), _spec_target(xv, spec), wv) for xv, wv in _part_rules(spec, trapezoid)
    ]


def _fitted_parts(spec: dict) -> list[tuple[float, float]]:
    """The parts of the fitted set.  A side of a two-part set is a point when the gap reaches the spectrum's end."""
    x_max, gap = spec["x_max"], spec["x_gap"]
    if gap is None:
        return [(-x_max, x_max)]
    return [(-x_max, gap[0]), (gap[1], x_max)]


def _part_rules(
    spec: dict, interval_rule: Callable[[float, float, float], tuple[np.ndarray, np.ndarray]]
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Nodes and weights ``(x, w)`` for each part of the fitted set.

    An interval gets ``interval_rule(lo, hi, share)``, where *share* is its
    fraction of the set's length; a point gets one node with the weight of
    :func:`_point_weight`.
    """
    parts = _fitted_parts(spec)
    length = sum(hi - lo for lo, hi in parts)
    return [
        (np.array([lo]), np.array([_point_weight(length)])) if lo == hi else interval_rule(lo, hi, (hi - lo) / length)
        for lo, hi in parts
    ]


def _point_weight(length: float) -> float:
    """Weight of a point of the fitted set, in the fit and the L2 error.

    A point has no length, so as a measure it would be ignored.  It counts as
    much as the whole set instead, or 1 when the set is only points.
    """
    return length or 1.0


def _summarise(coeffs: np.ndarray, val: list, basis: str, m: int) -> dict:
    """The solve record for packed coefficients: the series, its errors on *val*, and alpha."""
    a0, ak, bk = _unpack_coefficients(coeffs, basis, m)
    l2, linf = _validation_errors(coeffs, val)
    alpha = _alpha(a0, ak, bk)
    logger.debug("    solve done: l2=%.3e  linf=%.3e  alpha=%.3e", l2, linf, alpha)
    return {"a0": a0, "ak": ak, "bk": bk, "l2": l2, "linf": linf, "alpha": alpha}
