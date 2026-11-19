"""Core Fourier-extension fitting and candidate-selection logic for felcu."""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence

import numpy as np

from felcu._coefficients import (
    _add_baseline,
    _basis_matrices,
    _basis_matrix,
    _basis_size,
    _spec_target,
    _unpack_coefficients,
)
from felcu._linalg import (
    _alpha,
    _gauss_interval,
    _lstsq,
    _penalty,
    _ridge_qr,
    _validation_errors,
)
from felcu._logging import logger


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


def _make_fit_data(spec: dict, m: int, fit_oversample: int, val_n: int) -> dict:
    basis = spec["basis"]
    ncols = _basis_size(basis, m)

    # Quadrature nodes are shared between the intervals in proportion to their lengths.
    floor = 48 if spec["x_gap"] is None else 32
    nodes = fit_oversample * max(ncols, 1)
    rules = _part_rules(spec, lambda lo, hi, share: _gauss_interval(lo, hi, max(math.ceil(nodes * share), floor)))
    logger.debug(
        "  fit_data: x_gap=%s  basis=%s  m=%d  ncols=%d  nq=%s  val_n=%d",
        spec["x_gap"],
        basis,
        m,
        ncols,
        [xi.size for xi, _ in rules],
        val_n,
    )
    x = np.concatenate([xi for xi, _ in rules])
    weights = np.concatenate([wi for _, wi in rules])

    return {"x": x, "sw": np.sqrt(weights), "val": _validation_data(spec, m, val_n)}


def _fit_system(data: dict, spec: dict, m: int, order_fit: int) -> tuple[np.ndarray, np.ndarray]:
    x, sw = data["x"], data["sw"]
    A = np.vstack(_basis_matrices(x, m, spec["basis"], order_fit)) * np.tile(sw, order_fit + 1)[:, None]
    # The constant baseline is fitted exactly, so only the remainder is fitted here.
    b_blocks = [(_spec_target(x, spec, d) - (spec["baseline"] if d == 0 else 0.0)) * sw for d in range(order_fit + 1)]
    return A, np.concatenate(b_blocks)


def _finish_solve(residual_coeffs: np.ndarray, val: list, spec: dict, m: int) -> dict:
    coeffs = _add_baseline(residual_coeffs, spec["basis"], spec["baseline"])
    return _summarise(coeffs, val, spec["basis"], m)


def _summarise(coeffs: np.ndarray, val: list, basis: str, m: int) -> dict:
    """The solve record for packed coefficients: the series, its errors on *val*, and alpha."""
    a0, ak, bk = _unpack_coefficients(coeffs, basis, m)
    l2, linf = _validation_errors(coeffs, val)
    alpha = _alpha(a0, ak, bk)
    logger.debug("    solve done: l2=%.3e  linf=%.3e  alpha=%.3e", l2, linf, alpha)
    return {"a0": a0, "ak": ak, "bk": bk, "l2": l2, "linf": linf, "alpha": alpha}


def _make_linear_solver(
    data: dict,
    spec: dict,
    m: int,
    order_fit: int,
) -> Callable[[float], dict]:
    """Return ``solve(lam)`` for one m, building the fit system only once.

    The penalty is ``lam * D`` with ``D`` diagonal and invertible, so
    substituting ``y = D c`` turns ``min ||A c - b||^2 + lam^2 ||D c||^2``
    into ridge regression on ``A D^{-1}``.  One SVD of that matrix then
    solves every ``lam > 0`` cheaply.  ``lam = 0`` keeps the plain
    least-squares solve on ``A``.

    LAPACK's SVD can fail to converge on these near-singular systems, depending
    on the BLAS build and the CPU it runs on.  If it does, every ``lam > 0`` is
    solved by QR instead, which is slower but always succeeds.
    """
    basis, val = spec["basis"], data["val"]
    ncols = _basis_size(basis, m)
    if ncols == 0:
        return lambda lam: _finish_solve(np.zeros(0), val, spec, m)

    A, b = _fit_system(data, spec, m, order_fit)
    logger.debug("    fit system shape: %s", A.shape)
    d = _penalty(m, basis, 1.0, spec["sobolev_r"], spec["sobolev_w"])
    svd: dict[str, np.ndarray] = {}

    def solve(lam: float) -> dict:
        logger.debug("    solve: m=%d  lam=%.3e", m, lam)
        if lam == 0.0:
            return _finish_solve(_lstsq(A, b), val, spec, m)
        if not svd:
            # Factorise lazily: an m that is infeasible at lam = 0 never needs it.
            Ad = A / d
            try:
                U, s, Vt = np.linalg.svd(Ad, full_matrices=False)
            except np.linalg.LinAlgError:
                logger.debug("    SVD did not converge for m=%d; solving each lam by QR", m)
                svd["Ad"] = Ad
            else:
                keep = s > np.finfo(float).eps * max(A.shape) * s[0]
                svd.update(s=s[keep], Ut_b=U[:, keep].T @ b, V=Vt[keep].T)
        if "Ad" in svd:
            y = _ridge_qr(svd["Ad"], b, lam)
        else:
            s = svd["s"]
            y = svd["V"] @ (s / (s**2 + lam**2) * svd["Ut_b"])
        return _finish_solve(y / d, val, spec, m)

    return solve


def _candidate_for_m(
    spec: dict,
    m: int,
    lambdas: Sequence[float],
    target: float,
    metric: str,
    order_fit: int,
    fit_oversample: int,
    val_n: int,
) -> dict:
    logger.debug("candidate_for_m: m=%d  target=%.3e  metric=%s", m, target, metric)
    data = _make_fit_data(spec, m, fit_oversample, val_n)
    linear_solve = _make_linear_solver(data, spec, m, order_fit)
    cache: dict[float, dict] = {}

    def solve(lam: float) -> dict:
        lam = float(lam)
        if lam not in cache:
            cache[lam] = linear_solve(lam)
        return cache[lam]

    tol = _tolerance(target)
    s0 = solve(0.0)
    e0 = s0[metric]
    logger.debug("  unregularised error: %.3e  (target=%.3e)", e0, target)

    if not _meets_target(s0, metric, target):
        logger.debug("  m=%d infeasible even at lam=0, skipping lambda search", m)
        return {"m": m, "lambda": 0.0, "solve": s0}
    if math.isclose(e0, target, rel_tol=1e-11, abs_tol=tol):
        logger.debug("  m=%d converged at lam=0", m)
        return {"m": m, "lambda": 0.0, "solve": s0}

    grid = np.unique(np.asarray(lambdas, dtype=float))  # sorted; validated by the caller

    low_lam, low_solve, low_error = 0.0, s0, e0
    high_lam = None
    logger.debug("  scanning lambda grid (%d values)...", len(grid))
    for lam in grid:
        s = solve(float(lam))
        e = s[metric]
        logger.debug("  grid lam=%.3e -> error=%.3e", lam, e)
        if _meets_target(s, metric, target):
            low_lam, low_solve, low_error = float(lam), s, e
        elif float(lam) > low_lam:
            high_lam = float(lam)
            break

    if high_lam is None:
        high = max(float(grid[-1]), 1.0)
        logger.debug("  high_lam not found in grid, expanding (up to 40x)...")
        for _ in range(40):
            high *= 10.0
            s = solve(high)
            e = s[metric]
            logger.debug("  expand lam=%.3e -> error=%.3e", high, e)
            if _meets_target(s, metric, target):
                low_lam, low_solve, low_error = high, s, e
            else:
                high_lam = high
                break

    if high_lam is None:
        logger.debug("  m=%d: no high bracket found, returning best so far (lam=%.3e)", m, low_lam)
        return {"m": m, "lambda": low_lam, "solve": low_solve}
    if math.isclose(low_error, target, rel_tol=1e-11, abs_tol=tol):
        logger.debug("  m=%d: low_error already close enough, skipping bisection", m)
        return {"m": m, "lambda": low_lam, "solve": low_solve}

    logger.debug("  bisecting lam in [%.3e, %.3e]...", low_lam, high_lam)

    # Bisect in log(lam), or in lam itself when the lower end is lam = 0.
    to_u, from_u = (math.log, math.exp) if low_lam > 0.0 else ((lambda v: v), (lambda u: u))
    try:
        left, right = to_u(low_lam), to_u(high_lam)
        for i in range(30):
            mid = 0.5 * (left + right)
            s = solve(from_u(mid))
            logger.debug("  bisect[%d] lam=%.3e  error=%.3e", i, from_u(mid), s[metric])
            if _meets_target(s, metric, target):
                left = mid
            else:
                right = mid
        lam_star = from_u(left)
        s_star = solve(lam_star)
    except (np.linalg.LinAlgError, FloatingPointError, OverflowError) as exc:
        logger.warning("m=%d: lambda bisection failed (%s); using lam=%.3e", m, exc, low_lam)
        return {"m": m, "lambda": low_lam, "solve": low_solve}

    feasible = _meets_target(s_star, metric, target)
    logger.debug("  m=%d bisection done: lam_star=%.3e  feasible=%s", m, lam_star, feasible)
    if not feasible:
        # low_lam met the target on the scan, so fall back to it.
        return {"m": m, "lambda": low_lam, "solve": low_solve}
    return {"m": m, "lambda": lam_star, "solve": s_star}
