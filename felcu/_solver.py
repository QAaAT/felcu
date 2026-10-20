"""Core Fourier-extension fitting and candidate-selection logic for felcu."""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence

import numpy as np

from felcu._coefficients import (
    _add_baseline,
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
    _validation_errors,
)
from felcu._logging import logger


def _tolerance(target: float) -> float:
    """Slack allowed when comparing an achieved error with *target*."""
    return max(1e-15, abs(target) * 1e-12)


def _meets_target(solve: dict, metric: str, target: float) -> bool:
    """Whether a solve's error in *metric* ('l2' or 'linf') meets *target*."""
    return solve[metric] <= target + _tolerance(target)


def _make_fit_data(spec: dict, m: int, fit_oversample: int, val_n: int) -> dict:
    basis = spec["basis"]
    ncols = _basis_size(basis, m)

    if spec["domain"] == "single":
        nq = max(fit_oversample * max(ncols, 1), 48)
        logger.debug("  fit_data: domain=single  basis=%s  m=%d  ncols=%d  nq=%d  val_n=%d", basis, m, ncols, nq, val_n)
        x, weights = _gauss_interval(-spec["x_max"], spec["x_max"], nq)
        xv = np.linspace(-spec["x_max"], spec["x_max"], max(val_n, 201))
        val = [(xv, _basis_matrix(xv, m, basis), _spec_target(xv, spec))]
    else:
        xmin = spec["x_min"]
        xmax = spec["x_max"]
        nq = max((fit_oversample * max(ncols, 1) + 1) // 2, 32)
        logger.debug("  fit_data: domain=double  basis=%s  m=%d  ncols=%d  nq=%d  val_n=%d", basis, m, ncols, nq, val_n)
        xl, wl = _gauss_interval(-xmax, -xmin, nq)
        xr, wr = _gauss_interval(xmin, xmax, nq)
        x = np.concatenate([xl, xr])
        weights = np.concatenate([wl, wr])
        nv = max(val_n // 2, 201)
        xl_v = np.linspace(-xmax, -xmin, nv)
        xr_v = np.linspace(xmin, xmax, nv)
        val = [
            (xl_v, _basis_matrix(xl_v, m, basis), _spec_target(xl_v, spec)),
            (xr_v, _basis_matrix(xr_v, m, basis), _spec_target(xr_v, spec)),
        ]

    return {"x": x, "sw": np.sqrt(weights), "val": val}


def _fit_system(data: dict, spec: dict, m: int, order_fit: int) -> tuple[np.ndarray, np.ndarray]:
    basis = spec["basis"]
    x = data["x"]
    sw = data["sw"]
    A_blocks = []
    b_blocks = []
    for d in range(order_fit + 1):
        Ad = _basis_matrix(x, m, basis, d)
        # The constant baseline is fitted exactly, so only the remainder is fitted here.
        bd = _spec_target(x, spec, d) - (spec["baseline"] if d == 0 else 0.0)
        A_blocks.append(Ad * sw[:, None])
        b_blocks.append(bd * sw)
    return np.vstack(A_blocks), np.concatenate(b_blocks)


def _finish_solve(residual_coeffs: np.ndarray, data: dict, spec: dict, m: int) -> dict:
    basis = spec["basis"]
    coeffs = _add_baseline(residual_coeffs, basis, spec["baseline"])
    a0, ak, bk = _unpack_coefficients(coeffs, basis, m)
    l2, linf = _validation_errors(coeffs, data["val"])
    alpha = _alpha(a0, ak, bk)
    logger.debug("    solve done: l2=%.3e  linf=%.3e  alpha=%.3e", l2, linf, alpha)
    return {"a0": a0, "ak": ak, "bk": bk, "l2": l2, "linf": linf, "alpha": alpha}


def _make_linear_solver(
    data: dict,
    spec: dict,
    m: int,
    order_fit: int,
    sobolev_r: float,
    sobolev_w: float,
) -> Callable[[float], dict]:
    """Return ``solve(lam)`` for one m, building the fit system only once.

    The penalty is ``lam * D`` with ``D`` diagonal and invertible, so
    substituting ``y = D c`` turns ``min ||A c - b||^2 + lam^2 ||D c||^2``
    into ridge regression on ``A D^{-1}``.  One SVD of that matrix then
    solves every ``lam > 0`` cheaply.  ``lam = 0`` keeps the plain
    least-squares solve on ``A``.
    """
    basis = spec["basis"]
    ncols = _basis_size(basis, m)
    if ncols == 0:
        return lambda lam: _finish_solve(np.zeros(0), data, spec, m)

    A, b = _fit_system(data, spec, m, order_fit)
    logger.debug("    fit system shape: %s", A.shape)
    d = _penalty(m, basis, 1.0, sobolev_r, sobolev_w)
    svd: dict[str, np.ndarray] = {}

    def solve(lam: float) -> dict:
        logger.debug("    solve: m=%d  lam=%.3e", m, lam)
        if lam == 0.0:
            return _finish_solve(_lstsq(A, b), data, spec, m)
        if not svd:
            # Factorise lazily: an m that is infeasible at lam = 0 never needs it.
            U, s, Vt = np.linalg.svd(A / d, full_matrices=False)
            keep = s > np.finfo(float).eps * max(A.shape) * s[0]
            svd.update(s=s[keep], Ut_b=U[:, keep].T @ b, V=Vt[keep].T)
        s = svd["s"]
        y = svd["V"] @ (s / (s**2 + lam**2) * svd["Ut_b"])
        return _finish_solve(y / d, data, spec, m)

    return solve


def _candidate_for_m(
    spec: dict,
    m: int,
    lambdas: Sequence[float],
    target: float,
    metric: str,
    order_fit: int,
    sobolev_r: float,
    sobolev_w: float,
    fit_oversample: int,
    val_n: int,
) -> dict:
    logger.debug("candidate_for_m: m=%d  target=%.3e  metric=%s", m, target, metric)
    data = _make_fit_data(spec, m, fit_oversample, val_n)
    linear_solve = _make_linear_solver(data, spec, m, order_fit, sobolev_r, sobolev_w)
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

    if e0 > target + tol:
        logger.debug("  m=%d infeasible even at lam=0, skipping lambda search", m)
        return {"m": m, "lambda": 0.0, "solve": s0, "feasible": False}
    if math.isclose(e0, target, rel_tol=1e-11, abs_tol=tol):
        logger.debug("  m=%d converged at lam=0", m)
        return {"m": m, "lambda": 0.0, "solve": s0, "feasible": True}

    grid = np.sort(np.unique(np.asarray(lambdas, dtype=float)))
    grid = grid[grid >= 0]
    if grid.size == 0:
        raise ValueError("lambdas must contain at least one positive value.")

    low_lam, low_solve, low_error = 0.0, s0, e0
    high_lam = None
    logger.debug("  scanning lambda grid (%d values)...", len(grid))
    for lam in grid:
        s = solve(float(lam))
        e = s[metric]
        logger.debug("  grid lam=%.3e -> error=%.3e", lam, e)
        if e <= target + tol:
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
            if e <= target + tol:
                low_lam, low_solve, low_error = high, s, e
            else:
                high_lam = high
                break

    if high_lam is None:
        logger.debug("  m=%d: no high bracket found, returning best so far (lam=%.3e)", m, low_lam)
        return {"m": m, "lambda": low_lam, "solve": low_solve, "feasible": True}
    if math.isclose(low_error, target, rel_tol=1e-11, abs_tol=tol):
        logger.debug("  m=%d: low_error already close enough, skipping bisection", m)
        return {"m": m, "lambda": low_lam, "solve": low_solve, "feasible": True}

    logger.debug("  bisecting lam in [%.3e, %.3e]...", low_lam, high_lam)

    # Bisect in log(lam), or in lam itself when the lower end is lam = 0.
    to_u, from_u = (math.log, math.exp) if low_lam > 0.0 else ((lambda v: v), (lambda u: u))
    try:
        left, right = to_u(low_lam), to_u(high_lam)
        for i in range(30):
            mid = 0.5 * (left + right)
            s = solve(from_u(mid))
            logger.debug("  bisect[%d] lam=%.3e  error=%.3e", i, from_u(mid), s[metric])
            if s[metric] <= target:
                left = mid
            else:
                right = mid
        lam_star = from_u(left)
        s_star = solve(lam_star)
    except (np.linalg.LinAlgError, FloatingPointError, OverflowError) as exc:
        logger.warning("m=%d: lambda bisection failed (%s); using lam=%.3e", m, exc, low_lam)
        return {"m": m, "lambda": low_lam, "solve": low_solve, "feasible": True}

    feasible = _meets_target(s_star, metric, target)
    logger.debug("  m=%d bisection done: lam_star=%.3e  feasible=%s", m, lam_star, feasible)
    if not feasible:
        # low_lam met the target on the scan, so fall back to it.
        return {"m": m, "lambda": low_lam, "solve": low_solve, "feasible": True}
    return {"m": m, "lambda": lam_star, "solve": s_star, "feasible": True}
