"""Core Fourier-extension fitting and candidate-selection logic for felcu."""

from __future__ import annotations

import math
from typing import Callable, Sequence

import numpy as np

from felcu._coefficients import (
    _basis_derivative_matrix,
    _basis_matrix,
    _basis_size,
    _target_derivative,
    _add_baseline,
    _unpack_coefficients,
)
from felcu._linalg import (
    _alpha,
    _gauss_interval,
    _penalty,
    _qr_solve,
    _validation_errors,
)
from felcu._logging import logger


def _metric(solve: dict, name: str) -> float:
    return solve["l2"] if name == "l2" else solve["linf"]


def _make_fit_data(spec: dict, m: int, fit_oversample: int, val_n: int) -> dict:
    basis = spec["basis"]
    ncols = _basis_size(basis, m)
    f: Callable[[np.ndarray], np.ndarray] = spec["func"]

    if spec["domain"] == "single":
        nq = max(fit_oversample * max(ncols, 1), 48)
        logger.debug("  fit_data: domain=single  basis=%s  m=%d  ncols=%d  nq=%d  val_n=%d",
                     basis, m, ncols, nq, val_n)
        x, weights = _gauss_interval(-spec["x_max"], spec["x_max"], nq)
        xv = np.linspace(-spec["x_max"], spec["x_max"], max(val_n, 201))
        val = [(xv, _basis_matrix(xv, m, basis), f(xv))]
    else:
        xmin = spec["x_min"]
        xmax = spec["x_max"]
        nq = max((fit_oversample * max(ncols, 1) + 1) // 2, 32)
        logger.debug("  fit_data: domain=double  basis=%s  m=%d  ncols=%d  nq=%d  val_n=%d",
                     basis, m, ncols, nq, val_n)
        xl, wl = _gauss_interval(-xmax, -xmin, nq)
        xr, wr = _gauss_interval(xmin, xmax, nq)
        x = np.concatenate([xl, xr])
        weights = np.concatenate([wl, wr])
        nv = max(val_n // 2, 201)
        xl_v = np.linspace(-xmax, -xmin, nv)
        xr_v = np.linspace(xmin, xmax, nv)
        val = [
            (xl_v, _basis_matrix(xl_v, m, basis), f(xl_v)),
            (xr_v, _basis_matrix(xr_v, m, basis), f(xr_v)),
        ]

    return {"x": x, "sw": np.sqrt(weights), "val": val}


def _linear_solve(
    data: dict,
    spec: dict,
    m: int,
    lam: float,
    order_fit: int,
    order_reg: int,
) -> dict:
    basis = spec["basis"]
    ncols = _basis_size(basis, m)
    logger.debug("    linear_solve: m=%d  lam=%.3e  ncols=%d  order_fit=%d  order_reg=%d",
                 m, lam, ncols, order_fit, order_reg)

    if ncols == 0:
        residual_coeffs = np.zeros(0)
    else:
        A_blocks = []
        b_blocks = []
        x = data["x"]
        sw = data["sw"]

        for d in range(order_fit + 1):
            Ad = _basis_derivative_matrix(x, m, basis, d)
            bd = _target_derivative(x, spec, d)
            A_blocks.append(Ad * sw[:, None])
            b_blocks.append(bd * sw)

        A_fit = np.vstack(A_blocks)
        b_fit = np.concatenate(b_blocks)
        logger.debug("    lstsq system shape: %s", A_fit.shape)

        if lam == 0.0:
            residual_coeffs = _qr_solve(A_fit, b_fit)
        else:
            p = _penalty(m, basis, lam, order_reg)
            A_aug = np.vstack([A_fit, np.diag(p)])
            b_aug = np.concatenate([b_fit, np.zeros(ncols)])
            logger.debug("    augmented system shape: %s", A_aug.shape)
            residual_coeffs = _qr_solve(A_aug, b_aug)

    coeffs = _add_baseline(residual_coeffs, basis, spec["baseline"])
    a0, ak, bk = _unpack_coefficients(coeffs, basis, m)
    l2, linf = _validation_errors(coeffs, data["val"])
    logger.debug("    linear_solve done: l2=%.3e  linf=%.3e  alpha=%.3e",
                 l2, linf, _alpha(a0, ak, bk))
    return {"a0": a0, "ak": ak, "bk": bk, "l2": l2, "linf": linf, "alpha": _alpha(a0, ak, bk)}


def _candidate_for_m(
    spec: dict,
    m: int,
    lambdas: Sequence[float],
    target: float,
    metric: str,
    order_fit: int,
    order_reg: int,
    fit_oversample: int,
    val_n: int,
) -> dict:
    logger.debug("candidate_for_m: m=%d  target=%.3e  metric=%s", m, target, metric)
    data = _make_fit_data(spec, m, fit_oversample, val_n)
    cache: dict[float, dict] = {}

    def solve(lam: float) -> dict:
        lam = float(lam)
        if lam not in cache:
            cache[lam] = _linear_solve(data, spec, m, lam, order_fit, order_reg)
        return cache[lam]

    tol = max(1e-15, abs(target) * 1e-12)
    s0 = solve(0.0)
    e0 = _metric(s0, metric)
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
        e = _metric(s, metric)
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
            e = _metric(s, metric)
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

    def objective(lam: float) -> float:
        return _metric(solve(lam), metric) - target

    try:
        if low_lam == 0.0:
            left, right = 0.0, high_lam
            for i in range(30):
                mid = 0.5 * (left + right)
                obj = objective(mid)
                logger.debug("  bisect[%d] lam=%.3e  obj=%.3e", i, mid, obj)
                if obj <= 0.0:
                    left = mid
                else:
                    right = mid
            lam_star = left
        else:
            left, right = math.log(low_lam), math.log(high_lam)
            for i in range(30):
                mid = 0.5 * (left + right)
                obj = objective(math.exp(mid))
                logger.debug("  bisect[%d] lam=%.3e  obj=%.3e", i, math.exp(mid), obj)
                if obj <= 0.0:
                    left = mid
                else:
                    right = mid
            lam_star = math.exp(left)

        s_star = solve(lam_star)
        feasible = _metric(s_star, metric) <= target + tol
        logger.debug("  m=%d bisection done: lam_star=%.3e  feasible=%s", m, lam_star, feasible)
        if not feasible and low_error <= target + tol:
            return {"m": m, "lambda": low_lam, "solve": low_solve, "feasible": True}
        return {"m": m, "lambda": lam_star, "solve": s_star, "feasible": feasible}
    except (ValueError, FloatingPointError, OverflowError):
        logger.debug("  m=%d bisection raised, falling back to lam=%.3e", m, low_lam)
        return {"m": m, "lambda": low_lam, "solve": low_solve, "feasible": True}
