"""The Sobolev-regularised least-squares method: the fit, and the search for lambda at each order."""

from __future__ import annotations

import math
from collections.abc import Callable

import numpy as np

from felcu._coefficients import _add_baseline, _basis_matrices, _basis_size, _spec_target
from felcu._fitted import _meets_target, _part_rules, _summarise, _tolerance, _validation_data
from felcu._linalg import _gauss_interval, _lstsq, _penalty, _ridge_qr
from felcu._logging import logger


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


def _candidate_for_m(spec: dict, m: int, target: float, metric: str, val_n: int) -> dict:
    """The regularised fit of order *m* with the largest lambda that meets *target*, if any does."""
    logger.debug("candidate_for_m: m=%d  target=%.3e  metric=%s", m, target, metric)
    data = _make_fit_data(spec, m, spec["fit_oversample"], val_n)
    linear_solve = _make_linear_solver(data, spec, m, spec["order_fit"])
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

    grid = np.unique(np.asarray(spec["lambdas"], dtype=float))  # sorted; validated by the caller

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
