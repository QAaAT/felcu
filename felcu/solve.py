"""Public solver interface for felcu."""

from __future__ import annotations

from dataclasses import replace
from typing import Optional, Sequence

import numpy as np

from felcu._logging import logger
from felcu._solver import _candidate_for_m, _metric
from felcu.spectral import _make_spec, _spectral_data
from felcu.types import FourierExtensionResult
from felcu.utils import (
    canonical_error_metric,
    canonical_problem_name,
    default_lambda_grid,
    default_m_grid,
    to_complex_coefficients,
)


# ---------------------------------------------------------------------------
# Internal result construction
# ---------------------------------------------------------------------------


def _make_result(spec: dict, target: float, metric: str, cand: dict) -> FourierExtensionResult:
    s = cand["solve"]
    spectral = spec["spectral"]
    a0 = float(s["a0"])
    ak = np.asarray(s["ak"], dtype=float)
    bk = np.asarray(s["bk"], dtype=float)
    chosen_error = _metric(s, metric)
    tol = max(1e-15, abs(target) * 1e-12)

    return FourierExtensionResult(
        problem=spec["problem"],
        shifted=spec["shifted"],
        basis_used=spec["basis"],
        m=int(cand["m"]),
        eta=float(spec["eta"]),
        tau=float(spec["tau"]),
        alpha=float(s["alpha"]),
        target_error=float(target),
        error_metric=metric,
        achieved_error=float(chosen_error),
        achieved_l2_error=float(s["l2"]),
        achieved_linf_error=float(s["linf"]),
        converged=bool(chosen_error <= target + tol),
        lambda_used=float(cand["lambda"]),
        a0=a0,
        ak=ak.copy(),
        bk=bk.copy(),
        complex_coefficients=to_complex_coefficients(a0, ak, bk),
        k=np.arange(1, ak.size + 1, dtype=int),
        domain=spec["domain"],
        x_max=float(spec["x_max"]),
        x_min=None if spec["x_min"] is None else float(spec["x_min"]),
        sigma_min=None if spectral is None else float(spectral["sigma_min"]),
        sigma_max=None if spectral is None else float(spectral["sigma_max"]),
        spectral_kind_used=None if spectral is None else spectral["kind"],
        matrix_shape=None if spectral is None else spectral["matrix_shape"],
        matrix_is_hermitian=None if spectral is None else spectral["matrix_is_hermitian"],
        matrix_scale_provided=bool(spec["matrix_scale_provided"]),
    )


def _constant_result(spec: dict, target: float, metric: str, value: float) -> FourierExtensionResult:
    c = {
        "m": 0,
        "lambda": 0.0,
        "solve": {
            "a0": value,
            "ak": np.zeros(0),
            "bk": np.zeros(0),
            "l2": 0.0,
            "linf": 0.0,
            "alpha": abs(value),
        },
    }
    spec = spec | {"basis": "full" if spec["basis"] == "odd_sine" else spec["basis"]}
    out = _make_result(spec, target, metric, c)
    return replace(out, all_results=(out,))


def _exact_constant_if_any(
    spec: dict,
    target: float,
    metric: str,
) -> Optional[FourierExtensionResult]:
    import math as _math

    s = spec["spectral"]
    if s is None:
        return None
    lo, hi = s["sigma_min"], s["sigma_max"]

    if lo != hi:
        if spec["problem"] == "action" and not spec["shifted"] and max(abs(lo), abs(hi)) == 0:
            return _constant_result(spec, target, metric, 0.0)
        if spec["problem"] == "exponentiation" and not spec["shifted"] and max(abs(lo), abs(hi)) == 0:
            return _constant_result(spec, target, metric, 1.0)
        return None

    if spec["problem"] == "action":
        return _constant_result(spec, target, metric, lo)
    if spec["problem"] == "inversion" and spec["shifted"]:
        return _constant_result(spec, target, metric, 1.0 / lo)
    if spec["problem"] == "exponentiation":
        return _constant_result(spec, target, metric, _math.exp(lo))
    return None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def solve_fe_coefficients(
    problem: str,
    *,
    shifted: bool = False,
    matrix: Optional[np.ndarray | Sequence[Sequence[complex]]] = None,
    sigma_min: Optional[float] = None,
    sigma_max: Optional[float] = None,
    spectral_kind: str = "auto",
    eta: Optional[float] = None,
    target_error: float = 1e-12,
    error_metric: str = "linf",
    m: Optional[int] = None,
    m_values: Optional[Sequence[int] | int] = None,
    lambdas: Optional[Sequence[float]] = None,
    k_order: int = 1,
    order_fit: int = 1,
    fit_oversample: int = 8,
    val_n: int = 4001,
    selection: str = "smallest_m",
    hermitian_tol: float = 1e-12,
) -> FourierExtensionResult:
    """Compute Fourier-extension LCU block-encoding coefficients.

    Parameters
    ----------
    problem:
        One of ``'action'``, ``'inversion'``, or ``'exponentiation'``
        (several aliases accepted; see :func:`~felcu.utils.canonical_problem_name`).
    shifted:
        Use the shifted (centred) construction.
    matrix:
        Square matrix whose spectral information drives the fitting.
        Mutually exclusive with *sigma_min* / *sigma_max*.
    sigma_min, sigma_max:
        Spectral interval (eigenvalue or singular-value bounds).
    spectral_kind:
        ``'auto'``, ``'eigenvalues'``, or ``'singular_values'``.
    eta:
        Fourier-extension parameter (must exceed 1).  Computed automatically
        when omitted.
    target_error:
        Desired approximation error.
    error_metric:
        ``'l2'`` or ``'linf'``.
    m:
        Fixed Fourier truncation order; overrides *m_values*.
    m_values:
        Candidate truncation orders to search over.
    lambdas:
        Regularisation parameter grid.
    k_order:
        Derivative order used in the regulariser.
    order_fit:
        Number of derivative orders included in the fit system.
    fit_oversample:
        Oversampling factor for the Gauss quadrature grid.
    val_n:
        Number of validation points.
    selection:
        ``'smallest_m'`` (default) or ``'smallest_alpha'``.
    hermitian_tol:
        Tolerance for the Hermitian test.

    Returns
    -------
    FourierExtensionResult
        The selected result; ``result.all_results`` contains every candidate.
    """
    problem = canonical_problem_name(problem)
    metric = canonical_error_metric(error_metric)
    if target_error < 0:
        raise ValueError("target_error must be positive.")
    if k_order < 0 or order_fit < 0 or fit_oversample <= 0 or val_n < 3:
        raise ValueError("Require k_order >= 0, order_fit >= 0, fit_oversample > 0, and val_n >= 3.")
    if selection not in {"smallest_m", "smallest_alpha"}:
        raise ValueError("selection must be 'smallest_m' or 'smallest_alpha'.")

    spectral = _spectral_data(problem, shifted, matrix, sigma_min, sigma_max, spectral_kind, hermitian_tol, True)
    spec = _make_spec(problem, shifted, spectral, eta, hermitian_tol)

    logger.info(
        "solve_fe_coefficients: problem=%s  shifted=%s  metric=%s  target=%.3e  "
        "basis=%s  domain=%s  eta=%.4f  tau=%s",
        problem, shifted, metric, target_error,
        spec["basis"], spec["domain"], spec["eta"],
        f"{spec['tau']:.4f}" if np.isfinite(spec["tau"]) else "inf",
    )

    exact = _exact_constant_if_any(spec, float(target_error), metric)
    if exact is not None:
        logger.info("exact constant solution found, returning immediately")
        return exact

    if lambdas is None:
        lambdas = default_lambda_grid()
    else:
        lambdas = np.asarray(lambdas, dtype=float)
        if lambdas.ndim != 1 or lambdas.size == 0 or np.any(lambdas < 0):
            raise ValueError("lambdas must be a non-empty sequence of positive values.")

    if m is not None:
        if m < 0:
            raise ValueError("m must be non-negative.")
        m_grid = np.array([int(m)], dtype=int)
    elif m_values is None:
        m_grid = default_m_grid()
    elif np.isscalar(m_values):
        if int(m_values) < 0:
            raise ValueError("m_values must be non-negative.")
        m_grid = np.array([int(m_values)], dtype=int)
    else:
        m_grid = np.unique(np.asarray(m_values, dtype=int))
        if m_grid.ndim != 1 or m_grid.size == 0 or np.any(m_grid < 0):
            raise ValueError("m_values must be a non-empty sequence of non-negative integers.")

    logger.info("sweeping %d m values: %s ... %s", len(m_grid), m_grid[0], m_grid[-1])

    candidates = []
    for i, mi in enumerate(m_grid):
        logger.info("[%d/%d] m=%d ...", i + 1, len(m_grid), mi)
        c = _candidate_for_m(
            spec, int(mi), lambdas, float(target_error), metric,
            order_fit, k_order, fit_oversample, val_n,
        )
        err = _metric(c["solve"], metric)
        logger.info(
            "[%d/%d] m=%d  done: error=%.3e  feasible=%s  lam=%.3e  alpha=%.3e",
            i + 1, len(m_grid), mi, err, c["feasible"], c["lambda"], c["solve"]["alpha"],
        )
        candidates.append(c)

    tol = max(1e-15, abs(float(target_error)) * 1e-12)
    feasible = [c for c in candidates if _metric(c["solve"], metric) <= float(target_error) + tol]
    logger.info("%d/%d candidates are feasible", len(feasible), len(candidates))

    if feasible and selection == "smallest_m":
        best = min(feasible, key=lambda c: (c["m"], c["solve"]["alpha"], _metric(c["solve"], metric)))
    elif feasible:
        best = min(feasible, key=lambda c: (c["solve"]["alpha"], c["m"], _metric(c["solve"], metric)))
    else:
        best = min(candidates, key=lambda c: (_metric(c["solve"], metric), c["m"], c["solve"]["alpha"]))

    logger.info(
        "selected: m=%d  error=%.3e  converged=%s",
        best["m"], _metric(best["solve"], metric), bool(feasible),
    )

    all_results = tuple(
        _make_result(spec, float(target_error), metric, c)
        for c in sorted(candidates, key=lambda c: c["m"])
    )
    selected = _make_result(spec, float(target_error), metric, best)
    return replace(selected, all_results=all_results)


def compute_fe_coefficients(*args, **kwargs) -> tuple[float, np.ndarray, np.ndarray]:
    """Convenience wrapper; returns ``(a0, ak, bk)`` directly."""
    result = solve_fe_coefficients(*args, **kwargs)
    return result.a0, result.ak.copy(), result.bk.copy()
