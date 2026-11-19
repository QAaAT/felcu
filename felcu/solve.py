"""Public solver interface for felcu."""

from __future__ import annotations

import warnings
from collections.abc import Sequence
from dataclasses import replace

import numpy as np

from felcu._coefficients import _value
from felcu._fitted import _meets_target
from felcu._logging import logger
from felcu._methods import METHODS, OPTION_NAMES, _method_options
from felcu.spectral import _make_spec, _spectral_data
from felcu.types import ConvergenceWarning, FourierExtensionResult
from felcu.utils import (
    canonical_error_metric,
    canonical_method_name,
    canonical_problem_name,
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
    return FourierExtensionResult(
        problem=spec["problem"],
        method=spec["method"],
        basis_used=spec["basis"],
        m=int(cand["m"]),
        eta=float(spec["eta"]),
        tau=float(spec["tau"]),
        mu=float(spec["mu"]),
        alpha=float(s["alpha"]),
        target_error=float(target),
        error_metric=metric,
        achieved_error=float(s[metric]),
        achieved_l2_error=float(s["l2"]),
        achieved_linf_error=float(s["linf"]),
        converged=_meets_target(s, metric, target),
        a0=a0,
        ak=ak.copy(),
        bk=bk.copy(),
        complex_coefficients=to_complex_coefficients(a0, ak, bk),
        k=np.arange(1, ak.size + 1, dtype=int),
        x_max=float(spec["x_max"]),
        x_gap=spec["x_gap"],
        sigma_min=float(spectral["sigma_min"]),
        sigma_max=float(spectral["sigma_max"]),
        gap=spectral["gap"],
        spectral_kind_used=spectral["kind"],
        matrix_shape=spectral["matrix_shape"],
        matrix_is_hermitian=spectral["matrix_is_hermitian"],
        method_details=METHODS[spec["method"]].result_fields(spec) | cand.get("fields", {}),
    )


def _constant_result(spec: dict, target: float, metric: str, value: float, m: int) -> FourierExtensionResult:
    c = {
        "m": m,
        "fields": dict(METHODS[spec["method"]].constant_fields),
        "solve": {
            "a0": value,
            "ak": np.zeros(m),
            "bk": np.zeros(m),
            "l2": 0.0,
            "linf": 0.0,
            "alpha": abs(value),
        },
    }
    out = _make_result(spec, target, metric, c)
    return replace(out, all_results=(out,))


def _exact_constant_if_any(
    spec: dict,
    target: float,
    metric: str,
    m: int,
) -> FourierExtensionResult | None:
    """The exact constant series when the eigenvalue interval is a single point.

    A point eigenvalue interval means H = lambda I.  Equal singular values do
    not make A a multiple of the identity (A may be any scaled unitary), so
    singular-value fits are fitted as usual.  The series has order *m*, with
    every coefficient but the constant zero.
    """
    s = spec["spectral"]
    if s["kind"] != "eigenvalues" or s["sigma_min"] != s["sigma_max"]:
        return None
    return _constant_result(spec, target, metric, _value(spec["problem"], s["sigma_min"]), m)


def _orders(values, name: str) -> np.ndarray:
    """*values* as a 1-D array of non-negative integers, rejecting fractions rather than truncating them."""
    arr = np.asarray(values)
    valid = (
        arr.ndim == 1
        and arr.size > 0
        and arr.dtype.kind in "iuf"
        and bool(np.all(np.isfinite(arr) & (arr >= 0) & (arr == np.round(arr))))
    )
    if not valid:
        raise ValueError(f"{name} must be a non-negative integer or a non-empty sequence of them.")
    return arr.astype(int)


def _m_grid(m: int | None, m_values: Sequence[int] | int | None) -> np.ndarray:
    """Ascending candidate orders: *m* alone if given, else *m_values*, else the default grid."""
    if m is not None:
        if np.ndim(m) != 0:
            raise ValueError("m must be a non-negative integer.")
        return _orders([m], "m")
    if m_values is None:
        return default_m_grid()
    return np.unique(_orders(np.atleast_1d(m_values), "m_values"))


def _select(candidates: list[dict], metric: str, target: float, selection: str) -> tuple[dict, bool]:
    """Pick the candidate to return, and whether it meets *target*.

    Among feasible candidates, ``'smallest_m'`` prefers the lowest order and
    ``'smallest_alpha'`` the lowest subnormalisation alpha (ties broken by the
    other, then by error).  With none feasible, the lowest error wins.
    """
    feasible = [c for c in candidates if _meets_target(c["solve"], metric, target)]
    logger.info("%d/%d candidates are feasible", len(feasible), len(candidates))
    if not feasible:
        return min(candidates, key=lambda c: (c["solve"][metric], c["m"], c["solve"]["alpha"])), False
    if selection == "smallest_m":
        return min(feasible, key=lambda c: (c["m"], c["solve"]["alpha"], c["solve"][metric])), True
    return min(feasible, key=lambda c: (c["solve"]["alpha"], c["m"], c["solve"][metric])), True


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def solve_fe_coefficients(
    problem: str,
    *,
    method: str = "sobolev",
    matrix: np.ndarray | Sequence[Sequence[complex]] | None = None,
    sigma_min: float | None = None,
    sigma_max: float | None = None,
    gap: tuple[float, float] | None = None,
    spectral_kind: str = "auto",
    eta: float | None = None,
    target_error: float = 1e-12,
    error_metric: str = "linf",
    m: int | None = None,
    m_values: Sequence[int] | int | None = None,
    lambdas: Sequence[float] | None = None,
    sobolev_w: float | None = None,
    sobolev_r: float | None = None,
    order_fit: int | None = None,
    fit_oversample: int | None = None,
    val_n: int = 4001,
    selection: str = "smallest_alpha",
    hermitian_tol: float = 1e-12,
) -> FourierExtensionResult:
    """Compute Fourier-extension LCU block-encoding coefficients.

    The fit is centred on the spectrum: with ``mu`` and ``delta`` the
    centre and half-width of the fitted set, ``tau = pi / (eta delta)`` and
    ``f(H) = g(tau (H - mu I))`` (``result.mu``, ``result.tau``).  Singular
    values are fitted through the Hermitian dilation of the matrix, whose
    spectrum is symmetric, so ``mu = 0`` and the series is odd.

    Parameters
    ----------
    problem:
        One of ``'identity'`` (or ``'action'``), ``'inversion'``, ``'exponentiation'``, ``'square_root'``, or
        ``'logarithm'`` (several aliases accepted; see
        :func:`~felcu.utils.canonical_problem_name`).  Square root and
        logarithm need a positive-definite eigenvalue interval.
    method:
        Coefficient-selection strategy (see
        :func:`~felcu.canonical_method_name` for aliases):

        - ``'sobolev'`` (default): Sobolev-regularised least squares.
          Near-exponential convergence down to *target_error*, with
          subnormalisation ``O(||f(H)||)``.
        - ``'reflected'``: the Fourier series of g reflected about
          ``x = +-pi/2``.  Optimal subnormalisation
          ``alpha_m = ||f(H)|| - epsilon_m`` for convex endpoint-maximising f
          (identity, exponentiation, definite inversion) at the default
          ``eta = 2``, but only ``O(1/m)`` convergence, so the default
          *target_error* is out of reach.
        - ``'arcsine_taylor'``: the Taylor series of ``g(arcsin z)`` with
          ``z = sin x``.  Exponential convergence with factor
          ``sin(pi/eta)`` (requires ``eta > 2``) and subnormalisation
          bounded independently of the error.

        The explicit methods do not support the two-interval fit of
        inversion on an indefinite interval or on singular values, and for
        definite inversion, square root and logarithm need
        ``eta < 2 |mu| / delta``.
        They have no regularisation, so the Sobolev-only options
        (*lambdas*, *sobolev_w*, *sobolev_r*, *order_fit*,
        *fit_oversample*) must be left unset.
    matrix:
        Square matrix whose spectral information drives the fitting.
        Mutually exclusive with *sigma_min* / *sigma_max* / *gap*; one of
        the two is required.
    sigma_min, sigma_max:
        Spectral interval: eigenvalue or singular-value bounds, per
        *spectral_kind*.
    gap:
        For inversion on an indefinite eigenvalue interval
        (``sigma_min < 0 < sigma_max``), the eigenvalues ``(a, b)`` closest
        to 0 on either side: the spectrum lies in
        ``[sigma_min, a] U [b, sigma_max]``, and the series is fitted on both
        parts, leaving the gap around the pole free.  Required in that case
        and not allowed otherwise; with *matrix*, it is read from the
        eigenvalues.
    spectral_kind:
        Which spectral data the bounds (or *matrix*) describe.
        ``'eigenvalues'`` needs a Hermitian matrix.  ``'singular_values'``
        acts on a general matrix through its Hermitian dilation, with an odd
        series, so it is available for identity and inversion only.
        ``'auto'`` (default) uses singular values for identity and inversion
        of a non-Hermitian matrix, and eigenvalues otherwise.
    eta:
        Fourier-extension parameter (must exceed 1).  Computed automatically
        when omitted; see :func:`~felcu.default_eta`.
    target_error:
        Desired approximation error.
    error_metric:
        ``'l2'`` or ``'linf'``.
    m:
        Fixed Fourier truncation order; overrides *m_values*.
    m_values:
        Candidate truncation orders to search over.
    lambdas:
        Grid of regularisation strengths ``lambda = sqrt(2 pi gamma)``, where
        *gamma* is the weight of the H_r^w penalty in the paper's
        regularised problem.  The largest lambda meeting *target_error* is
        used.
    sobolev_w:
        Order *w* of the H_r^w regulariser, which weights the k-th Fourier
        mode by ``P_w(r|k|) = sum_{j=0}^{w} (r|k|)^(2j)``, i.e. penalises
        every derivative up to order *w*; for fractional *w*,
        ``P_w(z) = (1 - z^(2(w+1))) / (1 - z^2)``.  Must exceed ``0.5``;
        the paper's bounds cover ``0.5 < w <= 1``.  Defaults to ``1``.
    sobolev_r:
        Scale *r* of the H_r^w regulariser, with ``0 < r <= 1``.  Chosen by
        :func:`~felcu.default_sobolev_r` when omitted.
    order_fit:
        Number of derivative orders included in the fit system: ``1``
        (default) fits in the H^1 norm, ``0`` in plain L^2.
    fit_oversample:
        Oversampling factor for the Gauss quadrature grid.  Defaults to
        ``8``.
    val_n:
        Number of validation points.
    selection:
        ``'smallest_alpha'`` (default) picks the feasible order with the
        smallest subnormalisation alpha, which requires solving every order in
        the sweep.  ``'smallest_m'`` picks the shortest feasible series and
        stops the sweep there, so ``result.all_results`` only holds the
        orders up to it.
    hermitian_tol:
        Relative tolerance for the Hermitian test, which compares each entry
        of *matrix* with its mirror.

    Returns
    -------
    FourierExtensionResult
        The selected result; ``result.all_results`` contains every candidate.
    """
    arguments = locals()  # the parameters, before any other local is made
    problem = canonical_problem_name(problem)
    method = canonical_method_name(method)
    metric = canonical_error_metric(error_metric)
    options = _method_options(method, {name: arguments[name] for name in OPTION_NAMES})
    if not target_error >= 0:  # also catches NaN
        raise ValueError("target_error must be non-negative.")
    if val_n < 3:
        raise ValueError("val_n must be at least 3.")
    if selection not in {"smallest_m", "smallest_alpha"}:
        raise ValueError("selection must be 'smallest_m' or 'smallest_alpha'.")

    spectral = _spectral_data(problem, matrix, sigma_min, sigma_max, gap, spectral_kind, hermitian_tol)
    spec = _make_spec(problem, spectral, eta, method)

    spec = spec | METHODS[method].prepare(spec, options)

    logger.info(
        "solve_fe_coefficients: problem=%s  method=%s  kind=%s  metric=%s  target=%.3e  basis=%s  x_gap=%s  "
        "eta=%.4f  tau=%s  mu=%.4g",
        problem,
        method,
        spectral["kind"],
        metric,
        target_error,
        spec["basis"],
        spec["x_gap"],
        spec["eta"],
        f"{spec['tau']:.4f}" if np.isfinite(spec["tau"]) else "inf",
        spec["mu"],
    )

    target = float(target_error)
    m_grid = _m_grid(m, m_values)

    # Exact at any order, so the smallest one asked for, or 0 if none was.
    exact_m = int(m_grid[0]) if m is not None or m_values is not None else 0
    exact = _exact_constant_if_any(spec, target, metric, exact_m)
    if exact is not None:
        logger.info("exact constant solution found, returning immediately")
        return exact

    logger.info("sweeping %d m values: %s ... %s", len(m_grid), m_grid[0], m_grid[-1])

    candidates = []
    for i, mi in enumerate(m_grid):
        logger.info("[%d/%d] m=%d ...", i + 1, len(m_grid), mi)
        c = METHODS[method].candidate(spec, int(mi), target, metric, val_n)
        feasible = _meets_target(c["solve"], metric, target)
        logger.info(
            "[%d/%d] m=%d  done: error=%.3e  feasible=%s  alpha=%.3e  %s",
            i + 1,
            len(m_grid),
            mi,
            c["solve"][metric],
            feasible,
            c["solve"]["alpha"],
            "  ".join(f"{name}={value:.3e}" for name, value in c.get("fields", {}).items()),
        )
        candidates.append(c)
        if selection == "smallest_m" and feasible:
            # m_grid is ascending, so no later m can be smaller.
            logger.info("smallest feasible m found, stopping sweep")
            break

    best, converged = _select(candidates, metric, target, selection)
    logger.info("selected: m=%d  error=%.3e  converged=%s", best["m"], best["solve"][metric], converged)
    if not converged:
        warnings.warn(
            f"No candidate met target_error={target:.3g} in the {metric} metric; returning m={best['m']} "
            f"with error {best['solve'][metric]:.3g}.  Try larger m_values or a looser target_error.",
            ConvergenceWarning,
            stacklevel=2,
        )

    # m_grid has no repeats, so each order has one result.
    results = {c["m"]: _make_result(spec, target, metric, c) for c in candidates}
    all_results = tuple(results[mi] for mi in sorted(results))
    return replace(results[best["m"]], all_results=all_results)


def compute_fe_coefficients(*args, **kwargs) -> tuple[float, np.ndarray, np.ndarray]:
    """Convenience wrapper; returns ``(a0, ak, bk)`` directly.

    Takes the same arguments as :func:`solve_fe_coefficients` and discards
    everything but the coefficients.  Use :func:`solve_fe_coefficients`
    when you also need ``tau``, the achieved error, or whether the fit
    converged.

    Returns
    -------
    a0 : float
        Constant term.
    ak, bk : numpy.ndarray
        Sine and cosine coefficients (copies), each of length ``m``.
    """
    result = solve_fe_coefficients(*args, **kwargs)
    return result.a0, result.ak.copy(), result.bk.copy()
