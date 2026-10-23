"""Public solver interface for felcu."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace

import numpy as np

from felcu._explicit import _explicit_candidate
from felcu._logging import logger
from felcu._solver import _candidate_for_m, _meets_target
from felcu.spectral import _CENTRE_VALUE, _make_spec, _spectral_data, default_sobolev_r
from felcu.types import FourierExtensionResult
from felcu.utils import (
    canonical_error_metric,
    canonical_method_name,
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

    sobolev = spec["method"] == "sobolev"
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
        lambda_used=float(cand["lambda"]) if sobolev else None,
        sobolev_r=float(spec["sobolev_r"]) if sobolev else None,
        sobolev_w=float(spec["sobolev_w"]) if sobolev else None,
        a0=a0,
        ak=ak.copy(),
        bk=bk.copy(),
        complex_coefficients=to_complex_coefficients(a0, ak, bk),
        k=np.arange(1, ak.size + 1, dtype=int),
        domain=spec["domain"],
        x_max=float(spec["x_max"]),
        x_gap=spec["x_gap"],
        sigma_min=float(spectral["sigma_min"]),
        sigma_max=float(spectral["sigma_max"]),
        gap=spectral["gap"],
        spectral_kind_used=spectral["kind"],
        matrix_shape=spectral["matrix_shape"],
        matrix_is_hermitian=spectral["matrix_is_hermitian"],
        matrix_scale_provided=True,
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
    out = _make_result(spec, target, metric, c)
    return replace(out, all_results=(out,))


def _exact_constant_if_any(
    spec: dict,
    target: float,
    metric: str,
) -> FourierExtensionResult | None:
    """The exact constant series when the eigenvalue interval is a single point.

    A point eigenvalue interval means H = lambda I.  Equal singular values do
    not make A a multiple of the identity (A may be any scaled unitary), so
    singular-value fits are fitted as usual.
    """
    s = spec["spectral"]
    if s["kind"] != "eigenvalues" or s["sigma_min"] != s["sigma_max"]:
        return None
    return _constant_result(spec, target, metric, _CENTRE_VALUE[spec["problem"]](s["sigma_min"]))


def _lambda_grid(lambdas: Sequence[float] | None) -> np.ndarray:
    if lambdas is None:
        return default_lambda_grid()
    grid = np.asarray(lambdas, dtype=float)
    if grid.ndim != 1 or grid.size == 0 or np.any(grid < 0):
        raise ValueError("lambdas must be a non-empty sequence of positive values.")
    return grid


def _m_grid(m: int | None, m_values: Sequence[int] | int | None) -> np.ndarray:
    """Ascending candidate orders: *m* alone if given, else *m_values*, else the default grid."""
    if m is not None:
        if m < 0:
            raise ValueError("m must be non-negative.")
        return np.array([int(m)], dtype=int)
    if m_values is None:
        return default_m_grid()
    if np.isscalar(m_values):
        if int(m_values) < 0:
            raise ValueError("m_values must be non-negative.")
        return np.array([int(m_values)], dtype=int)
    grid = np.unique(np.asarray(m_values, dtype=int))
    if grid.ndim != 1 or grid.size == 0 or np.any(grid < 0):
        raise ValueError("m_values must be a non-empty sequence of non-negative integers.")
    return grid


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
        One of ``'action'``, ``'inversion'``, ``'exponentiation'``, or ``'square_root'``
        (several aliases accepted; see :func:`~felcu.utils.canonical_problem_name`).
    method:
        Coefficient-selection strategy (see
        :func:`~felcu.canonical_method_name` for aliases):

        - ``'sobolev'`` (default): Sobolev-regularised least squares.
          Near-exponential convergence down to *target_error*, with
          subnormalisation ``O(||f(H)||)``.
        - ``'reflected'``: the Fourier series of g reflected about
          ``x = +-pi/2``.  Optimal subnormalisation
          ``alpha_m = ||f(H)|| - epsilon_m`` for convex endpoint-maximising f
          (action, exponentiation, definite inversion) at the default
          ``eta = 2``, but only ``O(1/m)`` convergence, so the default
          *target_error* is out of reach.
        - ``'arcsine_taylor'``: the Taylor series of ``g(arcsin z)`` with
          ``z = sin x``.  Exponential convergence with factor
          ``sin(pi/eta)`` (requires ``eta > 2``) and subnormalisation
          bounded independently of the error.

        The explicit methods do not support the two-interval fit of
        inversion on an indefinite interval or on singular values, and for
        definite inversion and square root need ``eta < 2 |mu| / delta``.
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
        series, so it is available for action and inversion only.
        ``'auto'`` (default) uses singular values for action and inversion
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
        mode by ``1 + (r|k|)^(2w)``.  Must exceed ``0.5``; the paper's
        bounds cover ``0.5 < w <= 1``.  Defaults to ``1``.
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
        Tolerance for the Hermitian test.

    Returns
    -------
    FourierExtensionResult
        The selected result; ``result.all_results`` contains every candidate.
    """
    problem = canonical_problem_name(problem)
    method = canonical_method_name(method)
    metric = canonical_error_metric(error_metric)
    sobolev_options = {
        "lambdas": lambdas,
        "sobolev_w": sobolev_w,
        "sobolev_r": sobolev_r,
        "order_fit": order_fit,
        "fit_oversample": fit_oversample,
    }
    if method != "sobolev":
        given = [name for name, value in sobolev_options.items() if value is not None]
        if given:
            raise ValueError(f"{', '.join(given)} only apply to method='sobolev', not method='{method}'.")
    sobolev_w = 1.0 if sobolev_w is None else sobolev_w
    order_fit = 1 if order_fit is None else order_fit
    fit_oversample = 8 if fit_oversample is None else fit_oversample
    if target_error < 0:
        raise ValueError("target_error must be positive.")
    if order_fit < 0 or fit_oversample <= 0 or val_n < 3:
        raise ValueError("Require order_fit >= 0, fit_oversample > 0, and val_n >= 3.")
    if not sobolev_w > 0.5:
        raise ValueError("sobolev_w must exceed 0.5.")
    if selection not in {"smallest_m", "smallest_alpha"}:
        raise ValueError("selection must be 'smallest_m' or 'smallest_alpha'.")

    spectral = _spectral_data(problem, matrix, sigma_min, sigma_max, gap, spectral_kind, hermitian_tol)
    spec = _make_spec(problem, spectral, eta, method)

    if method != "sobolev":
        spec = spec | {"sobolev_r": None, "sobolev_w": None}
    else:
        if sobolev_r is None:
            sobolev_r = default_sobolev_r(
                problem,
                sigma_min=spectral["sigma_min"],
                sigma_max=spectral["sigma_max"],
                gap=spectral["gap"],
                spectral_kind=spectral["kind"],
                hermitian_tol=hermitian_tol,
                eta=spec["eta"],
                sobolev_w=sobolev_w,
            )
        if not 0 < sobolev_r <= 1:
            raise ValueError("sobolev_r must satisfy 0 < sobolev_r <= 1.")
        spec = spec | {"sobolev_r": float(sobolev_r), "sobolev_w": float(sobolev_w)}

    logger.info(
        "solve_fe_coefficients: problem=%s  method=%s  kind=%s  metric=%s  target=%.3e  basis=%s  domain=%s  "
        "eta=%.4f  tau=%s  mu=%.4g",
        problem,
        method,
        spectral["kind"],
        metric,
        target_error,
        spec["basis"],
        spec["domain"],
        spec["eta"],
        f"{spec['tau']:.4f}" if np.isfinite(spec["tau"]) else "inf",
        spec["mu"],
    )

    target = float(target_error)
    exact = _exact_constant_if_any(spec, target, metric)
    if exact is not None:
        logger.info("exact constant solution found, returning immediately")
        return exact

    lambdas = _lambda_grid(lambdas) if method == "sobolev" else None
    m_grid = _m_grid(m, m_values)
    logger.info("sweeping %d m values: %s ... %s", len(m_grid), m_grid[0], m_grid[-1])

    candidates = []
    for i, mi in enumerate(m_grid):
        logger.info("[%d/%d] m=%d ...", i + 1, len(m_grid), mi)
        if method == "sobolev":
            c = _candidate_for_m(
                spec,
                int(mi),
                lambdas,
                target,
                metric,
                order_fit,
                spec["sobolev_r"],
                spec["sobolev_w"],
                fit_oversample,
                val_n,
            )
        else:
            c = _explicit_candidate(spec, int(mi), target, metric, val_n)
        logger.info(
            "[%d/%d] m=%d  done: error=%.3e  feasible=%s  lam=%s  alpha=%.3e",
            i + 1,
            len(m_grid),
            mi,
            c["solve"][metric],
            c["feasible"],
            "-" if c["lambda"] is None else f"{c['lambda']:.3e}",
            c["solve"]["alpha"],
        )
        candidates.append(c)
        if selection == "smallest_m" and _meets_target(c["solve"], metric, target):
            # m_grid is ascending, so no later m can be smaller.
            logger.info("smallest feasible m found, stopping sweep")
            break

    best, converged = _select(candidates, metric, target, selection)
    logger.info("selected: m=%d  error=%.3e  converged=%s", best["m"], best["solve"][metric], converged)

    all_results = tuple(_make_result(spec, target, metric, c) for c in sorted(candidates, key=lambda c: c["m"]))
    selected = _make_result(spec, target, metric, best)
    return replace(selected, all_results=all_results)


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
