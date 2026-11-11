"""Public solver interface for felcu."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace

import numpy as np

from felcu._coefficients import _coefficient_dtype, _scalar, _value
from felcu._explicit import _explicit_candidate
from felcu._linalg import _coefficient_norm
from felcu._logging import logger
from felcu._solver import _candidate_for_m, _meets_target
from felcu.spectral import _make_spec, _spectral_data, default_sobolev_r
from felcu.types import FourierExtensionResult
from felcu.utils import (
    _canonical_polynomial_basis,
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


def _selection_order(spec: dict) -> float:
    """p of the selection score J_p: max(w - 1, 0) for Sobolev, d for reflected, 0 otherwise."""
    if spec["method"] == "sobolev":
        return max(float(spec["sobolev_w"]) - 1.0, 0.0)
    if spec["method"] == "reflected":
        return float(spec["reflection_order"])
    return 0.0


def _score(spec: dict, cand: dict) -> float:
    s = cand["solve"]
    return _coefficient_norm(s["a0"], np.asarray(s["ak"]), np.asarray(s["bk"]), _selection_order(spec))


def _make_result(spec: dict, target: float, metric: str, cand: dict) -> FourierExtensionResult:
    s = cand["solve"]
    spectral = spec["spectral"]
    a0 = _scalar(s["a0"])
    dtype = _coefficient_dtype(a0, s["ak"], s["bk"])
    ak = np.asarray(s["ak"], dtype=dtype)
    bk = np.asarray(s["bk"], dtype=dtype)

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
        x_max=float(spec["x_max"]),
        x_gap=spec["x_gap"],
        sigma_min=float(spectral["sigma_min"]),
        sigma_max=float(spectral["sigma_max"]),
        gap=spectral["gap"],
        spectral_kind_used=spectral["kind"],
        matrix_shape=spectral["matrix_shape"],
        matrix_is_hermitian=spectral["matrix_is_hermitian"],
        polynomial_coefficients=None if spec["polynomial"] is None else spec["polynomial"][0].copy(),
        polynomial_basis=None if spec["polynomial"] is None else spec["polynomial"][1],
        closure_mean=None if cand.get("closure_mean") is None else _scalar(cand["closure_mean"]),
        reflection_order=int(spec["reflection_order"]) if spec["method"] == "reflected" else None,
        selection_order=_selection_order(spec),
        selection_score=_score(spec, cand),
        readout_norm=float(spec["tau"]) * _coefficient_norm(a0, ak, bk, 1.0) if ak.size else 0.0,
    )


def _constant_result(spec: dict, target: float, metric: str, value: float | complex) -> FourierExtensionResult:
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
    return _constant_result(spec, target, metric, _value(spec["problem"], s["sigma_min"], spec["polynomial"]))


def _polynomial(
    problem: str, coefficients: Sequence[complex] | None, basis: str | None
) -> tuple[np.ndarray, str] | None:
    """``(coefficients, basis)`` for the polynomial problem, checked; ``None`` for the others."""
    if problem != "polynomial":
        if coefficients is not None or basis is not None:
            raise ValueError("polynomial_coefficients and polynomial_basis only apply to problem='polynomial'.")
        return None
    if coefficients is None:
        raise ValueError("problem='polynomial' needs polynomial_coefficients, in ascending order.")
    c = np.asarray(coefficients)
    if c.ndim != 1 or c.size == 0 or not np.issubdtype(c.dtype, np.number):
        raise ValueError("polynomial_coefficients must be a non-empty sequence of numbers.")
    c = c.astype(np.complex128 if np.iscomplexobj(c) else float)
    if not np.all(np.isfinite(c)):
        raise ValueError("polynomial_coefficients must be finite.")
    return c, _canonical_polynomial_basis("monomial" if basis is None else basis)


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


def _select(spec: dict, candidates: list[dict], metric: str, target: float, selection: str) -> tuple[dict, bool]:
    """Pick the candidate to return, and whether it meets *target*.

    Among feasible candidates, ``'smallest_norm'`` prefers the lowest
    selection score J_p (see :func:`_selection_order`), ``'smallest_alpha'``
    the lowest subnormalisation alpha = J_0 and ``'smallest_m'`` the lowest
    order, each with ties broken by order, alpha and error.  With none
    feasible, the lowest error wins.
    """
    feasible = [c for c in candidates if _meets_target(c["solve"], metric, target)]
    logger.info("%d/%d candidates are feasible", len(feasible), len(candidates))
    if not feasible:
        return min(candidates, key=lambda c: (c["solve"][metric], c["m"], c["solve"]["alpha"])), False
    if selection == "smallest_m":
        return min(feasible, key=lambda c: (c["m"], c["solve"]["alpha"], c["solve"][metric])), True
    if selection == "smallest_norm" and _selection_order(spec) > 0:
        return min(feasible, key=lambda c: (_score(spec, c), c["m"], c["solve"]["alpha"], c["solve"][metric])), True
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
    selection: str = "smallest_norm",
    hermitian_tol: float = 1e-12,
    polynomial_coefficients: Sequence[complex] | None = None,
    polynomial_basis: str | None = None,
    closure_mean: complex | None = None,
    reflection_order: int | None = None,
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
        ``'polynomial'`` (several aliases accepted; see
        :func:`~felcu.utils.canonical_problem_name`).  ``'polynomial'``
        takes its function from *polynomial_coefficients* and
        *polynomial_basis*.
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
          *target_error* is out of reach.  For ``eta > 2`` the profile is
          reflected about the endpoint where ``|g|`` is larger and closed
          with a quadratic continuation of mean *closure_mean*.
        - ``'arcsine_taylor'``: the Taylor series of ``g(arcsin z)`` with
          ``z = sin x``.  Exponential convergence with factor
          ``sin(pi/eta)`` (requires ``eta > 2``) and subnormalisation
          bounded independently of the error.

        The explicit methods do not support the two-interval fit of
        inversion on an indefinite interval or on singular values.
        Arcsine-Taylor needs ``eta < 2 |mu| / delta`` for definite
        inversion and square root.
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
        ``'smallest_norm'`` (default) picks the feasible order with the
        smallest selection score ``J_p``, with ``p = max(sobolev_w - 1, 0)``
        for Sobolev fits, ``p = reflection_order`` for reflected fits and
        ``p = 0`` for arcsine-Taylor.  ``J_0`` is alpha, and for ``p > 0``
        ``J_p = sum_{k != 0} |k|^p |c_k|``.  ``'smallest_alpha'`` uses alpha
        whatever ``p``.  Both require solving every order in the sweep.
        ``'smallest_m'`` picks the shortest feasible series and
        stops the sweep there, so ``result.all_results`` only holds the
        orders up to it.
    hermitian_tol:
        Tolerance for the Hermitian test.
    polynomial_coefficients:
        For ``problem='polynomial'`` (required there, rejected otherwise):
        the coefficients ``gamma_0, ..., gamma_n``, in ascending order, of
        ``f(z) = sum_j gamma_j B_j(z)``.  They may be complex.  For example,
        ``[1, 2, 3, 4]`` in the monomial basis is ``1 + 2z + 3z^2 + 4z^3``.
    polynomial_basis:
        ``'monomial'`` (default; ``B_j(z) = z^j``), ``'chebyshev'``
        (``T_j(z)``) or ``'legendre'`` (``P_j(z)``).  The basis acts on
        the physical argument ``z`` itself, with no mapping to
        ``[-1, 1]``.  A polynomial is entire, so it takes the entire-function
        default *eta*, and the default Sobolev scale is ``r = 1``.

    closure_mean:
        For ``method='reflected'`` with ``eta > 2`` only: the mean
        ``beta_0`` of the quadratic continuation, which may be complex.
        ``None`` (default) chooses it for each order m to minimise
        ``sum_k |c_k|`` without raising the maximum error above that of
        ``beta_0 = 0``.  To compare orders of one continuation, pass the
        value from ``result.closure_mean`` so that it stays fixed.  With
        ``reflection_order >= 1`` the mean is 0: leave it ``None`` (or
        pass 0).
    reflection_order:
        For ``method='reflected'`` only: the derivative order ``d >= 0``
        of the reflection, independent of *sobolev_w*.  ``0`` (default)
        reflects g itself.  ``d >= 1`` continues ``g^(d)`` with the
        quadratic continuation (so needs ``eta > 2``, default ``3``) and
        integrates the series back d times, matching g at the reflection
        endpoint.  Orders ``d >= 2`` are experimental: the periodic
        integral need not recover the lower derivatives of g, so check the
        achieved error, which is always measured against g itself.

    Returns
    -------
    FourierExtensionResult
        The selected result; ``result.all_results`` contains every candidate.
    """
    problem = canonical_problem_name(problem)
    method = canonical_method_name(method)
    metric = canonical_error_metric(error_metric)
    polynomial = _polynomial(problem, polynomial_coefficients, polynomial_basis)
    if reflection_order is not None:
        if method != "reflected":
            raise ValueError(f"reflection_order only applies to method='reflected', not method='{method}'.")
        if isinstance(reflection_order, bool) or not isinstance(reflection_order, (int, np.integer)):
            raise ValueError("reflection_order must be a non-negative integer.")
        if reflection_order < 0:
            raise ValueError("reflection_order must be a non-negative integer.")
    reflection_order = 0 if reflection_order is None else int(reflection_order)
    if closure_mean is not None:
        if method != "reflected":
            raise ValueError(f"closure_mean only applies to method='reflected', not method='{method}'.")
        if not (isinstance(closure_mean, (int, float, complex, np.number)) and np.isfinite(closure_mean)):
            raise ValueError("closure_mean must be a finite real or complex number, or None.")
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
    if selection not in {"smallest_norm", "smallest_alpha", "smallest_m"}:
        raise ValueError("selection must be 'smallest_norm', 'smallest_alpha' or 'smallest_m'.")

    spectral = _spectral_data(problem, matrix, sigma_min, sigma_max, gap, spectral_kind, hermitian_tol)
    spec = _make_spec(problem, spectral, eta, method, polynomial, reflection_order)
    if closure_mean is not None:
        if reflection_order >= 1 and closure_mean != 0:
            raise ValueError(
                f"reflection_order={reflection_order} needs closure mean 0 (closure_mean=None or 0), "
                f"not {closure_mean!r}: the integrated series has no free mean."
            )
        if spec["eta"] <= 2.0:
            raise ValueError(
                "closure_mean sets the quadratic continuation, which reflected uses only for eta > 2; "
                f"this fit has eta={spec['eta']:.6g}."
            )
        spec["closure_mean"] = _scalar(closure_mean)

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

    best, converged = _select(spec, candidates, metric, target, selection)
    logger.info("selected: m=%d  error=%.3e  converged=%s", best["m"], best["solve"][metric], converged)

    all_results = tuple(_make_result(spec, target, metric, c) for c in sorted(candidates, key=lambda c: c["m"]))
    selected = _make_result(spec, target, metric, best)
    return replace(selected, all_results=all_results)


def compute_fe_coefficients(*args, **kwargs) -> tuple[float | complex, np.ndarray, np.ndarray]:
    """Convenience wrapper; returns ``(a0, ak, bk)`` directly.

    Takes the same arguments as :func:`solve_fe_coefficients` and discards
    everything but the coefficients.  Use :func:`solve_fe_coefficients`
    when you also need ``tau``, the achieved error, or whether the fit
    converged.

    Returns
    -------
    a0 : float or complex
        Constant term.
    ak, bk : numpy.ndarray
        Sine and cosine coefficients (copies), each of length ``m``;
        complex for a complex-valued f.
    """
    result = solve_fe_coefficients(*args, **kwargs)
    return result.a0, result.ak.copy(), result.bk.copy()
