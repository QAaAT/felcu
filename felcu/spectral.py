"""Spectral information and problem-spec construction for felcu."""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np

from felcu.utils import canonical_problem_name

PI = float(np.pi)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _is_hermitian(A: np.ndarray, tol: float) -> bool:
    return bool(np.allclose(A, A.conj().T, rtol=tol, atol=tol))


def _spectral_kind(
    problem: str,
    shifted: bool,
    requested: str,
    matrix_is_hermitian: bool | None,
) -> str:
    if requested not in {"auto", "eigenvalues", "singular_values"}:
        raise ValueError("spectral_kind must be 'auto', 'eigenvalues', or 'singular_values'.")
    if requested != "auto":
        return requested
    if problem == "inversion" and not shifted:
        return "singular_values"
    if problem == "action" and not shifted and matrix_is_hermitian is False:
        return "singular_values"
    return "eigenvalues"


def _check_spectral_choice(
    problem: str,
    shifted: bool,
    kind: str,
    matrix_is_hermitian: bool | None,
) -> None:
    if shifted and kind != "eigenvalues":
        raise ValueError("Shifted constructions use an eigenvalue interval.")
    if problem == "exponentiation" and kind != "eigenvalues":
        raise ValueError("Exponentiation uses an eigenvalue interval.")
    if problem == "inversion" and not shifted and kind != "singular_values":
        raise ValueError("Unshifted inversion uses singular values.")
    if kind == "eigenvalues" and matrix_is_hermitian is False:
        raise ValueError("Eigenvalue-based constructions require a Hermitian matrix.")


def _definite_abs_bounds(lo: float, hi: float) -> tuple[float, float]:
    if lo > hi:
        raise ValueError("Require sigma_min <= sigma_max.")
    if lo == 0 or hi == 0 or lo * hi < 0:
        raise ValueError("This construction requires a definite Hermitian interval.")
    return min(abs(lo), abs(hi)), max(abs(lo), abs(hi))


def _spectral_data(
    problem: str,
    shifted: bool,
    matrix: np.ndarray | Sequence[Sequence[complex]] | None,
    sigma_min: float | None,
    sigma_max: float | None,
    spectral_kind: str,
    hermitian_tol: float,
    allow_missing: bool,
) -> dict | None:
    if matrix is not None and (sigma_min is not None or sigma_max is not None):
        raise ValueError("Provide either matrix or sigma_min/sigma_max, not both.")
    if matrix is None and sigma_min is None and sigma_max is None:
        if allow_missing:
            return None
        raise ValueError("Provide a matrix or both sigma_min and sigma_max.")
    if matrix is None and (sigma_min is None or sigma_max is None):
        raise ValueError("Provide both sigma_min and sigma_max.")

    if matrix is not None:
        A = np.asarray(matrix)
        if A.ndim != 2 or A.shape[0] != A.shape[1] or A.shape[0] == 0:
            raise ValueError("matrix must be a non-empty square array.")
        A = A.astype(np.complex128 if np.iscomplexobj(A) else float)
        hermitian = _is_hermitian(A, hermitian_tol)
        kind = _spectral_kind(problem, shifted, spectral_kind, hermitian)
        _check_spectral_choice(problem, shifted, kind, hermitian)
        values = np.linalg.eigvalsh(A) if kind == "eigenvalues" else np.linalg.svd(A, compute_uv=False)
        lo = float(np.min(values))
        hi = float(np.max(values))
        shape = (int(A.shape[0]), int(A.shape[1]))
    else:
        hermitian = None
        kind = _spectral_kind(problem, shifted, spectral_kind, None)
        _check_spectral_choice(problem, shifted, kind, hermitian)
        lo = float(sigma_min)
        hi = float(sigma_max)
        shape = None

    if kind == "singular_values":
        if lo < 0 or hi < 0 or lo > hi:
            raise ValueError("Singular-value bounds require 0 <= sigma_min <= sigma_max.")
        if problem == "inversion" and lo <= 0:
            raise ValueError("Inversion requires sigma_min > 0.")
    else:
        if lo > hi:
            raise ValueError("Require sigma_min <= sigma_max.")
        if problem == "inversion" and shifted:
            _definite_abs_bounds(lo, hi)

    return {
        "sigma_min": lo,
        "sigma_max": hi,
        "kind": kind,
        "matrix_shape": shape,
        "matrix_is_hermitian": hermitian,
    }


# ---------------------------------------------------------------------------
# Public: default eta
# ---------------------------------------------------------------------------


def default_eta(
    problem: str,
    *,
    shifted: bool = False,
    matrix: np.ndarray | Sequence[Sequence[complex]] | None = None,
    sigma_min: float | None = None,
    sigma_max: float | None = None,
    spectral_kind: str = "auto",
    hermitian_tol: float = 1e-12,
) -> float:
    """Return the default extension parameter eta for the given problem.

    The series is fitted on ``[-pi/eta, pi/eta]``, so ``eta > 1`` sets how
    much of the period is left free.  The defaults are

    - action and exponentiation: ``2``;
    - unshifted inversion: ``1 + 1/kappa``, with
      ``kappa = sigma_max / sigma_min``;
    - shifted inversion: ``(kappa + 1) / (kappa - 1)``, with *kappa* the
      ratio of the largest to smallest ``|lambda|`` (``2`` if
      ``kappa == 1``).

    Parameters
    ----------
    problem:
        One of ``'action'``, ``'inversion'``, or ``'exponentiation'`` (or
        an alias; see :func:`~felcu.canonical_problem_name`).
    shifted:
        Use the shifted (centred) construction.
    matrix:
        Square matrix to read the spectral interval from.  Mutually
        exclusive with *sigma_min* / *sigma_max*.
    sigma_min, sigma_max:
        Spectral interval bounds.
    spectral_kind:
        ``'auto'``, ``'eigenvalues'``, or ``'singular_values'``.
    hermitian_tol:
        Tolerance for the Hermitian test.

    Returns
    -------
    float
        The default *eta*.

    Raises
    ------
    ValueError
        For inversion, if no spectral information is given or the interval
        is invalid (see :func:`~felcu.solve_fe_coefficients`).

    Notes
    -----
    Spectral information is only used for inversion; for action and
    exponentiation it is ignored.
    """
    problem = canonical_problem_name(problem)
    if problem in {"action", "exponentiation"}:
        return 2.0

    s = _spectral_data(problem, shifted, matrix, sigma_min, sigma_max, spectral_kind, hermitian_tol, False)
    assert s is not None
    lo, hi = s["sigma_min"], s["sigma_max"]
    if shifted:
        lo, hi = _definite_abs_bounds(lo, hi)
    kappa = hi / lo
    if shifted:
        return 2.0 if abs(kappa - 1.0) < 1e-15 else (kappa + 1.0) / (kappa - 1.0)
    return 1.0 + 1.0 / kappa


# ---------------------------------------------------------------------------
# Public: default Sobolev scale r
# ---------------------------------------------------------------------------


def default_sobolev_r(
    problem: str,
    *,
    shifted: bool = False,
    matrix: np.ndarray | Sequence[Sequence[complex]] | None = None,
    sigma_min: float | None = None,
    sigma_max: float | None = None,
    spectral_kind: str = "auto",
    hermitian_tol: float = 1e-12,
    eta: float | None = None,
) -> float:
    """Return the default scale r of the H_r^w regulariser for the given problem.

    The regulariser weights the k-th harmonic by ``(1 + r|k|)^w``, so *r*
    sets the frequency at which the penalty starts to grow.

    Parameters
    ----------
    problem:
        One of ``'action'``, ``'inversion'``, or ``'exponentiation'`` (or
        an alias; see :func:`~felcu.canonical_problem_name`).
    shifted:
        Use the shifted (centred) construction.
    matrix:
        Square matrix to read the spectral interval from.  Mutually
        exclusive with *sigma_min* / *sigma_max*.
    sigma_min, sigma_max:
        Spectral interval bounds.  May be omitted, in which case no
        spectral information is available.
    spectral_kind:
        ``'auto'``, ``'eigenvalues'``, or ``'singular_values'``.
    hermitian_tol:
        Tolerance for the Hermitian test.
    eta:
        Extension parameter of the fit.  Defaults to
        :func:`~felcu.default_eta` for the same problem.

    Returns
    -------
    float
        The default *r*, which must lie in ``(0, 1]``.

    Notes
    -----
    The defaults are

    - action: ``1``;
    - inversion: ``1/kappa``, with *kappa* the condition number of the
      spectral interval;
    - exponentiation: ``1 / (1 + eta * w / pi)``, with *w* the half-width
      of the spectral interval.

    Without spectral information the default is ``1``.
    """
    problem = canonical_problem_name(problem)
    s = _spectral_data(problem, shifted, matrix, sigma_min, sigma_max, spectral_kind, hermitian_tol, True)
    if s is None:
        # No spectral information: fitting the unscaled function.
        return 1.0

    lo, hi = s["sigma_min"], s["sigma_max"]
    half_width = 0.5 * (hi - lo)
    radius = max(abs(lo), abs(hi))
    # Condition number; infinite when the interval touches or contains 0.
    kappa = radius / min(abs(lo), abs(hi)) if lo * hi > 0 else math.inf

    if eta is None:
        eta = default_eta(
            problem,
            shifted=shifted,
            sigma_min=lo,
            sigma_max=hi,
            spectral_kind=s["kind"],
            hermitian_tol=hermitian_tol,
        )

    if problem == "action":
        return 1.0
    if problem == "inversion":
        return 1.0 / kappa
    if problem == "exponentiation":
        return 1.0 / (1.0 + eta * half_width / np.pi)
    return 1.0


# ---------------------------------------------------------------------------
# Internal: build the problem spec dict used throughout the solver
# ---------------------------------------------------------------------------


def _make_spec(
    problem: str,
    shifted: bool,
    spectral: dict | None,
    eta: float | None,
    hermitian_tol: float,
) -> dict:
    if shifted and spectral is None:
        raise ValueError("Shifted constructions require matrix-scale information.")
    if problem == "inversion" and spectral is None:
        raise ValueError("Inversion requires matrix-scale information so that the fitted interval avoids the pole.")

    if eta is None:
        if spectral is None:
            eta = default_eta(problem)
        else:
            eta = default_eta(
                problem,
                shifted=shifted,
                sigma_min=spectral["sigma_min"],
                sigma_max=spectral["sigma_max"],
                spectral_kind=spectral["kind"],
                hermitian_tol=hermitian_tol,
            )
    eta = float(eta)
    if eta <= 1.0:
        raise ValueError("eta must exceed 1.")

    base = {
        "problem": problem,
        "shifted": shifted,
        "eta": eta,
        "x_max": PI / eta,
        "x_min": None,
        "spectral": spectral,
        "matrix_scale_provided": spectral is not None,
        "domain": "single",
        "tau": 1.0,
        "mu": 0.0,
    }
    # The fitted target is g(x) = F(x / tau + mu) (see _coefficients._target)
    # and the baseline is the constant g would be without the scaled variable.
    if spectral is None:
        if problem == "action":
            return base | {"basis": "odd_sine", "baseline": 0.0}
        return base | {"basis": "full", "baseline": 1.0}

    lo = spectral["sigma_min"]
    hi = spectral["sigma_max"]
    mu = 0.5 * (lo + hi)
    halfwidth = 0.5 * (hi - lo)
    radius = hi if spectral["kind"] == "singular_values" else max(abs(lo), abs(hi))

    def scale(width: float) -> float:
        return math.inf if width == 0 else PI / (eta * width)

    if problem == "inversion" and not shifted:
        tau = PI / (eta * hi)
        return base | {"basis": "odd_sine", "domain": "double", "tau": tau, "x_min": tau * lo, "baseline": 0.0}

    if not shifted:
        tau = scale(radius)
        if problem == "action":
            return base | {"basis": "odd_sine", "tau": tau, "baseline": 0.0}
        return base | {"basis": "full", "tau": tau, "baseline": 1.0}

    tau = scale(halfwidth)
    if problem == "action":
        return base | {"basis": "const_plus_sine", "tau": tau, "mu": mu, "baseline": mu}
    if problem == "inversion":
        return base | {"basis": "full", "tau": tau, "mu": mu, "baseline": 1.0 / mu}
    return base | {"basis": "full", "tau": tau, "mu": mu, "baseline": math.exp(mu)}
