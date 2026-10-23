"""Spectral information and problem-spec construction for felcu."""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np

from felcu.utils import canonical_problem_name

PI = float(np.pi)

# Functions with a branch point at 0, fitted on a positive-definite interval
# by the shifted construction only (Sec. VI D of the paper).
POSITIVE_DEFINITE_PROBLEMS = frozenset({"square_root"})

# Functions fitted on an eigenvalue interval only: they have no unshifted
# (singular-value) construction.
EIGENVALUE_ONLY_PROBLEMS = frozenset({"exponentiation"}) | POSITIVE_DEFINITE_PROBLEMS


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _is_hermitian(A: np.ndarray, tol: float) -> bool:
    return bool(np.allclose(A, A.conj().T, rtol=tol, atol=tol))


def _spectral_kind(
    problem: str,
    shifted: bool | None,
    requested: str,
    matrix_is_hermitian: bool | None,
) -> str:
    if requested not in {"auto", "eigenvalues", "singular_values"}:
        raise ValueError("spectral_kind must be 'auto', 'eigenvalues', or 'singular_values'.")
    if requested != "auto":
        return requested
    if shifted is not None:
        return "eigenvalues" if shifted else "singular_values"
    if problem == "inversion":
        return "singular_values"
    if problem == "action" and matrix_is_hermitian is False:
        return "singular_values"
    return "eigenvalues"


def _check_spectral_choice(
    problem: str,
    shifted: bool,
    kind: str,
    matrix_is_hermitian: bool | None,
) -> None:
    if problem in EIGENVALUE_ONLY_PROBLEMS and kind != "eigenvalues":
        raise ValueError(f"{problem} uses the shifted construction on an eigenvalue interval.")
    if shifted and kind != "eigenvalues":
        raise ValueError("Shifted constructions use an eigenvalue interval.")
    if not shifted and kind != "singular_values":
        raise ValueError("Unshifted constructions use a singular-value interval; eigenvalue intervals are shifted.")
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
    shifted: bool | None,
    matrix: np.ndarray | Sequence[Sequence[complex]] | None,
    sigma_min: float | None,
    sigma_max: float | None,
    spectral_kind: str,
    hermitian_tol: float,
) -> dict:
    """The spectral interval, its kind, and the construction it implies.

    *shifted* ``None`` picks the construction from the kind: shifted for
    eigenvalues, unshifted for singular values.
    """
    if matrix is not None and (sigma_min is not None or sigma_max is not None):
        raise ValueError("Provide either matrix or sigma_min/sigma_max, not both.")
    if matrix is None and (sigma_min is None or sigma_max is None):
        raise ValueError("Provide a matrix or both sigma_min and sigma_max.")

    if matrix is not None:
        A = np.asarray(matrix)
        if A.ndim != 2 or A.shape[0] != A.shape[1] or A.shape[0] == 0:
            raise ValueError("matrix must be a non-empty square array.")
        A = A.astype(np.complex128 if np.iscomplexobj(A) else float)
        hermitian = _is_hermitian(A, hermitian_tol)
        kind = _spectral_kind(problem, shifted, spectral_kind, hermitian)
        if shifted is None:
            shifted = kind == "eigenvalues"
        _check_spectral_choice(problem, shifted, kind, hermitian)
        values = np.linalg.eigvalsh(A) if kind == "eigenvalues" else np.linalg.svd(A, compute_uv=False)
        lo = float(np.min(values))
        hi = float(np.max(values))
        shape = (int(A.shape[0]), int(A.shape[1]))
    else:
        hermitian = None
        kind = _spectral_kind(problem, shifted, spectral_kind, None)
        if shifted is None:
            shifted = kind == "eigenvalues"
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
        if problem == "inversion":
            _definite_abs_bounds(lo, hi)
        if problem in POSITIVE_DEFINITE_PROBLEMS and lo <= 0:
            raise ValueError(f"{problem} requires a positive-definite interval, 0 < sigma_min <= sigma_max.")

    return {
        "sigma_min": lo,
        "sigma_max": hi,
        "kind": kind,
        "shifted": shifted,
        "matrix_shape": shape,
        "matrix_is_hermitian": hermitian,
    }


# ---------------------------------------------------------------------------
# Public: default eta
# ---------------------------------------------------------------------------


def default_eta(
    problem: str,
    *,
    shifted: bool | None = None,
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
    - unshifted (singular-value) inversion: ``1 + 1/kappa``, with
      ``kappa = sigma_max / sigma_min``;
    - shifted inversion and square root:
      ``(kappa + 1) / (kappa - 1)``, with *kappa* the ratio of the largest
      to smallest ``|lambda|`` (``2`` if ``kappa == 1``).

    Parameters
    ----------
    problem:
        One of ``'action'``, ``'inversion'``, ``'exponentiation'``, or ``'square_root'`` (or
        an alias; see :func:`~felcu.canonical_problem_name`).
    shifted:
        Use the shifted (centred) construction, which fits an eigenvalue
        interval, or the unshifted one, which fits a singular-value
        interval.  ``None`` follows *spectral_kind*.
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
        For inversion and square root, if no spectral information is given or the interval
        is invalid (see :func:`~felcu.solve_fe_coefficients`).

    Notes
    -----
    Spectral information is only used for inversion and square root; for
    action and exponentiation it is ignored.  The square root uses the
    shifted-inversion rule, which places the branch point at 0 on the
    periodic boundary.
    """
    problem = canonical_problem_name(problem)
    if problem in {"action", "exponentiation"}:
        return 2.0

    s = _spectral_data(problem, shifted, matrix, sigma_min, sigma_max, spectral_kind, hermitian_tol)
    lo, hi = s["sigma_min"], s["sigma_max"]
    if s["shifted"]:
        lo, hi = _definite_abs_bounds(lo, hi)
    kappa = hi / lo
    if s["shifted"]:
        return 2.0 if abs(kappa - 1.0) < 1e-15 else (kappa + 1.0) / (kappa - 1.0)
    return 1.0 + 1.0 / kappa


# ---------------------------------------------------------------------------
# Public: default Sobolev scale r
# ---------------------------------------------------------------------------


def default_sobolev_r(
    problem: str,
    *,
    shifted: bool | None = None,
    matrix: np.ndarray | Sequence[Sequence[complex]] | None = None,
    sigma_min: float | None = None,
    sigma_max: float | None = None,
    spectral_kind: str = "auto",
    hermitian_tol: float = 1e-12,
    eta: float | None = None,
    sobolev_w: float = 1.0,
) -> float:
    """Return the default scale r of the H_r^w regulariser for the given problem.

    The regulariser weights the k-th Fourier mode by ``1 + (r|k|)^(2w)``, so
    *r* should match the shortest length scale the fitted function g has
    to resolve.  The defaults follow the scale-adapted choices of
    Brearley, Howarth and Adcock (Sec. VI):

    - action: ``r_id = 1``, since g is linear;
    - exponentiation: ``r_exp = 1 / (1 + 1/tau)``, since ``g' = g / tau``.
      With ``tau = pi / (eta delta)`` for the half-width ``delta``, this
      is ``1 / (1 + eta delta / pi)``;
    - inversion: the gap to the pole, ``r_def = min(1, 2 pi / (kappa + 1))``
      for the shifted (definite) fit and ``r_indef = min(1, pi / (kappa + 1))``
      for the unshifted (singular-value) two-interval fit;
    - square root: ``r_sqrt = kappa^(-(2w - 1) / (2w))``, which balances the
      two contributions to the extension's H_r^w norm (``kappa^(-1/2)`` for
      ``w = 1``).

    Parameters
    ----------
    problem:
        One of ``'action'``, ``'inversion'``, ``'exponentiation'``, or ``'square_root'`` (or
        an alias; see :func:`~felcu.canonical_problem_name`).
    shifted:
        Use the shifted (centred) construction, which fits an eigenvalue
        interval, or the unshifted one, which fits a singular-value
        interval.  ``None`` follows *spectral_kind*.
    matrix:
        Square matrix to read the spectral interval from.  Mutually
        exclusive with *sigma_min* / *sigma_max*.
    sigma_min, sigma_max:
        Spectral interval bounds.
    spectral_kind:
        ``'auto'``, ``'eigenvalues'``, or ``'singular_values'``.
    hermitian_tol:
        Tolerance for the Hermitian test.
    eta:
        Extension parameter of the fit.  Defaults to
        :func:`~felcu.default_eta` for the same problem.
    sobolev_w:
        Order *w* of the regulariser; only the square-root rule uses it.

    Returns
    -------
    float
        The default *r*, which lies in ``(0, 1]``.

    Raises
    ------
    ValueError
        If no spectral information is given, or it is invalid (see
        :func:`~felcu.solve_fe_coefficients`).
    """
    problem = canonical_problem_name(problem)
    s = _spectral_data(problem, shifted, matrix, sigma_min, sigma_max, spectral_kind, hermitian_tol)

    if problem == "action":
        return 1.0
    if problem == "exponentiation":
        tau = _make_spec(problem, s, eta, hermitian_tol)["tau"]
        return 1.0 / (1.0 + 1.0 / tau)

    # The remaining problems have a pole or branch point at 0, and the
    # interval is definite (shifted) or of singular values.
    lo, hi = abs(s["sigma_min"]), abs(s["sigma_max"])
    kappa = max(lo, hi) / min(lo, hi)
    if problem == "square_root":
        return kappa ** (-(2.0 * sobolev_w - 1.0) / (2.0 * sobolev_w))
    return min(1.0, (2.0 if s["shifted"] else 1.0) * PI / (kappa + 1.0))


# ---------------------------------------------------------------------------
# Internal: build the problem spec dict used throughout the solver
# ---------------------------------------------------------------------------


def _make_spec(
    problem: str,
    spectral: dict,
    eta: float | None,
    hermitian_tol: float,
) -> dict:
    shifted = spectral["shifted"]
    if eta is None:
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
        "matrix_scale_provided": True,
        "domain": "single",
        "tau": 1.0,
        "mu": 0.0,
    }
    # The fitted target is g(x) = F(x / tau + mu) (see _coefficients._target)
    # and the baseline is the constant g would be without the scaled variable.
    lo = spectral["sigma_min"]
    hi = spectral["sigma_max"]

    # Unshifted fits act on singular values, so g must be odd.
    if not shifted:
        tau = math.inf if hi == 0 else PI / (eta * hi)
        if problem == "inversion":
            return base | {"basis": "odd_sine", "domain": "double", "tau": tau, "x_min": tau * lo, "baseline": 0.0}
        return base | {"basis": "odd_sine", "tau": tau, "baseline": 0.0}

    mu = 0.5 * (lo + hi)
    halfwidth = 0.5 * (hi - lo)
    tau = math.inf if halfwidth == 0 else PI / (eta * halfwidth)
    if problem == "action":
        return base | {"basis": "const_plus_sine", "tau": tau, "mu": mu, "baseline": mu}
    if problem == "inversion":
        return base | {"basis": "full", "tau": tau, "mu": mu, "baseline": 1.0 / mu}
    if problem == "square_root":
        return base | {"basis": "full", "tau": tau, "mu": mu, "baseline": math.sqrt(mu)}
    return base | {"basis": "full", "tau": tau, "mu": mu, "baseline": math.exp(mu)}
