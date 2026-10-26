"""Spectral information and problem-spec construction for felcu."""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np

from felcu._coefficients import _value
from felcu.utils import canonical_method_name, canonical_problem_name

PI = float(np.pi)

# Functions with a branch point at 0, fitted on a positive-definite interval
# only (Sec. VI D of the paper).
POSITIVE_DEFINITE_PROBLEMS = frozenset({"square_root"})

# Functions analytic on the whole plane.  The others have a pole or branch
# point at 0 that bounds how far the explicit methods may extend them.
ENTIRE_PROBLEMS = frozenset({"action", "exponentiation"})

# Odd functions.  Only these can act on singular values, through the odd
# series that the Hermitian dilation of a general matrix needs.
ODD_PROBLEMS = frozenset({"action", "inversion"})

# Default eta of the arcsine-Taylor construction for an entire f (Sec. VI).
ARCSINE_TAYLOR_ETA = 3.0


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _is_hermitian(A: np.ndarray, tol: float) -> bool:
    return bool(np.allclose(A, A.conj().T, rtol=tol, atol=tol))


def _resolve_spectral_kind(problem: str, requested: str, matrix_is_hermitian: bool | None) -> str:
    """``'eigenvalues'`` or ``'singular_values'``: which spectral data the fit acts on."""
    if requested not in {"auto", "eigenvalues", "singular_values"}:
        raise ValueError("spectral_kind must be 'auto', 'eigenvalues', or 'singular_values'.")
    kind = requested
    if kind == "auto":
        kind = "singular_values" if problem in ODD_PROBLEMS and matrix_is_hermitian is False else "eigenvalues"
    if kind == "singular_values" and problem not in ODD_PROBLEMS:
        raise ValueError(f"{problem} uses an eigenvalue interval: singular values only support action and inversion.")
    if kind == "eigenvalues" and matrix_is_hermitian is False:
        raise ValueError("Eigenvalue intervals require a Hermitian matrix; use spectral_kind='singular_values'.")
    return kind


def _eigenvalue_gap(values: np.ndarray) -> tuple[float, float] | None:
    """The eigenvalues either side of 0, if *values* has both signs."""
    negative, positive = values[values < 0], values[values > 0]
    if negative.size == 0 or positive.size == 0:
        return None
    return float(np.max(negative)), float(np.min(positive))


def _check_inversion_interval(lo: float, hi: float, gap: tuple[float, float] | None) -> None:
    if lo > 0 or hi < 0:
        if gap is not None:
            raise ValueError("gap only applies to an indefinite interval, sigma_min < 0 < sigma_max.")
        return
    if lo == 0 or hi == 0:
        raise ValueError("Inversion requires an interval clear of 0: sigma_min and sigma_max must be nonzero.")
    if gap is None:
        raise ValueError(
            "Inversion on an indefinite interval needs the gap around 0: pass gap=(a, b), the eigenvalues "
            "closest to 0 on either side."
        )
    a, b = gap
    if not lo <= a < 0 < b <= hi:
        raise ValueError("gap=(a, b) requires sigma_min <= a < 0 < b <= sigma_max.")


def _spectral_data(
    problem: str,
    matrix: np.ndarray | Sequence[Sequence[complex]] | None,
    sigma_min: float | None,
    sigma_max: float | None,
    gap: tuple[float, float] | None,
    spectral_kind: str,
    hermitian_tol: float,
) -> dict:
    """The spectral interval, and the set the fit covers in the spectral variable.

    Singular values are fitted through the Hermitian dilation of the matrix,
    whose eigenvalues are their negatives and themselves, so the fitted set is
    symmetric about 0.
    """
    if matrix is not None and (sigma_min is not None or sigma_max is not None or gap is not None):
        raise ValueError("Provide either matrix or sigma_min/sigma_max, not both.")
    if matrix is None and (sigma_min is None or sigma_max is None):
        raise ValueError("Provide a matrix or both sigma_min and sigma_max.")

    A = None
    hermitian = None
    if matrix is not None:
        A = np.asarray(matrix)
        if A.ndim != 2 or A.shape[0] != A.shape[1] or A.shape[0] == 0:
            raise ValueError("matrix must be a non-empty square array.")
        A = A.astype(np.complex128 if np.iscomplexobj(A) else float)
        hermitian = _is_hermitian(A, hermitian_tol)

    kind = _resolve_spectral_kind(problem, spectral_kind, hermitian)
    if A is None:
        lo, hi, shape = float(sigma_min), float(sigma_max), None
        gap = None if gap is None else (float(gap[0]), float(gap[1]))
    else:
        values = np.linalg.eigvalsh(A) if kind == "eigenvalues" else np.linalg.svd(A, compute_uv=False)
        if problem == "inversion" and np.any(values == 0):
            raise ValueError("Inversion requires a nonsingular matrix: it has an eigenvalue or singular value of 0.")
        lo, hi, shape = float(np.min(values)), float(np.max(values)), (int(A.shape[0]), int(A.shape[1]))
        if kind == "eigenvalues" and problem == "inversion":
            gap = _eigenvalue_gap(values)
    if lo > hi:
        raise ValueError("Require sigma_min <= sigma_max.")
    if gap is not None and not (kind == "eigenvalues" and problem == "inversion"):
        raise ValueError("gap only applies to inversion on an eigenvalue interval.")

    if kind == "singular_values":
        if lo < 0:
            raise ValueError("Singular-value bounds require 0 <= sigma_min <= sigma_max.")
        if problem == "inversion" and lo == 0:
            raise ValueError("Inversion requires sigma_min > 0.")
        fit_gap = (-lo, lo) if problem == "inversion" else None
        fit_lo, fit_hi = -hi, hi
    else:
        if problem == "inversion":
            _check_inversion_interval(lo, hi, gap)
        if problem in POSITIVE_DEFINITE_PROBLEMS and lo <= 0:
            raise ValueError(f"{problem} requires a positive-definite interval, 0 < sigma_min <= sigma_max.")
        fit_gap = gap
        fit_lo, fit_hi = lo, hi

    return {
        "kind": kind,
        "sigma_min": lo,
        "sigma_max": hi,
        "gap": gap,
        "fit_lo": fit_lo,
        "fit_hi": fit_hi,
        "fit_gap": fit_gap,
        "matrix_shape": shape,
        "matrix_is_hermitian": hermitian,
    }


def _check_method(method: str, spectral: dict) -> None:
    if method != "sobolev" and spectral["fit_gap"] is not None:
        raise ValueError(
            f"method='{method}' does not support a two-interval fit (inversion on an indefinite interval or on "
            "singular values): the pole lies between the intervals.  Use method='sobolev'."
        )


def _check_explicit_eta(spec: dict) -> None:
    """Check that *eta* suits the reflected or arcsine-Taylor construction of *spec*.

    Both evaluate g on [-pi/2, pi/2], i.e. f on [mu - eta delta/2, mu + eta delta/2],
    which must stay clear of the pole or branch point at 0.
    """
    method, eta = spec["method"], spec["eta"]
    if method == "reflected" and eta < 2.0:
        raise ValueError("The reflected extension requires eta >= 2.")
    if method == "arcsine_taylor" and eta <= 2.0:
        raise ValueError("The arcsine-Taylor construction requires eta > 2.")
    if spec["problem"] in ENTIRE_PROBLEMS or not math.isfinite(spec["tau"]):
        return
    delta = PI / (eta * spec["tau"])
    bound = 2.0 * abs(spec["mu"]) / delta
    if eta >= bound:
        raise ValueError(
            f"method='{method}' with eta={eta:.6g} extends {spec['problem']} past its singularity at 0; "
            f"eta must be below 2|mu|/delta = {bound:.6g}."
        )


def _centre_and_halfwidth(spectral: dict) -> tuple[float, float]:
    lo, hi = spectral["fit_lo"], spectral["fit_hi"]
    return 0.5 * (lo + hi), 0.5 * (hi - lo)


def _default_eta(problem: str, method: str, spectral: dict | None) -> float:
    """The default eta; *spectral* is only read, and may only be omitted, for an entire f."""
    if method == "reflected":
        return 2.0
    if problem in ENTIRE_PROBLEMS:
        return ARCSINE_TAYLOR_ETA if method == "arcsine_taylor" else 2.0

    mu, delta = _centre_and_halfwidth(spectral)
    gap = spectral["fit_gap"]
    if gap is not None:
        # The gap around the pole and the free part of the period are equally wide.
        return 1.0 + (gap[1] - gap[0]) / (2.0 * delta)
    # The pole or branch point at 0 lies at x = -tau mu, on the periodic
    # boundary when eta = |mu| / delta.
    if method == "arcsine_taylor":
        return ARCSINE_TAYLOR_ETA if delta == 0 else min(ARCSINE_TAYLOR_ETA, 1.0 + abs(mu) / delta)
    return 2.0 if delta == 0 else abs(mu) / delta


# ---------------------------------------------------------------------------
# Public: default eta
# ---------------------------------------------------------------------------


def default_eta(
    problem: str,
    *,
    method: str = "sobolev",
    matrix: np.ndarray | Sequence[Sequence[complex]] | None = None,
    sigma_min: float | None = None,
    sigma_max: float | None = None,
    gap: tuple[float, float] | None = None,
    spectral_kind: str = "auto",
    hermitian_tol: float = 1e-12,
) -> float:
    """Return the default extension parameter eta for the given problem and method.

    The series is fitted on ``[-pi/eta, pi/eta]``, so ``eta > 1`` sets how
    much of the period is left free.  With ``mu`` and ``delta`` the centre
    and half-width of the fitted spectrum, the defaults for the
    Sobolev-regularised method are

    - action and exponentiation: ``2``;
    - inversion on a definite interval, and square root:
      ``|mu| / delta = (kappa + 1) / (kappa - 1)``, which puts the pole or
      branch point at 0 on the periodic boundary (``2`` if ``kappa == 1``);
    - inversion on an indefinite interval or on singular values:
      ``1 + (b - a) / (2 delta)``, with ``(a, b)`` the gap around the pole,
      which leaves the gap and the free part of the period equally wide.
      For singular values (or a symmetric gap) this is ``1 + 1/kappa``.

    The reflected extension always defaults to ``2``, which maps the
    spectral endpoints onto the reflection points and gives the optimal
    subnormalisation.  Arcsine-Taylor needs ``eta > 2``: it defaults to
    ``3`` for action and exponentiation, and to
    ``min(3, 1 + |mu|/delta)`` for definite inversion and square root, the
    midpoint of the admissible range ``2 < eta < 2 |mu|/delta``.

    Parameters
    ----------
    problem:
        One of ``'action'``, ``'inversion'``, ``'exponentiation'``, or ``'square_root'`` (or
        an alias; see :func:`~felcu.canonical_problem_name`).
    method:
        ``'sobolev'``, ``'reflected'``, or ``'arcsine_taylor'`` (or an
        alias; see :func:`~felcu.canonical_method_name`).
    matrix:
        Square matrix to read the spectral interval from.  Mutually
        exclusive with *sigma_min* / *sigma_max* / *gap*.
    sigma_min, sigma_max:
        Spectral interval bounds.
    gap:
        Eigenvalue gap ``(a, b)`` around 0, for inversion on an indefinite
        interval; see :func:`~felcu.solve_fe_coefficients`.
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
        For inversion and square root, if no spectral information is given
        or the interval is invalid (see :func:`~felcu.solve_fe_coefficients`),
        or for a two-interval inversion with an explicit method.

    Notes
    -----
    Spectral information is only used for inversion and square root; for
    action and exponentiation it is ignored.
    """
    problem = canonical_problem_name(problem)
    method = canonical_method_name(method)
    if problem in ENTIRE_PROBLEMS:
        return _default_eta(problem, method, None)
    spectral = _spectral_data(problem, matrix, sigma_min, sigma_max, gap, spectral_kind, hermitian_tol)
    _check_method(method, spectral)
    return _default_eta(problem, method, spectral)


# ---------------------------------------------------------------------------
# Public: default Sobolev scale r
# ---------------------------------------------------------------------------


def default_sobolev_r(
    problem: str,
    *,
    matrix: np.ndarray | Sequence[Sequence[complex]] | None = None,
    sigma_min: float | None = None,
    sigma_max: float | None = None,
    gap: tuple[float, float] | None = None,
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
    - exponentiation: ``r_exp = 1 / (1 + 1/tau)``, since ``g' = g / tau``,
      which is ``1 / (1 + eta delta / pi)``;
    - inversion: ``min(1, d)``, with ``d`` the distance from the fitted set
      to the pole in the scaled variable.  At the default *eta* this is the
      paper's ``r_def = min(1, 2 pi / (kappa + 1))`` for a definite interval
      and ``r_indef = min(1, pi / (kappa + 1))`` for singular values;
    - square root: ``r_sqrt = kappa^(-(2w - 1) / (2w))``, which balances the
      two contributions to the extension's H_r^w norm (``kappa^(-1/2)`` for
      ``w = 1``).

    Parameters
    ----------
    problem:
        One of ``'action'``, ``'inversion'``, ``'exponentiation'``, or ``'square_root'`` (or
        an alias; see :func:`~felcu.canonical_problem_name`).
    matrix:
        Square matrix to read the spectral interval from.  Mutually
        exclusive with *sigma_min* / *sigma_max* / *gap*.
    sigma_min, sigma_max:
        Spectral interval bounds.
    gap:
        Eigenvalue gap ``(a, b)`` around 0, for inversion on an indefinite
        interval; see :func:`~felcu.solve_fe_coefficients`.
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
        Except for action, if no spectral information is given or the
        interval is invalid.
    """
    problem = canonical_problem_name(problem)
    if problem == "action":
        return 1.0
    spectral = _spectral_data(problem, matrix, sigma_min, sigma_max, gap, spectral_kind, hermitian_tol)
    tau = _make_spec(problem, spectral, eta)["tau"]
    if problem == "exponentiation":
        return 1.0 / (1.0 + 1.0 / tau)

    if problem == "square_root":
        kappa = spectral["sigma_max"] / spectral["sigma_min"]
        return kappa ** (-(2.0 * sobolev_w - 1.0) / (2.0 * sobolev_w))
    # Inversion: the distance from the fitted set to the pole at 0.
    lo, hi, gap = spectral["fit_lo"], spectral["fit_hi"], spectral["fit_gap"]
    nearest = min(abs(lo), abs(hi)) if gap is None else min(-gap[0], gap[1])
    return min(1.0, tau * nearest)


# ---------------------------------------------------------------------------
# Internal: build the problem spec dict used throughout the solver
# ---------------------------------------------------------------------------


def _make_spec(problem: str, spectral: dict, eta: float | None, method: str = "sobolev") -> dict:
    """The fit of *problem* on *spectral*: ``g(x) = f(mu + x / tau)`` on ``tau (spectrum - mu)``.

    Every fit is centred: ``mu`` and ``delta`` are the centre and half-width
    of the fitted set and ``tau = pi / (eta delta)``, so the fitted set
    spans ``[-pi/eta, pi/eta]``, less any gap around a pole.
    """
    _check_method(method, spectral)
    if eta is None:
        eta = _default_eta(problem, method, spectral)
    eta = float(eta)
    if eta <= 1.0:
        raise ValueError("eta must exceed 1.")

    mu, delta = _centre_and_halfwidth(spectral)
    tau = math.inf if delta == 0 else PI / (eta * delta)
    x_max = PI / eta
    gap = spectral["fit_gap"]
    x_gap = None
    if gap is not None:
        # A gap that reaches an end of the spectrum leaves that side a single
        # point, placed exactly at the end so that it is not a sliver.
        x_gap = (
            -x_max if gap[0] == spectral["fit_lo"] else tau * (gap[0] - mu),
            x_max if gap[1] == spectral["fit_hi"] else tau * (gap[1] - mu),
        )

    # The dilation of a general matrix needs an odd series.  Otherwise the
    # constant carries f(mu), which the fit subtracts as a baseline when mu
    # is in the fitted set.
    if spectral["kind"] == "singular_values":
        basis, baseline = "odd_sine", 0.0
    else:
        basis = "const_plus_sine" if problem == "action" else "full"
        baseline = 0.0 if gap is not None else _value(problem, mu)

    spec = {
        "problem": problem,
        "method": method,
        "eta": eta,
        "x_max": x_max,
        "x_gap": x_gap,
        "spectral": spectral,
        "tau": tau,
        "mu": mu,
        "basis": basis,
        "baseline": baseline,
    }
    if method != "sobolev":
        _check_explicit_eta(spec)
    return spec
