"""Public utility functions for felcu."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from felcu._coefficients import _target
from felcu.types import ArrayLike, FourierExtensionResult


def canonical_problem_name(name: str) -> str:
    """Normalise a problem name to one of 'action', 'inversion', 'exponentiation', 'square_root'.

    Matching ignores case and surrounding whitespace.

    Parameters
    ----------
    name:
        Problem name.  Accepted aliases are ``'identity'`` and
        ``'matrix-action'`` for ``'action'``; ``'inverse'`` for
        ``'inversion'``; ``'exp'`` and ``'exponential'`` for
        ``'exponentiation'``; and ``'sqrt'`` and ``'square-root'`` for
        ``'square_root'``.

    Returns
    -------
    str
        ``'action'``, ``'inversion'``, ``'exponentiation'``, or
        ``'square_root'``.

    Raises
    ------
    ValueError
        If *name* is not a recognised problem or alias.
    """
    aliases = {
        "action": "action",
        "identity": "action",
        "matrix-action": "action",
        "inversion": "inversion",
        "inverse": "inversion",
        "exp": "exponentiation",
        "exponentiation": "exponentiation",
        "exponential": "exponentiation",
        "square_root": "square_root",
        "square-root": "square_root",
        "sqrt": "square_root",
    }
    key = name.strip().lower()
    if key not in aliases:
        raise ValueError("problem must be 'action', 'inversion', 'exponentiation', or 'square_root'.")
    return aliases[key]


def canonical_method_name(name: str) -> str:
    """Normalise a coefficient-selection method to 'sobolev', 'reflected' or 'arcsine_taylor'.

    Matching ignores case and surrounding whitespace.

    Parameters
    ----------
    name:
        Method name.  Accepted aliases are ``'regularised'``,
        ``'regularized'`` and ``'reg'`` for ``'sobolev'``; ``'refl'``,
        ``'reflected_extension'`` and ``'reflected-extension'`` for
        ``'reflected'``; and ``'asin'``,
        ``'arcsine'`` and ``'arcsine-taylor'`` for ``'arcsine_taylor'``.

    Returns
    -------
    str
        ``'sobolev'``, ``'reflected'``, or ``'arcsine_taylor'``.

    Raises
    ------
    ValueError
        If *name* is not a recognised method or alias.
    """
    aliases = {
        "sobolev": "sobolev",
        "regularised": "sobolev",
        "regularized": "sobolev",
        "reg": "sobolev",
        "reflected": "reflected",
        "refl": "reflected",
        "reflected_extension": "reflected",
        "reflected-extension": "reflected",
        "arcsine_taylor": "arcsine_taylor",
        "arcsine-taylor": "arcsine_taylor",
        "arcsine": "arcsine_taylor",
        "asin": "arcsine_taylor",
    }
    key = name.strip().lower()
    if key not in aliases:
        raise ValueError("method must be 'sobolev', 'reflected', or 'arcsine_taylor'.")
    return aliases[key]


def canonical_error_metric(name: str) -> str:
    """Normalise an error-metric name to 'l2' or 'linf'.

    Matching ignores case and surrounding whitespace.

    Parameters
    ----------
    name:
        Metric name.  ``'2'`` is accepted for ``'l2'``; ``'inf'``,
        ``'max'``, ``'infinity'``, ``'l_inf'`` and ``'l-infinity'`` for
        ``'linf'``.

    Returns
    -------
    str
        ``'l2'`` or ``'linf'``.

    Raises
    ------
    ValueError
        If *name* is not a recognised metric or alias.
    """
    aliases = {
        "l2": "l2",
        "2": "l2",
        "linf": "linf",
        "l_inf": "linf",
        "l-infinity": "linf",
        "infinity": "linf",
        "inf": "linf",
        "max": "linf",
    }
    key = name.strip().lower()
    if key not in aliases:
        raise ValueError("error_metric must be 'l2' or 'linf'.")
    return aliases[key]


def default_m_grid(
    m_min: int = 4,
    m_max: int = 256,
    count: int = 32,
    power: float = 2.0,
) -> np.ndarray:
    """Return a grid of candidate Fourier truncation orders.

    The grid is ``m_min + (m_max - m_min) * t**power`` for *count* equally
    spaced ``t`` in ``[0, 1]``, truncated to integers.  With ``power > 1``
    the grid is denser at small ``m``.  This is the default ``m_values``
    for :func:`~felcu.solve_fe_coefficients`.

    Parameters
    ----------
    m_min, m_max:
        Smallest and largest order.
    count:
        Number of grid points before duplicates are removed.
    power:
        Spacing exponent; ``1`` gives a uniform grid.

    Returns
    -------
    numpy.ndarray
        Sorted unique integer orders.  Truncation can merge neighbouring
        points, so there may be fewer than *count*.  If ``count == 1`` the
        grid is ``[m_max]``.

    Raises
    ------
    ValueError
        Unless ``0 <= m_min <= m_max`` and ``count > 0``.
    """
    if m_min < 0 or m_max < m_min or count <= 0:
        raise ValueError("Require 0 <= m_min <= m_max and count > 0.")
    if count == 1:
        return np.array([m_max], dtype=int)
    grid = m_min + (m_max - m_min) * np.linspace(0.0, 1.0, count) ** power
    return np.unique(grid.astype(int))


def default_lambda_grid(
    lambda_max: float = 1.0,
    lambda_min: float = 1e-16,
    count: int = 12,
) -> np.ndarray:
    """Return a log-spaced grid of regularisation parameters.

    This is the default ``lambdas`` for
    :func:`~felcu.solve_fe_coefficients`.  The solver sorts the grid
    itself and searches beyond it, so the grid only needs to bracket the
    answer roughly.

    Parameters
    ----------
    lambda_max, lambda_min:
        Largest and smallest value.
    count:
        Number of values.

    Returns
    -------
    numpy.ndarray
        *count* geometrically spaced values from *lambda_max* down to
        *lambda_min*.

    Raises
    ------
    ValueError
        Unless ``0 < lambda_min <= lambda_max`` and ``count >= 2``.
    """
    if lambda_min <= 0 or lambda_max <= 0 or lambda_min > lambda_max or count < 2:
        raise ValueError("Require 0 < lambda_min <= lambda_max and count >= 2.")
    return np.geomspace(lambda_max, lambda_min, count)


def evaluate_fourier_series(
    x: ArrayLike,
    a0: float | complex,
    ak: Sequence,
    bk: Sequence,
) -> np.ndarray:
    """Evaluate the real Fourier series g(x) = a0 + sum_k ak sin(kx) + bk cos(kx).

    The sum runs over ``k = 1, ..., len(ak)``.

    Parameters
    ----------
    x:
        Evaluation points, of any shape.
    a0:
        Constant term.
    ak, bk:
        Sine and cosine coefficients, of equal length.  Complex values are
        allowed.

    Returns
    -------
    numpy.ndarray
        ``g(x)``, with the same shape as *x*.  The result is complex if any
        coefficient is complex.

    Raises
    ------
    ValueError
        If *ak* and *bk* have different lengths.
    """
    x = np.asarray(x)
    x_flat = x.reshape(-1)
    ak = np.asarray(ak).reshape(-1)
    bk = np.asarray(bk).reshape(-1)
    if ak.shape != bk.shape:
        raise ValueError("ak and bk must have the same length.")

    dtype = np.result_type(x, a0, ak, bk, float)
    y = np.full(x_flat.shape, a0, dtype=dtype)
    if ak.size:
        k = np.arange(1, ak.size + 1, dtype=float)
        angles = np.outer(x_flat.astype(float), k)
        y = y + np.sin(angles) @ ak + np.cos(angles) @ bk
    return y.reshape(x.shape)


def to_complex_coefficients(
    a0_or_result: FourierExtensionResult | float | complex,
    ak: Sequence | None = None,
    bk: Sequence | None = None,
) -> np.ndarray:
    r"""Convert a0, ak, bk to [c_{-m}, ..., c_0, ..., c_m].

    The complex coefficients give the same series in exponential form,
    ``g(x) = sum_{k=-m}^{m} c_k exp(i k x)``, with

    - ``c_0 = a0``,
    - ``c_k = (bk[k-1] - i ak[k-1]) / 2`` for ``k > 0``,
    - ``c_{-k} = (bk[k-1] + i ak[k-1]) / 2``.

    Parameters
    ----------
    a0_or_result:
        Either a :class:`~felcu.FourierExtensionResult`, in which case its
        coefficients are used and *ak* / *bk* are ignored, or the constant
        term ``a0``.
    ak, bk:
        Sine and cosine coefficients, of equal length.  Required unless a
        result is passed.

    Returns
    -------
    numpy.ndarray
        Complex array of length ``2m + 1``, ordered from ``c_{-m}`` to
        ``c_m``.

    Raises
    ------
    ValueError
        If *ak* or *bk* is missing, or their lengths differ.
    """
    if isinstance(a0_or_result, FourierExtensionResult):
        a0 = a0_or_result.a0
        ak = a0_or_result.ak
        bk = a0_or_result.bk
    elif ak is None or bk is None:
        raise ValueError("Provide both ak and bk, or pass a FourierExtensionResult.")
    else:
        a0 = a0_or_result

    ak = np.asarray(ak, dtype=np.complex128).reshape(-1)
    bk = np.asarray(bk, dtype=np.complex128).reshape(-1)
    if ak.shape != bk.shape:
        raise ValueError("ak and bk must have the same length.")

    c0 = np.array([complex(a0)], dtype=np.complex128)
    if ak.size == 0:
        return c0
    c_pos = 0.5 * (bk - 1j * ak)
    c_neg = 0.5 * (bk[::-1] + 1j * ak[::-1])
    return np.concatenate([c_neg, c0, c_pos])


def fitted_domain_segments(
    result: FourierExtensionResult,
    num: int = 2001,
) -> list[np.ndarray]:
    """Return x-arrays covering the fitted domain of *result*.

    Use these to plot or check a fit, e.g. by comparing
    :func:`evaluate_fourier_series` with :func:`evaluate_target_function`
    on each segment.

    Parameters
    ----------
    result:
        A result from :func:`~felcu.solve_fe_coefficients`.
    num:
        Number of points.  For a double-interval domain, each interval
        gets ``num // 2`` points.

    Returns
    -------
    list of numpy.ndarray
        One array spanning ``[-x_max, x_max]`` for a single-interval
        domain, or two arrays spanning ``[-x_max, x_gap[0]]`` and
        ``[x_gap[1], x_max]`` for a double-interval one.

    Raises
    ------
    ValueError
        If ``num < 2``.
    """
    if num < 2:
        raise ValueError("num must be at least 2.")
    if result.x_gap is None:
        return [np.linspace(-result.x_max, result.x_max, num)]
    n = max(num // 2, 2)
    return [
        np.linspace(-result.x_max, result.x_gap[0], n),
        np.linspace(result.x_gap[1], result.x_max, n),
    ]


def evaluate_target_function(
    x: ArrayLike,
    result: FourierExtensionResult,
) -> np.ndarray:
    """Evaluate the exact target function for *result* at points *x*.

    This is the function the series was fitted to, in the scaled variable,
    so it can be compared directly with :func:`evaluate_fourier_series`.
    It is ``g(x) = f(mu + x / tau)``, not ``f`` itself: ``mu + x / tau``,
    ``1 / (mu + x / tau)``, ``exp(mu + x / tau)`` or ``sqrt(mu + x / tau)``.
    If ``tau`` is infinite (a zero-width interval), the target is constant.

    Parameters
    ----------
    x:
        Evaluation points in the scaled variable.
    result:
        A result from :func:`~felcu.solve_fe_coefficients`.

    Returns
    -------
    numpy.ndarray
        Target values, with the same shape as *x*.

    Raises
    ------
    ValueError
        If the problem is unknown.
    """
    return _target(x, result.problem, result.tau, result.mu)
