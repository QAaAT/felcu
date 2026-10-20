"""Type definitions for felcu."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np


ArrayLike = np.ndarray | Sequence[float] | float


@dataclass(frozen=True)
class FourierExtensionResult:
    """Result of a Fourier-extension coefficient computation.

    Returned by :func:`~felcu.solve_fe_coefficients`.  The fitted series is
    ``g(x) = a0 + sum_k ak[k-1] sin(kx) + bk[k-1] cos(kx)``, and the matrix
    function is ``f(H) = g(tau H)`` (unshifted) or
    ``f(H) = g(tau (H - mu I))`` (shifted, ``mu`` the centre of
    ``[sigma_min, sigma_max]``).

    Attributes
    ----------
    problem : str
        ``'action'``, ``'inversion'``, or ``'exponentiation'``.
    shifted : bool
        Whether the shifted (centred) construction was used.
    basis_used : str
        ``'odd_sine'`` (sine terms only), ``'const_plus_sine'`` (constant
        plus sine terms), or ``'full'``.
    m : int
        Truncation order: the number of harmonics.
    eta : float
        Extension parameter; the series was fitted on
        ``[-pi/eta, pi/eta]``.
    tau : float
        Scale between the matrix spectrum and the fitted variable.  ``1``
        without matrix-scale information; infinite for a zero-width
        interval.
    alpha : float
        LCU normalisation, ``|a0| + sum_k sqrt(ak**2 + bk**2)``.
    target_error : float
        The requested error.
    error_metric : str
        ``'l2'`` or ``'linf'``: the metric compared with *target_error*.
    achieved_error : float
        Achieved error in *error_metric*.
    achieved_l2_error : float
        Achieved L2 error on the fitted domain.
    achieved_linf_error : float
        Achieved maximum pointwise error on the fitted domain.
    converged : bool
        Whether *achieved_error* meets *target_error*.
    lambda_used : float
        Sobolev regularisation strength chosen for this ``m``.
    sobolev_r : float
        Scale *r* of the H_r^w regulariser that was used.
    sobolev_w : float
        Order *w* of the H_r^w regulariser that was used.
    a0 : float
        Constant term.
    ak : numpy.ndarray
        Sine coefficients, length ``m``.
    bk : numpy.ndarray
        Cosine coefficients, length ``m`` (all zero for the sine-only
        bases).
    complex_coefficients : numpy.ndarray
        The same series as ``[c_{-m}, ..., c_m]``; see
        :func:`~felcu.to_complex_coefficients`.
    k : numpy.ndarray
        Harmonic indices ``1, ..., m`` matching *ak* and *bk*.
    domain : str
        ``'single'`` for ``[-x_max, x_max]``, or ``'double'`` for
        ``[-x_max, -x_min] U [x_min, x_max]`` (unshifted inversion).
    x_max : float
        Outer edge of the fitted domain, ``pi / eta``.
    x_min : float or None
        Inner edge of a double domain; ``None`` otherwise.
    sigma_min, sigma_max : float or None
        Spectral interval used (eigenvalues or singular values, per
        *spectral_kind_used*); ``None`` without matrix-scale information.
    spectral_kind_used : str or None
        ``'eigenvalues'`` or ``'singular_values'``; ``None`` without
        matrix-scale information.
    matrix_shape : tuple of int or None
        Shape of the input matrix, if one was given.
    matrix_is_hermitian : bool or None
        Whether the input matrix passed the Hermitian test, if one was
        given.
    matrix_scale_provided : bool
        Whether a matrix or spectral bounds were given.
    all_results : tuple of FourierExtensionResult
        Every candidate tried, sorted by ``m``.  Empty on the candidates
        themselves.
    """

    problem: str
    shifted: bool
    basis_used: str
    m: int
    eta: float
    tau: float
    alpha: float
    target_error: float
    error_metric: str
    achieved_error: float
    achieved_l2_error: float
    achieved_linf_error: float
    converged: bool
    lambda_used: float
    sobolev_r: float
    sobolev_w: float
    a0: float
    ak: np.ndarray
    bk: np.ndarray
    complex_coefficients: np.ndarray
    k: np.ndarray
    domain: str
    x_max: float
    x_min: Optional[float]
    sigma_min: Optional[float]
    sigma_max: Optional[float]
    spectral_kind_used: Optional[str]
    matrix_shape: Optional[tuple[int, int]]
    matrix_is_hermitian: Optional[bool]
    matrix_scale_provided: bool
    all_results: tuple["FourierExtensionResult", ...] = ()

    @property
    def coefficients(self) -> np.ndarray:
        """Coefficients packed as a single vector in the order of *basis_used*.

        ``[ak]`` for ``'odd_sine'``, ``[a0, ak]`` for ``'const_plus_sine'``,
        and ``[a0, ak, bk]`` for ``'full'``.
        """
        from felcu._coefficients import _pack_coefficients
        return _pack_coefficients(self.a0, self.ak, self.bk, self.basis_used)

    @property
    def achieved_inf_error(self) -> float:
        """Alias for :attr:`achieved_linf_error`."""
        return self.achieved_linf_error
