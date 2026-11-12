"""Type definitions for felcu."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

ArrayLike = np.ndarray | Sequence[float] | float


class ConvergenceWarning(UserWarning):
    """No candidate order met the requested *target_error*.

    Issued by :func:`~felcu.solve_fe_coefficients`, which still returns the
    best candidate, with ``converged=False``.  Silence it with
    ``warnings.simplefilter("ignore", felcu.ConvergenceWarning)``.
    """


@dataclass(frozen=True)
class FourierExtensionResult:
    """Result of a Fourier-extension coefficient computation.

    Returned by :func:`~felcu.solve_fe_coefficients`.  The fitted series is
    ``g(x) = a0 + sum_k ak[k-1] sin(kx) + bk[k-1] cos(kx)``, and the matrix
    function is ``f(H) = g(tau (H - mu I))``.  For singular values, ``H``
    is the Hermitian dilation of the matrix and ``mu = 0``.

    Attributes
    ----------
    problem : str
        ``'identity'``, ``'inversion'``, ``'exponentiation'``, ``'square_root'``, or
        ``'logarithm'``.
    method : str
        Coefficient-selection strategy: ``'sobolev'``, ``'reflected'``, or
        ``'arcsine_taylor'``.
    basis_used : str
        ``'odd_sine'`` (sine terms only), ``'const_plus_sine'`` (constant
        plus sine terms), or ``'full'``.
    m : int
        Truncation order: the number of harmonics.
    eta : float
        Extension parameter; the series was fitted on
        ``[-pi/eta, pi/eta]``.
    tau : float
        Scale between the matrix spectrum and the fitted variable,
        ``pi / (eta delta)`` with ``delta`` the half-width of the fitted
        spectrum; infinite for a zero-width interval.
    mu : float
        Centre of the fitted spectrum, which the fit is shifted by: the
        midpoint of ``[sigma_min, sigma_max]`` for eigenvalues, and ``0`` for
        singular values.
    alpha : float
        Subnormalisation ``alpha_m = sum_k |c_k| = |a0| + sum_k sqrt(ak**2 + bk**2)``.
    target_error : float
        The requested error.
    error_metric : str
        ``'l2'`` or ``'linf'``: the metric compared with *target_error*.
    achieved_error : float
        Achieved error in *error_metric*.
    achieved_l2_error : float
        Achieved L2 error on the fitted domain.  A side of a two-interval
        domain that is a single point counts as much as the whole domain.
    achieved_linf_error : float
        Achieved maximum pointwise error on the fitted domain.
    converged : bool
        Whether *achieved_error* meets *target_error*.
    lambda_used : float or None
        Sobolev regularisation strength chosen for this ``m``; ``None``
        for the explicit (``'reflected'``, ``'arcsine_taylor'``) methods.
    sobolev_r : float or None
        Scale *r* of the H_r^w regulariser that was used; ``None`` for the
        explicit methods.
    sobolev_w : float or None
        Order *w* of the H_r^w regulariser that was used; ``None`` for the
        explicit methods.
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
    x_max : float
        Outer edge of the fitted domain, ``pi / eta``.
    x_gap : tuple of float or None
        For inversion on an indefinite interval or on singular values, the
        gap around the pole that the fit leaves free, in the fitted
        variable: the fitted set is ``[-x_max, x_gap[0]] U [x_gap[1], x_max]``.
        ``None`` when it is ``[-x_max, x_max]``.
    sigma_min, sigma_max : float
        Spectral interval used (eigenvalues or singular values, per
        *spectral_kind_used*).
    gap : tuple of float or None
        Eigenvalue gap ``(a, b)`` around 0 for inversion on an indefinite
        interval; ``None`` otherwise.
    spectral_kind_used : str
        ``'eigenvalues'`` or ``'singular_values'``.
    matrix_shape : tuple of int or None
        Shape of the input matrix, if one was given.
    matrix_is_hermitian : bool or None
        Whether the input matrix passed the Hermitian test, if one was
        given.
    all_results : tuple of FourierExtensionResult
        Every candidate tried, sorted by ``m``.  Empty on the candidates
        themselves.
    """

    problem: str
    method: str
    basis_used: str
    m: int
    eta: float
    tau: float
    mu: float
    alpha: float
    target_error: float
    error_metric: str
    achieved_error: float
    achieved_l2_error: float
    achieved_linf_error: float
    converged: bool
    lambda_used: float | None
    sobolev_r: float | None
    sobolev_w: float | None
    a0: float
    ak: np.ndarray
    bk: np.ndarray
    complex_coefficients: np.ndarray
    k: np.ndarray
    x_max: float
    x_gap: tuple[float, float] | None
    sigma_min: float
    sigma_max: float
    gap: tuple[float, float] | None
    spectral_kind_used: str
    matrix_shape: tuple[int, int] | None
    matrix_is_hermitian: bool | None
    all_results: tuple[FourierExtensionResult, ...] = ()

    @property
    def coefficients(self) -> np.ndarray:
        """Coefficients packed as a single vector in the order of *basis_used*.

        ``[ak]`` for ``'odd_sine'``, ``[a0, ak]`` for ``'const_plus_sine'``,
        and ``[a0, ak, bk]`` for ``'full'``.
        """
        from felcu._coefficients import _pack_coefficients

        return _pack_coefficients(self.a0, self.ak, self.bk, self.basis_used)

    def evaluate(self, x: ArrayLike) -> np.ndarray:
        """Evaluate the approximation to f at points *x* of the original variable.

        Applies the scaling for you: this is ``g(tau (x - mu))``, which
        approximates ``f(x)`` for *x* in the fitted spectrum.  For singular
        values ``mu = 0`` and *x* is a singular value (or its negative, an
        eigenvalue of the dilation).  To evaluate ``g`` itself in the scaled
        variable, use :func:`~felcu.evaluate_fourier_series`.

        Parameters
        ----------
        x:
            Points, of any shape, at which to approximate ``f``.

        Returns
        -------
        numpy.ndarray
            The approximation, with the same shape as *x*.
        """
        from felcu.utils import evaluate_fourier_series

        x = np.asarray(x, dtype=float)
        if not np.isfinite(self.tau):
            # A zero-width interval: the series is the constant a0.
            return np.full(x.shape, self.a0)
        return evaluate_fourier_series(self.tau * (x - self.mu), self.a0, self.ak, self.bk)

    def apply(
        self,
        matrix: np.ndarray | Sequence[Sequence[complex]],
        hermitian_tol: float = 1e-12,
        interval_tol: float = 1e-8,
    ) -> np.ndarray:
        """Apply the approximation to a Hermitian matrix: ``g(tau (H - mu I)) ~ f(H)``.

        Computed classically by eigendecomposition, to check a fit against
        an exact ``f(H)``.  The eigenvalues of *matrix* must lie in the
        fitted spectrum: outside it the series is a periodic extension that
        bears no relation to ``f``.  They must also lie in the domain of
        ``f``, which is checked separately, so the interval slack can never
        admit an eigenvalue where ``f`` is undefined: they must be positive
        for the square root and the logarithm, and nonzero for inversion,
        with the sign of the fitted spectrum when that is definite.  For a
        singular-value fit, pass the Hermitian dilation of the matrix.

        Both checks use the eigenvalues as computed in floating point, so a
        matrix is accepted exactly when its computed eigenvalues lie in the
        fitted spectrum and the domain of ``f``.  A matrix that is singular
        in exact arithmetic, but whose computed eigenvalues pass both checks,
        is nonsingular to working precision, and the result is
        ``g(tau (H - mu I))`` for those computed eigenvalues.

        Parameters
        ----------
        matrix:
            Square Hermitian matrix ``H``.
        hermitian_tol:
            Relative tolerance for the Hermitian test.
        interval_tol:
            How far, relative to the half-width of the fitted spectrum, an
            eigenvalue may stray outside it before it is rejected.  At the
            edges of an inversion gap it is relative to the edge's distance
            from 0 instead, and for a zero-width spectrum ``[mu, mu]`` to
            ``|mu|`` (absolute when ``mu = 0``).

        Returns
        -------
        numpy.ndarray
            The matrix ``g(tau (H - mu I))``, the same shape as *matrix*.

        Raises
        ------
        ValueError
            If *matrix* is not square, not Hermitian, or has an eigenvalue
            outside the domain of ``f`` or outside the fitted spectrum.
        """
        from felcu.spectral import _as_double, _is_hermitian

        H = _as_double(np.asarray(matrix))
        if H.ndim != 2 or H.shape[0] != H.shape[1]:
            raise ValueError("matrix must be square.")
        if not _is_hermitian(H, hermitian_tol):
            raise ValueError("matrix must be Hermitian; apply the result to its Hermitian dilation instead.")
        w, V = np.linalg.eigh(H)
        bad = self._outside_domain(w)
        if bad.any():
            raise ValueError(
                f"matrix has eigenvalues outside the domain of "
                f"{ {'inversion': '1/x', 'square_root': 'sqrt(x)', 'logarithm': 'log(x)'}[self.problem] }, "
                f"e.g. {w[bad][0]:.6g}; {self._domain_description()}."
            )
        outside = self._outside_fitted_spectrum(w, interval_tol)
        if outside.any():
            raise ValueError(
                f"matrix has eigenvalues outside the fitted spectrum, e.g. {w[outside][0]:.6g}; "
                f"refit for its spectrum (the fit covers [{self.sigma_min:.6g}, {self.sigma_max:.6g}]"
                + (f" less the gap ({self.gap[0]:.6g}, {self.gap[1]:.6g})" if self.gap is not None else "")
                + f" in {self.spectral_kind_used.replace('_', ' ')})."
            )
        return (V * self.evaluate(w)) @ V.conj().T

    def _outside_domain(self, w: np.ndarray) -> np.ndarray:
        """Mask of the eigenvalues *w* where f is undefined, or across its singularity from the fit.

        Independent of the interval slack, which can reach past 0 when the
        fitted spectrum ends close to it.
        """
        if self.problem in ("square_root", "logarithm"):
            return w <= 0
        if self.problem == "inversion":
            if self.spectral_kind_used == "eigenvalues" and self.sigma_min > 0:
                return w <= 0
            if self.spectral_kind_used == "eigenvalues" and self.sigma_max < 0:
                return w >= 0
            return w == 0
        return np.zeros(w.shape, dtype=bool)

    def _domain_description(self) -> str:
        """What :meth:`_outside_domain` requires, for its error message."""
        if self.problem in ("square_root", "logarithm"):
            return "the matrix must be positive definite"
        if self.spectral_kind_used == "eigenvalues" and self.sigma_min > 0:
            return "the fit is positive definite, so the matrix must be too"
        if self.spectral_kind_used == "eigenvalues" and self.sigma_max < 0:
            return "the fit is negative definite, so the matrix must be too"
        return "the matrix must be nonsingular"

    def _outside_fitted_spectrum(self, w: np.ndarray, tol: float) -> np.ndarray:
        """Mask of the eigenvalues *w* that lie outside the fitted set, with relative slack *tol*."""
        if not np.isfinite(self.tau):
            # A zero-width interval: only mu itself is fitted, with slack
            # relative to mu (an absolute floor would dwarf a tiny mu).
            return np.abs(w - self.mu) > tol * (abs(self.mu) if self.mu != 0 else 1.0)
        # In the scaled variable the fitted set is [-x_max, x_max] less x_gap.
        x = self.tau * (w - self.mu)
        slack = tol * self.x_max
        outside = np.abs(x) > self.x_max + slack
        if self.x_gap is not None:
            # The gap edges get slack relative to their distance from the pole
            # at x = -tau mu, not to the half-width: a gap narrower than the
            # half-width slack would otherwise be shut entirely.
            pole = -self.tau * self.mu
            lo, hi = self.x_gap
            outside |= (x > lo + tol * abs(lo - pole)) & (x < hi - tol * abs(hi - pole))
        return outside

    @property
    def achieved_inf_error(self) -> float:
        """Alias for :attr:`achieved_linf_error`."""
        return self.achieved_linf_error

    def to_dict(self, include_candidates: bool = False) -> dict:
        """Return the result as a JSON-compatible dictionary.

        Arrays become lists, ``complex_coefficients`` becomes
        ``{"k": [-m..m], "real": [...], "imag": [...]}``, and non-finite
        floats (``tau`` is infinite for a zero-width interval) become the
        strings ``"inf"``, ``"-inf"`` or ``"nan"``.  The dictionary also
        records ``format``, ``format_version`` and ``felcu_version``.

        Parameters
        ----------
        include_candidates:
            Also store :attr:`all_results`, every order tried in the sweep.
        """
        from felcu._io import to_dict

        return to_dict(self, include_candidates)

    @classmethod
    def from_dict(cls, data: dict) -> FourierExtensionResult:
        """Rebuild a result from :meth:`to_dict` output.

        Raises
        ------
        ValueError
            If *data* is not a saved felcu result, has an unsupported
            ``format_version``, or is missing a field.
        """
        from felcu._io import from_dict

        return from_dict(cls, data)

    def to_json(
        self, path: str | Path | None = None, *, indent: int | None = 2, include_candidates: bool = False
    ) -> str:
        """Serialise the result to JSON, and write it to *path* if given.

        The file is standard JSON (no ``NaN`` or ``Infinity`` literals) and
        can be read from any language; floats are written with enough
        digits to round-trip exactly.  See :meth:`to_dict` for the layout.

        Parameters
        ----------
        path:
            File to write.  If omitted, only the string is returned.
        indent:
            Indentation passed to :func:`json.dumps`; ``None`` for one line.
        include_candidates:
            Also store :attr:`all_results`.

        Returns
        -------
        str
            The JSON text.

        Examples
        --------
        >>> result.to_json("exp_coefficients.json")
        >>> loaded = FourierExtensionResult.from_json("exp_coefficients.json")
        """
        from felcu._io import to_json

        return to_json(self, path, indent=indent, include_candidates=include_candidates)

    @classmethod
    def from_json(cls, source: str | Path) -> FourierExtensionResult:
        """Load a result saved with :meth:`to_json`.

        Parameters
        ----------
        source:
            A path to a JSON file, or the JSON text itself.

        Raises
        ------
        ValueError
            As for :meth:`from_dict`.
        """
        from felcu._io import from_json

        return from_json(cls, source)
