"""Type definitions for felcu."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

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
        ``'action'``, ``'inversion'``, ``'exponentiation'``, or ``'square_root'``.
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
        Subnormalisation ``alpha_m = sum_k |c_k| = |a0| + sum_k sqrt(ak**2 + bk**2)``.
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
    x_min: float | None
    sigma_min: float | None
    sigma_max: float | None
    spectral_kind_used: str | None
    matrix_shape: tuple[int, int] | None
    matrix_is_hermitian: bool | None
    matrix_scale_provided: bool
    all_results: tuple[FourierExtensionResult, ...] = ()

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
