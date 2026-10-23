r"""felcu — Fourier-extension coefficients for LCU block encodings.

The coefficients are returned in the real Fourier basis

    g(x) = a0 + sum_{k=1}^m ak[k-1] sin(k x) + bk[k-1] cos(k x).

A matrix or spectral bounds are required.  The scalar function is fitted in a
scaled variable x.  An eigenvalue interval is fitted shifted (centred at its
midpoint mu); singular values cannot be shifted, so they are fitted unshifted
by an odd series:

    eigenvalues, shifted:        g(x) = f(x/tau + mu),   f(H) = g(tau (H - mu I))
    singular values, unshifted:  g(x) = f(x/tau),        f(A) = g(tau A)

Singular values are used for inversion and for action on a non-Hermitian
matrix; everything else uses eigenvalues.

Typical usage
-------------
>>> from felcu import solve_fe_coefficients
>>> result = solve_fe_coefficients("exponentiation", sigma_min=0.1, sigma_max=2.0)
>>> result.a0, result.ak, result.bk
"""

from felcu.solve import compute_fe_coefficients, solve_fe_coefficients
from felcu.spectral import default_eta, default_sobolev_r
from felcu.types import FourierExtensionResult
from felcu.utils import (
    ArrayLike,
    canonical_error_metric,
    canonical_problem_name,
    default_lambda_grid,
    default_m_grid,
    evaluate_fourier_series,
    evaluate_target_function,
    fitted_domain_segments,
    to_complex_coefficients,
)

__all__ = [
    # Types
    "FourierExtensionResult",
    "ArrayLike",
    # Main solver
    "solve_fe_coefficients",
    "compute_fe_coefficients",
    # Utilities
    "canonical_problem_name",
    "canonical_error_metric",
    "default_eta",
    "default_sobolev_r",
    "default_m_grid",
    "default_lambda_grid",
    "evaluate_fourier_series",
    "evaluate_target_function",
    "fitted_domain_segments",
    "to_complex_coefficients",
]
