r"""felcu — Fourier-extension coefficients for LCU block encodings.

The coefficients are returned in the real Fourier basis

    g(x) = a0 + sum_{k=1}^m ak[k-1] sin(k x) + bk[k-1] cos(k x).

When matrix-scale information is supplied, the scalar function is fitted in a
scaled variable x:

    unshifted: g(x) = f(x/tau),        f(H) = g(tau H)
    shifted:   g(x) = f(x/tau + mu),   f(H) = g(tau (H - mu I))

When no matrix scale is supplied, the fitted function is the unscaled scalar
function f(x), except for inversion where a nonzero interval is required.

The coefficients come from one of three strategies (``method=``):
Sobolev-regularised least squares (the default), reflected extensions, or
arcsine-Taylor expansions.

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
    canonical_method_name,
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
    "canonical_method_name",
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
