"""Regression tests: compare solver output with reference results saved as JSON.

The cases are defined in ``regression_cases.py``, which also regenerates the
references (``python tests/regression_cases.py``).
"""

import dataclasses

import numpy as np
import pytest
from regression_cases import CASES, DATA_DIR, reference_path, solve

from felcu import FourierExtensionResult, evaluate_fourier_series, fitted_domain_segments

# Different BLAS/LAPACK builds and NumPy versions (CI covers several of each)
# change the rounding, so the comparison is in three tiers.
#
# Fields computed in closed form from the inputs must agree to rounding.
_CLOSED_FORM = {"eta", "tau", "x_max", "x_min", "sigma_min", "sigma_max", "sobolev_r", "sobolev_w", "target_error"}
_CLOSED_FORM_RTOL = 1e-12

# The fit itself is ill-conditioned: individual coefficients can move by more
# than the target error along near-null directions without changing the
# series, so the series is compared as a function on the fitted domain.
_SERIES_ATOL = 1e-3  # relative to target_error
_COEFFICIENTS = {"a0", "ak", "bk", "complex_coefficients"}

# lambda is found by bisection on a flat error curve, and the achieved errors
# follow it, so they move more than alpha does.
_SOLVER_RTOL = {
    "alpha": 1e-6,
    "lambda_used": 1e-4,
    "achieved_error": 1e-4,
    "achieved_l2_error": 1e-4,
    "achieved_linf_error": 1e-4,
}


def _series(result, x):
    return evaluate_fourier_series(x, result.a0, result.ak, result.bk)


@pytest.mark.parametrize("name", CASES)
def test_matches_reference(name):
    expected = FourierExtensionResult.from_json(reference_path(name))
    actual = solve(name)
    for field in dataclasses.fields(FourierExtensionResult):
        x, y = getattr(actual, field.name), getattr(expected, field.name)
        if field.name == "all_results" or field.name in _COEFFICIENTS:
            continue
        if field.name in _CLOSED_FORM and y is not None:
            np.testing.assert_allclose(x, y, rtol=_CLOSED_FORM_RTOL, err_msg=field.name)
        elif field.name in _SOLVER_RTOL:
            np.testing.assert_allclose(x, y, rtol=_SOLVER_RTOL[field.name], err_msg=field.name)
        elif isinstance(y, np.ndarray):
            np.testing.assert_array_equal(x, y, err_msg=field.name)
        else:
            assert x == y, field.name

    atol = _SERIES_ATOL * expected.target_error
    for x in fitted_domain_segments(expected, num=4001):
        np.testing.assert_allclose(_series(actual, x), _series(expected, x), rtol=0, atol=atol, err_msg="series")


def test_every_case_has_a_reference():
    assert sorted(p.stem for p in DATA_DIR.glob("*.json")) == sorted(CASES)
