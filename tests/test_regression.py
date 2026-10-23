"""Regression tests: compare solver output with reference results saved as JSON.

The cases are defined in ``regression_cases.py``, which also regenerates the
references (``python tests/regression_cases.py``).
"""

import dataclasses

import numpy as np
import pytest
from regression_cases import CASES, DATA_DIR, reference_path, solve

from felcu import FourierExtensionResult

# Different BLAS/LAPACK builds (CI runs on Linux, macOS and Windows) change the
# last few digits, so floats are compared with a tolerance; everything else
# must match exactly.
RTOL = 1e-6
COEFFICIENT_ATOL = 1e-9  # relative to alpha
_COEFFICIENTS = {"a0", "ak", "bk", "complex_coefficients"}


@pytest.mark.parametrize("name", CASES)
def test_matches_reference(name):
    expected = FourierExtensionResult.from_json(reference_path(name))
    actual = solve(name)
    for field in dataclasses.fields(FourierExtensionResult):
        if field.name == "all_results":
            continue
        x, y = getattr(actual, field.name), getattr(expected, field.name)
        if field.name in _COEFFICIENTS:
            np.testing.assert_allclose(x, y, rtol=0, atol=COEFFICIENT_ATOL * expected.alpha, err_msg=field.name)
        elif isinstance(y, float):
            np.testing.assert_allclose(x, y, rtol=RTOL, err_msg=field.name)
        elif isinstance(y, np.ndarray):
            np.testing.assert_array_equal(x, y, err_msg=field.name)
        else:
            assert x == y, field.name


def test_every_case_has_a_reference():
    assert sorted(p.stem for p in DATA_DIR.glob("*.json")) == sorted(CASES)
