"""Tests for felcu.utils and felcu.types."""

import numpy as np
import pytest

from felcu.utils import (
    canonical_problem_name,
    canonical_error_metric,
    default_m_grid,
    default_lambda_grid,
    evaluate_fourier_series,
    to_complex_coefficients,
)


# ---------------------------------------------------------------------------
# canonical_problem_name
# ---------------------------------------------------------------------------

class TestCanonicalProblemName:
    def test_aliases_resolve(self):
        assert canonical_problem_name("action") == "action"
        assert canonical_problem_name("identity") == "action"
        assert canonical_problem_name("matrix-action") == "action"
        assert canonical_problem_name("inversion") == "inversion"
        assert canonical_problem_name("inverse") == "inversion"
        assert canonical_problem_name("exp") == "exponentiation"
        assert canonical_problem_name("exponentiation") == "exponentiation"
        assert canonical_problem_name("exponential") == "exponentiation"

    def test_case_and_whitespace_insensitive(self):
        assert canonical_problem_name("  ACTION  ") == "action"
        assert canonical_problem_name("Inversion") == "inversion"

    def test_unknown_raises(self):
        with pytest.raises(ValueError, match="problem must be"):
            canonical_problem_name("eigenvalue")


# ---------------------------------------------------------------------------
# canonical_error_metric
# ---------------------------------------------------------------------------

class TestCanonicalErrorMetric:
    def test_aliases_resolve(self):
        assert canonical_error_metric("l2") == "l2"
        assert canonical_error_metric("2") == "l2"
        assert canonical_error_metric("linf") == "linf"
        assert canonical_error_metric("max") == "linf"
        assert canonical_error_metric("infinity") == "linf"

    def test_unknown_raises(self):
        with pytest.raises(ValueError, match="error_metric must be"):
            canonical_error_metric("l1")


# ---------------------------------------------------------------------------
# default_m_grid
# ---------------------------------------------------------------------------

class TestDefaultMGrid:
    def test_returns_integers(self):
        grid = default_m_grid()
        assert grid.dtype == int

    def test_bounds_respected(self):
        grid = default_m_grid(m_min=5, m_max=50)
        assert grid[0] >= 5
        assert grid[-1] <= 50

    def test_single_count_returns_max(self):
        grid = default_m_grid(m_min=4, m_max=100, count=1)
        assert list(grid) == [100]

    def test_invalid_args_raise(self):
        with pytest.raises(ValueError):
            default_m_grid(m_min=-1)
        with pytest.raises(ValueError):
            default_m_grid(m_min=50, m_max=10)


# ---------------------------------------------------------------------------
# default_lambda_grid
# ---------------------------------------------------------------------------

class TestDefaultLambdaGrid:
    def test_decreasing(self):
        grid = default_lambda_grid()
        assert np.all(np.diff(grid) < 0)

    def test_bounds(self):
        grid = default_lambda_grid(lambda_max=1.0, lambda_min=1e-6)
        assert pytest.approx(grid[0]) == 1.0
        assert pytest.approx(grid[-1]) == 1e-6

    def test_invalid_raises(self):
        with pytest.raises(ValueError):
            default_lambda_grid(lambda_min=0.0)
        with pytest.raises(ValueError):
            default_lambda_grid(count=1)


# ---------------------------------------------------------------------------
# evaluate_fourier_series
# ---------------------------------------------------------------------------

class TestEvaluateFourierSeries:
    def test_constant(self):
        x = np.linspace(-np.pi, np.pi, 100)
        y = evaluate_fourier_series(x, a0=3.0, ak=[], bk=[])
        np.testing.assert_allclose(y, 3.0)

    def test_sine_term(self):
        x = np.array([0.0, np.pi / 2, np.pi])
        # g(x) = sin(x)  →  a0=0, ak=[1], bk=[0]
        y = evaluate_fourier_series(x, a0=0.0, ak=[1.0], bk=[0.0])
        np.testing.assert_allclose(y, np.sin(x), atol=1e-14)

    def test_cosine_term(self):
        x = np.array([0.0, np.pi / 2, np.pi])
        # g(x) = cos(x)  →  a0=0, ak=[0], bk=[1]
        y = evaluate_fourier_series(x, a0=0.0, ak=[0.0], bk=[1.0])
        np.testing.assert_allclose(y, np.cos(x), atol=1e-14)

    def test_shape_preserved(self):
        x = np.zeros((3, 4))
        y = evaluate_fourier_series(x, 1.0, [0.5], [0.5])
        assert y.shape == (3, 4)

    def test_mismatched_ak_bk_raises(self):
        with pytest.raises(ValueError):
            evaluate_fourier_series(np.array([0.0]), 0.0, [1.0, 2.0], [1.0])


# ---------------------------------------------------------------------------
# to_complex_coefficients
# ---------------------------------------------------------------------------

class TestToComplexCoefficients:
    def test_constant_only(self):
        c = to_complex_coefficients(2.0, ak=[], bk=[])
        assert c.shape == (1,)
        assert c[0] == pytest.approx(2.0 + 0j)

    def test_symmetry(self):
        # c_{-k} = conj(c_k) for a real series
        ak = np.array([1.0, 0.5])
        bk = np.array([0.3, 0.7])
        c = to_complex_coefficients(0.0, ak=ak, bk=bk)
        m = len(ak)
        # c has length 2m+1; c[m] = c0, c[m+k] = c_k, c[m-k] = c_{-k}
        for k in range(1, m + 1):
            np.testing.assert_allclose(c[m - k], np.conj(c[m + k]), atol=1e-15)

    def test_missing_bk_raises(self):
        with pytest.raises(ValueError):
            to_complex_coefficients(0.0, ak=[1.0])
