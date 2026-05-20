"""Tests for felcu.solve (solve_fe_coefficients and compute_fe_coefficients)."""

import numpy as np
import pytest

from felcu import solve_fe_coefficients, compute_fe_coefficients, evaluate_fourier_series
from felcu.utils import fitted_domain_segments, evaluate_target_function


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _max_error(result, n=500):
    """L∞ error between the fitted series and the exact target over the domain."""
    errors = []
    for xs in fitted_domain_segments(result, num=n):
        y_fit = evaluate_fourier_series(xs, result.a0, result.ak, result.bk)
        y_true = evaluate_target_function(xs, result)
        errors.append(np.max(np.abs(y_fit - y_true)))
    return max(errors)


# ---------------------------------------------------------------------------
# Basic convergence tests (small m to stay fast)
# ---------------------------------------------------------------------------

class TestConvergence:
    def test_exponentiation_converges(self):
        result = solve_fe_coefficients(
            "exponentiation",
            sigma_min=0.1, sigma_max=2.0,
            target_error=1e-3, m=16,
        )
        assert result.converged
        assert result.achieved_l2_error <= 1e-3 * 1.01

    def test_action_converges(self):
        result = solve_fe_coefficients(
            "action",
            sigma_min=-1.0, sigma_max=1.0,
            target_error=1e-3, m=8,
        )
        assert result.converged

    def test_inversion_converges(self):
        result = solve_fe_coefficients(
            "inversion",
            sigma_min=0.5, sigma_max=2.0,
            target_error=1e-3, m=16,
        )
        assert result.converged

    def test_linf_metric(self):
        result = solve_fe_coefficients(
            "exponentiation",
            sigma_min=0.5, sigma_max=1.5,
            target_error=1e-2,
            error_metric="linf",
            m=20,
        )
        assert result.error_metric == "linf"
        assert result.achieved_linf_error <= 1e-2 * 1.01


# ---------------------------------------------------------------------------
# Result fields
# ---------------------------------------------------------------------------

class TestResultFields:
    def setup_method(self):
        self.result = solve_fe_coefficients(
            "exponentiation",
            sigma_min=0.5, sigma_max=1.5,
            target_error=1e-2, m=12,
        )

    def test_ak_bk_same_length(self):
        assert len(self.result.ak) == len(self.result.bk) == self.result.m

    def test_complex_coefficients_length(self):
        assert len(self.result.complex_coefficients) == 2 * self.result.m + 1

    def test_k_matches_m(self):
        assert len(self.result.k) == self.result.m
        assert self.result.k[0] == 1
        assert self.result.k[-1] == self.result.m

    def test_all_results_sorted_by_m(self):
        ms = [r.m for r in self.result.all_results]
        assert ms == sorted(ms)

    def test_coefficients_property(self):
        c = self.result.coefficients
        # 'full' basis → length 2m+1
        assert c.ndim == 1

    def test_achieved_inf_error_alias(self):
        assert self.result.achieved_inf_error == self.result.achieved_linf_error


# ---------------------------------------------------------------------------
# compute_fe_coefficients convenience wrapper
# ---------------------------------------------------------------------------

class TestComputeFECoefficients:
    def test_returns_tuple(self):
        a0, ak, bk = compute_fe_coefficients(
            "exponentiation", sigma_min=0.5, sigma_max=1.5,
            target_error=1e-2, m=10,
        )
        assert isinstance(a0, float)
        assert isinstance(ak, np.ndarray)
        assert isinstance(bk, np.ndarray)
        assert len(ak) == len(bk)


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------

class TestInputValidation:
    def test_negative_target_error_raises(self):
        with pytest.raises(ValueError, match="target_error"):
            solve_fe_coefficients("exponentiation", sigma_min=0.5, sigma_max=1.5, target_error=-1.0)

    def test_matrix_and_sigma_raises(self):
        A = np.eye(2)
        with pytest.raises(ValueError):
            solve_fe_coefficients("exponentiation", matrix=A, sigma_min=0.5, sigma_max=1.5)

    def test_invalid_selection_raises(self):
        with pytest.raises(ValueError, match="selection"):
            solve_fe_coefficients(
                "exponentiation", sigma_min=0.5, sigma_max=1.5,
                m=4, selection="biggest_m",
            )

    def test_no_scale_inversion_raises(self):
        with pytest.raises(ValueError):
            solve_fe_coefficients("inversion")  # inversion always needs a scale

    def test_shifted_no_scale_raises(self):
        with pytest.raises(ValueError):
            solve_fe_coefficients("action", shifted=True)

    def test_matrix_input(self):
        A = np.diag([0.5, 1.0, 1.5])
        result = solve_fe_coefficients("exponentiation", matrix=A, target_error=1e-2, m=12)
        assert result.converged
        assert result.matrix_shape == (3, 3)
