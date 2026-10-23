"""Interface tests: the public contract of the solver.

Input validation, defaults and options, result fields, candidate selection
and exact constants.  Saving and loading are in ``test_io.py``.
"""

import numpy as np
import pytest
from helpers import max_error

from felcu import compute_fe_coefficients, default_sobolev_r, solve_fe_coefficients

pytestmark = pytest.mark.interface


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
                "exponentiation",
                sigma_min=0.5,
                sigma_max=1.5,
                m=4,
                selection="biggest_m",
            )

    @pytest.mark.parametrize("problem", ["action", "inversion", "exponentiation", "square_root"])
    def test_requires_spectral_information(self, problem):
        with pytest.raises(ValueError, match="Provide a matrix"):
            solve_fe_coefficients(problem)

    def test_negative_sobolev_w_raises(self):
        with pytest.raises(ValueError, match="sobolev_w"):
            solve_fe_coefficients("exponentiation", sigma_min=0.5, sigma_max=1.5, m=4, sobolev_w=-1.0)

    def test_negative_sobolev_r_raises(self):
        with pytest.raises(ValueError, match="sobolev_r"):
            solve_fe_coefficients("exponentiation", sigma_min=0.5, sigma_max=1.5, m=4, sobolev_r=-1.0)

    def test_matrix_input(self):
        A = np.diag([0.5, 1.0, 1.5])
        result = solve_fe_coefficients("exponentiation", matrix=A, target_error=1e-2, m=12)
        assert result.converged
        assert result.matrix_shape == (3, 3)

    @pytest.mark.parametrize("problem", ["square_root"])
    @pytest.mark.parametrize("lo", [0.0, -0.5])
    def test_requires_positive_definite(self, problem, lo):
        with pytest.raises(ValueError, match="positive-definite"):
            solve_fe_coefficients(problem, sigma_min=lo, sigma_max=1.0)

    @pytest.mark.parametrize("problem", ["exponentiation", "square_root"])
    def test_singular_values_only_for_odd_problems(self, problem):
        with pytest.raises(ValueError, match="singular values only support"):
            solve_fe_coefficients(problem, sigma_min=0.5, sigma_max=1.0, spectral_kind="singular_values")

    def test_eigenvalues_require_hermitian(self):
        with pytest.raises(ValueError, match="Hermitian"):
            solve_fe_coefficients("exponentiation", matrix=[[1.0, 1.0], [0.0, 1.0]])

    def test_indefinite_inversion_requires_gap(self):
        with pytest.raises(ValueError, match="gap"):
            solve_fe_coefficients("inversion", sigma_min=-1.0, sigma_max=2.0)

    @pytest.mark.parametrize(
        "kwargs",
        [
            dict(sigma_min=-1.0, sigma_max=2.0, gap=(-1.5, 0.5)),  # a < sigma_min
            dict(sigma_min=-1.0, sigma_max=2.0, gap=(0.1, 0.5)),  # a > 0
            dict(sigma_min=-1.0, sigma_max=2.0, gap=(-0.5, 2.5)),  # b > sigma_max
            dict(sigma_min=0.5, sigma_max=2.0, gap=(-0.5, 0.5)),  # definite
            dict(sigma_min=0.0, sigma_max=2.0, gap=(-0.5, 0.5)),  # endpoint at the pole
            dict(sigma_min=0.5, sigma_max=2.0, gap=(-0.5, 0.5), spectral_kind="singular_values"),
        ],
    )
    def test_invalid_gap_raises(self, kwargs):
        with pytest.raises(ValueError):
            solve_fe_coefficients("inversion", **kwargs)

    def test_gap_only_for_inversion(self):
        with pytest.raises(ValueError, match="gap only applies"):
            solve_fe_coefficients("action", sigma_min=-1.0, sigma_max=2.0, gap=(-0.5, 0.5))

    def test_gap_and_matrix_raises(self):
        with pytest.raises(ValueError, match="not both"):
            solve_fe_coefficients("inversion", matrix=np.diag([-1.0, 1.0]), gap=(-1.0, 1.0))

    def test_singular_matrix_inversion_raises(self):
        with pytest.raises(ValueError, match="clear of 0"):
            solve_fe_coefficients("inversion", matrix=np.diag([0.0, 1.0]))

    @pytest.mark.parametrize("method", ["reflected", "arcsine_taylor"])
    @pytest.mark.parametrize(
        "kwargs",
        [
            dict(sigma_min=0.5, sigma_max=2.0, spectral_kind="singular_values"),
            dict(sigma_min=-2.0, sigma_max=2.0, gap=(-0.5, 0.5)),
        ],
    )
    def test_two_interval_inversion_rejected(self, method, kwargs):
        with pytest.raises(ValueError, match="two-interval"):
            solve_fe_coefficients("inversion", method=method, **kwargs)

    def test_eta_ranges(self):
        with pytest.raises(ValueError, match="eta >= 2"):
            solve_fe_coefficients("action", method="reflected", sigma_min=-1.0, sigma_max=1.0, eta=1.5)
        with pytest.raises(ValueError, match="eta > 2"):
            solve_fe_coefficients("action", method="arcsine_taylor", sigma_min=-1.0, sigma_max=1.0, eta=2.0)

    @pytest.mark.parametrize("method", ["reflected", "arcsine_taylor"])
    def test_extension_must_avoid_the_singularity(self, method):
        # 2 mu / delta = 2.1 for [0.025, 1].
        with pytest.raises(ValueError, match=r"2\|mu\|/delta = 2\.1"):
            solve_fe_coefficients("square_root", method=method, sigma_min=0.025, sigma_max=1.0, eta=2.7)
        r = solve_fe_coefficients("square_root", method=method, sigma_min=0.025, sigma_max=1.0, eta=2.05)
        assert np.isfinite(r.alpha) and r.alpha < 5.0


# ---------------------------------------------------------------------------
# Sobolev options
# ---------------------------------------------------------------------------


class TestSobolevOptions:
    common = dict(sigma_min=0.5, sigma_max=1.5, target_error=1e-4, m=12)

    def test_defaults_recorded(self):
        result = solve_fe_coefficients("exponentiation", **self.common)
        assert result.sobolev_w == 1.0
        assert result.sobolev_r == default_sobolev_r("exponentiation", sigma_min=0.5, sigma_max=1.5)

    def test_explicit_values_recorded(self):
        result = solve_fe_coefficients("exponentiation", sobolev_r=0.5, sobolev_w=2.0, **self.common)
        assert result.sobolev_r == 0.5
        assert result.sobolev_w == 2.0
        assert result.converged

    def test_r_upper_bound_allowed(self):
        assert solve_fe_coefficients("exponentiation", sobolev_r=1.0, **self.common).converged

    @pytest.mark.parametrize("r", [0.0, 1.5])
    def test_r_out_of_range_raises(self, r):
        with pytest.raises(ValueError, match="sobolev_r"):
            solve_fe_coefficients("exponentiation", sobolev_r=r, **self.common)

    @pytest.mark.parametrize("w", [0.0, 0.5])
    def test_w_out_of_range_raises(self, w):
        with pytest.raises(ValueError, match="sobolev_w"):
            solve_fe_coefficients("exponentiation", sobolev_w=w, **self.common)

    def test_solver_passes_eta_to_default_r(self):
        result = solve_fe_coefficients("exponentiation", eta=4.0, **self.common)
        assert result.sobolev_r == default_sobolev_r("exponentiation", sigma_min=0.5, sigma_max=1.5, eta=4.0)

    @pytest.mark.parametrize("option", ["lambdas", "sobolev_w", "sobolev_r", "order_fit", "fit_oversample"])
    def test_sobolev_options_rejected(self, option):
        value = {"lambdas": [1e-3], "sobolev_w": 1.0, "sobolev_r": 0.5, "order_fit": 0, "fit_oversample": 4}[option]
        with pytest.raises(ValueError, match=option):
            solve_fe_coefficients("action", method="reflected", **{option: value})


# ---------------------------------------------------------------------------
# Result fields
# ---------------------------------------------------------------------------


class TestResultFields:
    def setup_method(self):
        self.result = solve_fe_coefficients(
            "exponentiation",
            sigma_min=0.5,
            sigma_max=1.5,
            target_error=1e-2,
            m=12,
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


class TestExplicitMethodResults:
    @pytest.mark.parametrize("method", ["reflected", "arcsine_taylor"])
    def test_fields(self, method):
        r = solve_fe_coefficients("exponentiation", method=method, sigma_min=0.1, sigma_max=2.0, target_error=1e-2)
        assert r.method == method
        assert r.lambda_used is None and r.sobolev_r is None and r.sobolev_w is None
        assert all(c.method == method for c in r.all_results)
        assert max_error(r) == pytest.approx(r.achieved_linf_error, rel=1e-3)

    def test_sobolev_is_the_default(self):
        r = solve_fe_coefficients("exponentiation", sigma_min=0.1, sigma_max=2.0, target_error=1e-6, m=16)
        assert r.method == "sobolev"
        assert r.sobolev_w == 1.0


# ---------------------------------------------------------------------------
# Candidate selection
# ---------------------------------------------------------------------------


class TestSelection:
    common = dict(sigma_min=0.1, sigma_max=2.0, target_error=1e-8, m_values=range(8, 41, 4))

    def test_smallest_alpha_is_default(self):
        result = solve_fe_coefficients("exponentiation", **self.common)
        feasible = [c for c in result.all_results if c.converged]
        assert result.alpha == min(c.alpha for c in feasible)
        assert [c.m for c in result.all_results] == list(self.common["m_values"])

    def test_smallest_m_stops_at_first_feasible(self):
        result = solve_fe_coefficients("exponentiation", selection="smallest_m", **self.common)
        assert result.converged
        assert result.all_results[-1].m == result.m
        assert not any(c.converged for c in result.all_results[:-1])

    def test_unreachable_target_returns_the_best(self):
        r = solve_fe_coefficients(
            "exponentiation", method="reflected", sigma_min=0.1, sigma_max=2.0, m_values=[8, 16, 32]
        )
        assert not r.converged
        assert r.m == 32

    def test_selection_smallest_m(self):
        r = solve_fe_coefficients(
            "action",
            method="arcsine_taylor",
            sigma_min=-1.0,
            sigma_max=1.0,
            target_error=1e-8,
            selection="smallest_m",
        )
        assert r.converged
        assert r.all_results[-1].m == r.m


# ---------------------------------------------------------------------------
# Exact constants
# ---------------------------------------------------------------------------


class TestExactConstant:
    @pytest.mark.parametrize("problem, f", [("square_root", np.sqrt)])
    def test_zero_width_is_exact_constant(self, problem, f):
        result = solve_fe_coefficients(problem, sigma_min=2.0, sigma_max=2.0)
        assert result.m == 0
        assert result.a0 == pytest.approx(f(2.0))

    def test_exact_constant(self):
        r = solve_fe_coefficients("square_root", method="arcsine_taylor", sigma_min=4.0, sigma_max=4.0)
        assert r.m == 0 and r.a0 == 2.0
        assert r.method == "arcsine_taylor" and r.lambda_used is None


# ---------------------------------------------------------------------------
# compute_fe_coefficients convenience wrapper
# ---------------------------------------------------------------------------


class TestComputeFECoefficients:
    def test_returns_tuple(self):
        a0, ak, bk = compute_fe_coefficients(
            "exponentiation",
            sigma_min=0.5,
            sigma_max=1.5,
            target_error=1e-2,
            m=10,
        )
        assert isinstance(a0, float)
        assert isinstance(ak, np.ndarray)
        assert isinstance(bk, np.ndarray)
        assert len(ak) == len(bk)
