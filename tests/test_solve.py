"""Tests for felcu.solve (solve_fe_coefficients and compute_fe_coefficients)."""

import numpy as np
import pytest

from felcu import (
    compute_fe_coefficients,
    default_eta,
    default_sobolev_r,
    evaluate_fourier_series,
    solve_fe_coefficients,
)
from felcu.utils import evaluate_target_function, fitted_domain_segments

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
            sigma_min=0.1,
            sigma_max=2.0,
            target_error=1e-3,
            m=16,
        )
        assert result.converged
        assert result.achieved_l2_error <= 1e-3 * 1.01

    def test_action_converges(self):
        result = solve_fe_coefficients(
            "action",
            sigma_min=-1.0,
            sigma_max=1.0,
            target_error=1e-3,
            m=8,
        )
        assert result.converged

    def test_inversion_converges(self):
        result = solve_fe_coefficients(
            "inversion",
            sigma_min=0.5,
            sigma_max=2.0,
            target_error=1e-3,
            m=16,
        )
        assert result.converged

    def test_linf_metric(self):
        result = solve_fe_coefficients(
            "exponentiation",
            sigma_min=0.5,
            sigma_max=1.5,
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


# ---------------------------------------------------------------------------
# H_r^w regulariser
# ---------------------------------------------------------------------------


class TestSobolevRegulariser:
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

    @pytest.mark.parametrize(
        "problem, kwargs",
        [
            ("action", {}),
            ("action", dict(sigma_min=-1.0, sigma_max=2.0)),
            ("inversion", dict(sigma_min=0.5, sigma_max=3.0)),
            ("exponentiation", dict(shifted=True, sigma_min=1.0, sigma_max=3.0)),
        ],
    )
    def test_default_r_in_range(self, problem, kwargs):
        assert 0 < default_sobolev_r(problem, **kwargs) <= 1

    def test_default_r_exponentiation(self):
        # r_exp = 1 / (1 + eta delta / pi) shifted; unshifted uses the radius 3 instead of delta = 1
        kwargs = dict(sigma_min=1.0, sigma_max=3.0)
        assert default_sobolev_r("exponentiation", shifted=True, **kwargs) == pytest.approx(1 / (1 + 2.0 / np.pi))
        assert default_sobolev_r("exponentiation", shifted=True, eta=4.0, **kwargs) == pytest.approx(
            1 / (1 + 4.0 / np.pi)
        )
        assert default_sobolev_r("exponentiation", **kwargs) == pytest.approx(1 / (1 + 6.0 / np.pi))

    def test_default_r_inversion(self):
        # r_def = min(1, 2 pi / (kappa + 1)), r_indef = min(1, pi / (kappa + 1))
        assert default_sobolev_r("inversion", shifted=True, sigma_min=1.0, sigma_max=20.0) == pytest.approx(
            2 * np.pi / 21
        )
        assert default_sobolev_r("inversion", sigma_min=0.05, sigma_max=1.0) == pytest.approx(np.pi / 21)
        assert default_sobolev_r("inversion", sigma_min=0.5, sigma_max=1.0) == 1.0

    def test_solver_passes_eta_to_default_r(self):
        result = solve_fe_coefficients("exponentiation", eta=4.0, **self.common)
        assert result.sobolev_r == default_sobolev_r("exponentiation", sigma_min=0.5, sigma_max=1.5, eta=4.0)


# ---------------------------------------------------------------------------
# Square root
# ---------------------------------------------------------------------------


class TestSquareRoot:
    common = dict(shifted=True, sigma_min=0.15, sigma_max=1.0, target_error=1e-6, m_values=range(8, 65, 8))

    @pytest.mark.parametrize("problem", ["square_root"])
    def test_converges(self, problem):
        result = solve_fe_coefficients(problem, **self.common)
        assert result.converged
        assert result.basis_used == "full"
        assert _max_error(result) <= 1e-6 * 1.01

    @pytest.mark.parametrize("problem, f", [("sqrt", np.sqrt)])
    def test_matrix_function(self, problem, f):
        rng = np.random.default_rng(5)
        Q, _ = np.linalg.qr(rng.standard_normal((6, 6)))
        lam = np.linspace(0.3, 2.0, 6)
        H = Q @ np.diag(lam) @ Q.T
        r = solve_fe_coefficients(problem, shifted=True, matrix=H, target_error=1e-8, m_values=range(8, 65, 8))
        mu = 0.5 * (r.sigma_min + r.sigma_max)
        g = evaluate_fourier_series(r.tau * (lam - mu), r.a0, r.ak, r.bk)
        np.testing.assert_allclose(Q @ np.diag(g) @ Q.T, Q @ np.diag(f(lam)) @ Q.T, atol=1e-7)

    @pytest.mark.parametrize("problem", ["square_root"])
    def test_target_derivatives(self, problem):
        from felcu._coefficients import _target

        x, h = np.linspace(-0.9, 0.9, 7), 1e-5
        for d in (1, 2):
            fd = (_target(x + h, problem, 1.3, 2.0, d - 1) - _target(x - h, problem, 1.3, 2.0, d - 1)) / (2 * h)
            np.testing.assert_allclose(_target(x, problem, 1.3, 2.0, d), fd, rtol=1e-6)

    @pytest.mark.parametrize("problem", ["square_root"])
    def test_requires_shifted(self, problem):
        with pytest.raises(ValueError, match="shifted=True"):
            solve_fe_coefficients(problem, sigma_min=0.5, sigma_max=1.0)

    @pytest.mark.parametrize("problem", ["square_root"])
    @pytest.mark.parametrize("lo", [0.0, -0.5])
    def test_requires_positive_definite(self, problem, lo):
        with pytest.raises(ValueError, match="positive-definite"):
            solve_fe_coefficients(problem, shifted=True, sigma_min=lo, sigma_max=1.0)

    @pytest.mark.parametrize("problem", ["square_root"])
    def test_requires_spectral_information(self, problem):
        with pytest.raises(ValueError):
            solve_fe_coefficients(problem, shifted=True)

    @pytest.mark.parametrize("problem, f", [("square_root", np.sqrt)])
    def test_zero_width_is_exact_constant(self, problem, f):
        result = solve_fe_coefficients(problem, shifted=True, sigma_min=2.0, sigma_max=2.0)
        assert result.m == 0
        assert result.a0 == pytest.approx(f(2.0))

    @pytest.mark.parametrize("problem", ["square_root"])
    def test_default_eta_puts_branch_point_on_boundary(self, problem):
        kappa = 1.0 / 0.15
        assert default_eta(problem, shifted=True, sigma_min=0.15, sigma_max=1.0) == pytest.approx(
            (kappa + 1) / (kappa - 1)
        )

    def test_default_r(self):
        kwargs = dict(shifted=True, sigma_min=0.04, sigma_max=1.0)  # kappa = 25
        assert default_sobolev_r("square_root", **kwargs) == pytest.approx(25**-0.5)
        assert default_sobolev_r("square_root", sobolev_w=0.75, **kwargs) == pytest.approx(25 ** (-1 / 3))


# ---------------------------------------------------------------------------
# Candidate selection and the fast lambda solve
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


class TestLinearSolver:
    @pytest.mark.parametrize("lam", [1e-6, 1e-3, 1e-1])
    def test_matches_augmented_lstsq(self, lam):
        from felcu._linalg import _penalty
        from felcu._solver import _fit_system, _make_fit_data, _make_linear_solver
        from felcu.spectral import _make_spec, _spectral_data

        spectral = _spectral_data("exponentiation", False, None, 0.1, 2.0, "auto", 1e-12, True)
        spec = _make_spec("exponentiation", False, spectral, None, 1e-12)
        m, r, w = 16, 0.5, 1.5
        data = _make_fit_data(spec, m, 8, 401)

        fast = _make_linear_solver(data, spec, m, 1, r, w)(lam)

        A, b = _fit_system(data, spec, m, 1)
        p = _penalty(m, spec["basis"], lam, r, w)
        c, *_ = np.linalg.lstsq(np.vstack([A, np.diag(p)]), np.concatenate([b, np.zeros(len(p))]), rcond=None)
        # Two different solvers on an ill-conditioned system agree to rounding
        # relative to the coefficient vector, not to each (possibly tiny) entry.
        atol = 1e-9 * np.max(np.abs(c))
        np.testing.assert_allclose(fast["ak"], c[1 : m + 1], rtol=0, atol=atol)
        np.testing.assert_allclose(fast["bk"], c[m + 1 :], rtol=0, atol=atol)


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

    def test_no_scale_inversion_raises(self):
        with pytest.raises(ValueError):
            solve_fe_coefficients("inversion")  # inversion always needs a scale

    def test_shifted_no_scale_raises(self):
        with pytest.raises(ValueError):
            solve_fe_coefficients("action", shifted=True)

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
