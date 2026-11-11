"""Quadratic continuations of the reflected extension, for eta > 2."""

import json
import math

import numpy as np
import pytest

from felcu import FourierExtensionResult, evaluate_fourier_series, fitted_domain_segments, solve_fe_coefficients

EXP = dict(method="reflected", sigma_min=0.1, sigma_max=2.0)
COMPLEX = dict(
    polynomial_coefficients=[1 + 0.5j, -0.3j, 0.25, 0.4 - 0.1j],
    polynomial_basis="chebyshev",
    method="reflected",
    sigma_min=-0.5,
    sigma_max=1.5,
)


def _profile_coefficients(g, eta, beta0, m, n=100_000):
    """beta_1..beta_m of the completed profile, built directly from the formula.

    Midpoint rule on [0, theta] and [theta, pi] separately, so the corner at
    theta falls on a panel edge.
    """
    x_max = math.pi / eta
    theta, ell = 2 * x_max, math.pi - 2 * x_max
    left, right = g(-x_max), g(x_max)
    x_star, q = (-x_max, 1.0) if abs(left) >= abs(right) else (x_max, -1.0)
    zeta = g(x_star) / abs(g(x_star))
    y1 = (np.arange(n) + 0.5) * theta / n
    y2 = theta + (np.arange(n) + 0.5) * ell / n
    h = np.conj(zeta) * g(x_star + q * y1)
    integral = np.sum(h) * theta / n
    h_theta = np.conj(zeta) * g(x_star + q * theta)
    s = (y2 - theta) / ell
    p = h_theta - 1.5 * (h_theta + (integral - math.pi * beta0) / ell) * (2 * s - s**2)
    k = np.arange(1, m + 1)
    beta = (2 / math.pi) * (np.cos(np.outer(k, y1)) @ h * theta / n + np.cos(np.outer(k, y2)) @ p * ell / n)
    mean = (integral + np.sum(p) * ell / n) / math.pi
    return zeta, x_star, mean, beta


@pytest.mark.convergence
class TestConstruction:
    @pytest.mark.parametrize("eta", [2.5, 3.0, 4.0])
    @pytest.mark.parametrize("beta0", [0.0, 1.3])
    def test_matches_the_completed_profile(self, eta, beta0):
        r = solve_fe_coefficients("exponentiation", eta=eta, m=8, closure_mean=beta0, **EXP)

        def g(x):
            return np.exp(r.mu + np.asarray(x) / r.tau)

        zeta, x_star, mean, beta = _profile_coefficients(g, eta, beta0, 8)
        assert mean == pytest.approx(beta0, abs=1e-8)  # the closure fixes the profile's mean
        k = np.arange(1, 9)
        assert r.a0 == pytest.approx(zeta * beta0, abs=1e-12)
        np.testing.assert_allclose(r.ak, zeta * beta * np.sin(k * x_star), atol=1e-8)
        np.testing.assert_allclose(r.bk, zeta * beta * np.cos(k * x_star), atol=1e-8)

    def test_complex_profile(self):
        r = solve_fe_coefficients("polynomial", eta=3.0, m=6, closure_mean=0.2 - 0.1j, **COMPLEX)
        p = np.polynomial.Chebyshev(COMPLEX["polynomial_coefficients"])

        def g(x):
            return p(r.mu + np.asarray(x) / r.tau)

        zeta, x_star, _, beta = _profile_coefficients(g, 3.0, 0.2 - 0.1j, 6)
        k = np.arange(1, 7)
        assert r.a0 == pytest.approx(zeta * (0.2 - 0.1j), abs=1e-12)
        np.testing.assert_allclose(r.ak, zeta * beta * np.sin(k * x_star), atol=1e-8)
        np.testing.assert_allclose(r.bk, zeta * beta * np.cos(k * x_star), atol=1e-8)

    @pytest.mark.parametrize("eta", [2.5, 3.0])
    def test_converges_on_the_fitted_interval(self, eta):
        errors = [solve_fe_coefficients("exponentiation", eta=eta, m=m, **EXP).achieved_linf_error for m in (32, 128)]
        assert 0.2 < errors[1] / errors[0] < 0.3  # O(1/m): the even extension has a corner at x*

    def test_fixed_mean_is_one_series_in_m(self):
        # Holding beta_0 fixed, the coefficients do not depend on m.
        r16 = solve_fe_coefficients("exponentiation", eta=3.0, m=16, closure_mean=1.0, **EXP)
        r32 = solve_fe_coefficients("exponentiation", eta=3.0, m=32, closure_mean=1.0, **EXP)
        np.testing.assert_allclose(r32.ak[:16], r16.ak, atol=1e-14)
        np.testing.assert_allclose(r32.bk[:16], r16.bk, atol=1e-14)

    def test_eta_two_is_the_ordinary_reflection(self):
        r = solve_fe_coefficients("identity", method="reflected", sigma_min=1.0, sigma_max=4.0, m=9)
        assert r.closure_mean is None and r.basis_used == "const_plus_sine"


@pytest.mark.convergence
class TestClosureMeanSelection:
    @pytest.mark.parametrize(
        "problem, kwargs",
        [
            ("exponentiation", EXP | dict(eta=2.5)),
            ("exponentiation", EXP | dict(eta=3.0)),
            ("square_root", dict(method="reflected", sigma_min=0.15, sigma_max=1.0, eta=3.0)),
            ("polynomial", COMPLEX | dict(eta=3.0)),
        ],
    )
    def test_minimises_alpha_without_raising_the_error(self, problem, kwargs):
        auto = solve_fe_coefficients(problem, m=32, **kwargs)
        zero = solve_fe_coefficients(problem, m=32, closure_mean=0.0, **kwargs)
        assert auto.achieved_linf_error <= zero.achieved_linf_error * (1 + 1e-9)
        assert auto.alpha <= zero.alpha
        # No nearby mean that meets the same error does better.
        b = auto.closure_mean
        for step in (0.05, -0.05, 0.05j, -0.05j) if isinstance(b, complex) else (0.05, -0.05):
            r = solve_fe_coefficients(problem, m=32, closure_mean=b + step, **kwargs)
            if r.achieved_linf_error <= zero.achieved_linf_error * (1 + 1e-9):
                assert r.alpha >= auto.alpha * (1 - 1e-9)

    def test_real_profile_gives_a_real_mean(self):
        r = solve_fe_coefficients("exponentiation", eta=3.0, m=16, **EXP)
        assert isinstance(r.closure_mean, float)
        assert r.ak.dtype == np.float64

    def test_complex_profile_gives_a_complex_mean(self):
        r = solve_fe_coefficients("polynomial", eta=3.0, m=16, **COMPLEX)
        assert isinstance(r.closure_mean, complex)

    def test_each_candidate_records_its_mean(self):
        r = solve_fe_coefficients("exponentiation", eta=3.0, m_values=[8, 16], target_error=1.0, **EXP)
        assert all(c.closure_mean is not None for c in r.all_results)

    def test_matches_the_series_on_the_fitted_interval(self):
        r = solve_fe_coefficients("polynomial", eta=3.0, m=64, **COMPLEX)
        p = np.polynomial.Chebyshev(COMPLEX["polynomial_coefficients"])
        x = fitted_domain_segments(r)[0]
        error = np.max(np.abs(evaluate_fourier_series(x, r.a0, r.ak, r.bk) - p(r.mu + x / r.tau)))
        assert error == pytest.approx(r.achieved_linf_error, rel=1e-3)


@pytest.mark.interface
class TestInterface:
    def test_full_basis(self):
        r = solve_fe_coefficients("identity", method="reflected", sigma_min=1.0, sigma_max=4.0, eta=3.0, m=8)
        assert r.basis_used == "full"

    def test_closure_mean_only_for_reflected(self):
        with pytest.raises(ValueError, match="closure_mean only applies to method='reflected'"):
            solve_fe_coefficients("exponentiation", eta=3.0, m=8, closure_mean=1.0, sigma_min=0.1, sigma_max=2.0)

    def test_closure_mean_needs_eta_above_two(self):
        with pytest.raises(ValueError, match="eta > 2"):
            solve_fe_coefficients("exponentiation", m=8, closure_mean=1.0, **EXP)

    @pytest.mark.parametrize("value", [np.nan, np.inf, "1.0"])
    def test_closure_mean_must_be_a_finite_number(self, value):
        with pytest.raises(ValueError, match="closure_mean must be"):
            solve_fe_coefficients("exponentiation", eta=3.0, m=8, closure_mean=value, **EXP)

    def test_singular_values_rejected(self):
        with pytest.raises(ValueError, match="not odd"):
            solve_fe_coefficients("identity", method="reflected", matrix=[[1.0, 0.5], [-0.2, 0.8]], eta=3.0, m=8)

    def test_json_round_trip(self):
        r = solve_fe_coefficients("polynomial", eta=3.0, m=8, **COMPLEX)
        data = json.loads(r.to_json())
        assert set(data["closure_mean"]) == {"real", "imag"}
        assert FourierExtensionResult.from_json(r.to_json()).closure_mean == r.closure_mean
        real = solve_fe_coefficients("exponentiation", eta=3.0, m=8, **EXP)
        assert FourierExtensionResult.from_json(real.to_json()).closure_mean == real.closure_mean
