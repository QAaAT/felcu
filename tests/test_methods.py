"""Tests for the explicit methods: reflected extensions and arcsine-Taylor."""

import math

import numpy as np
import pytest

from felcu import (
    FourierExtensionResult,
    canonical_method_name,
    default_eta,
    evaluate_fourier_series,
    solve_fe_coefficients,
    to_complex_coefficients,
)
from felcu.utils import evaluate_target_function, fitted_domain_segments

# (problem, kwargs, ||f(H)||) for convex endpoint-maximising f, where reflection is norm-saturating.
SATURATING = [
    ("action", dict(shifted=True, sigma_min=-1.0, sigma_max=3.0), 3.0),
    ("action", dict(sigma_min=-2.0, sigma_max=1.0), 2.0),
    ("exponentiation", dict(shifted=True, sigma_min=0.1, sigma_max=2.0), math.exp(2.0)),
    ("exponentiation", dict(sigma_min=-0.5, sigma_max=1.5), math.exp(1.5)),
    ("inversion", dict(shifted=True, sigma_min=0.5, sigma_max=1.0), 2.0),
    ("inversion", dict(shifted=True, sigma_min=-1.0, sigma_max=-0.2), 5.0),
]


def _max_error(result, n=3001):
    errors = []
    for xs in fitted_domain_segments(result, num=n):
        y_fit = evaluate_fourier_series(xs, result.a0, result.ak, result.bk)
        errors.append(np.max(np.abs(y_fit - evaluate_target_function(xs, result))))
    return max(errors)


class TestReflected:
    @pytest.mark.parametrize("problem, kwargs, norm", SATURATING)
    def test_saturates_the_norm_bound(self, problem, kwargs, norm):
        # Optimal subnormalisation: alpha_m = ||f(H)|| - epsilon_m, with the error attained at an endpoint.
        for m in (8, 33, 128):
            r = solve_fe_coefficients(problem, method="reflected", m=m, **kwargs)
            assert r.eta == 2.0
            assert r.alpha + r.achieved_linf_error == pytest.approx(norm, rel=1e-10)

    def test_algebraic_convergence(self):
        kw = dict(method="reflected", shifted=True, sigma_min=0.1, sigma_max=2.0)
        e1 = solve_fe_coefficients("exponentiation", m=32, **kw).achieved_linf_error
        e4 = solve_fe_coefficients("exponentiation", m=128, **kw).achieved_linf_error
        assert 0.2 < e4 / e1 < 0.3

    def test_identity_is_the_triangle_wave(self):
        # Sec. III A: a_k = (4 / (pi k^2)) sin(k pi / 2) / tau, with the constant mu exact.
        r = solve_fe_coefficients("action", method="reflected", shifted=True, sigma_min=1.0, sigma_max=4.0, m=9)
        k = np.arange(1, 10)
        np.testing.assert_allclose(r.ak, 4 / (np.pi * k**2) * np.sin(k * np.pi / 2) / r.tau, atol=1e-14)
        assert r.a0 == pytest.approx(2.5, abs=1e-14)
        assert r.basis_used == "const_plus_sine"

    def test_square_root_is_bounded(self):
        kw = dict(method="reflected", shifted=True, sigma_min=0.15, sigma_max=1.0)
        alphas = [solve_fe_coefficients("square_root", m=m, **kw).alpha for m in (16, 64, 256)]
        assert alphas == sorted(alphas)
        assert alphas[-1] < 1.1

    def test_larger_eta_moves_the_endpoint_inside(self):
        kw = dict(method="reflected", shifted=True, sigma_min=0.1, sigma_max=2.0, m=64)
        r2 = solve_fe_coefficients("exponentiation", **kw)
        r3 = solve_fe_coefficients("exponentiation", eta=3.0, **kw)
        assert r3.achieved_linf_error < r2.achieved_linf_error / 2
        assert r3.alpha > r2.alpha


class TestArcsineTaylor:
    def test_identity_coefficients_match_the_paper(self):
        # Eq. (arcsin_lcu_coefficients) with m = 2T + 1, scaled by 1/tau.
        T = 6
        r = solve_fe_coefficients(
            "action", method="arcsine_taylor", shifted=True, sigma_min=-1.0, sigma_max=3.0, m=2 * T + 1
        )
        c = [
            (-1) ** k
            * sum(math.comb(2 * j, j) * math.comb(2 * j + 1, j - k) / (16**j * (2 * j + 1)) for j in range(k, T + 1))
            for k in range(T + 1)
        ]
        np.testing.assert_allclose(r.ak[0::2], np.array(c) / r.tau, rtol=1e-12, atol=1e-14)
        np.testing.assert_allclose(r.ak[1::2], 0.0, atol=1e-14)
        assert r.a0 == pytest.approx(1.0, abs=1e-14)
        # Eq. (arcsin_subnormalization_definition).
        series = sum(math.comb(2 * k, k) / (4**k * (2 * k + 1)) for k in range(T + 1))
        assert r.alpha == pytest.approx(1.0 + series / r.tau, rel=1e-13)

    @pytest.mark.parametrize(
        "problem, kwargs, eta",
        [
            ("action", dict(shifted=True, sigma_min=-1.0, sigma_max=3.0), 3.0),
            ("exponentiation", dict(shifted=True, sigma_min=0.1, sigma_max=2.0), 3.0),
            ("inversion", dict(shifted=True, sigma_min=0.5, sigma_max=1.0), 2.5),
            ("square_root", dict(shifted=True, sigma_min=0.15, sigma_max=1.0), 2.7),
            ("exponentiation", {}, 3.0),
        ],
    )
    def test_exponential_convergence(self, problem, kwargs, eta):
        rho = math.sin(math.pi / eta)
        kw = dict(method="arcsine_taylor", eta=eta, **kwargs)
        e40 = solve_fe_coefficients(problem, m=40, **kw).achieved_linf_error
        e80 = solve_fe_coefficients(problem, m=80, **kw).achieved_linf_error
        assert e80 < 2 * rho**40 * e40

    @pytest.mark.parametrize("problem, kwargs, norm", SATURATING[:1] + SATURATING[2:3] + SATURATING[4:5])
    def test_alpha_approaches_the_enlarged_endpoint(self, problem, kwargs, norm):
        # Endpoint-saturated subnormalisation: alpha_inf = |f(mu + q eta delta / 2)|, approached from below.
        eta = 2.5
        lo, hi = kwargs["sigma_min"], kwargs["sigma_max"]
        mu, delta = (lo + hi) / 2, (hi - lo) / 2
        f = {"action": lambda t: t, "exponentiation": math.exp, "inversion": lambda t: 1 / t}[problem]
        limit = max(abs(f(mu - eta * delta / 2)), abs(f(mu + eta * delta / 2)))
        kw = dict(method="arcsine_taylor", eta=eta, **kwargs)
        alphas = [solve_fe_coefficients(problem, m=m, **kw).alpha for m in (64, 256)]
        assert norm < alphas[0] < alphas[1] < limit
        # The tail of the arcsin series decays only as T^(-1/2).
        assert alphas[1] == pytest.approx(limit, rel=0.1)

    def test_reaches_tight_targets(self):
        r = solve_fe_coefficients(
            "exponentiation", method="arcsine_taylor", shifted=True, sigma_min=0.1, sigma_max=2.0, target_error=1e-12
        )
        assert r.converged
        assert _max_error(r) <= 1.01e-12

    def test_block_encoding_error(self):
        rng = np.random.default_rng(0)
        Q, _ = np.linalg.qr(rng.standard_normal((6, 6)))
        lam = np.array([0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
        H = Q @ np.diag(lam) @ Q.T
        r = solve_fe_coefficients("inversion", method="arcsine_taylor", shifted=True, matrix=H, m=60)
        mu = 0.5 * (r.sigma_min + r.sigma_max)
        w, V = np.linalg.eigh(r.tau * (H - mu * np.eye(6)))
        k = np.arange(-r.m, r.m + 1)
        g_m = V @ np.diag(np.exp(1j * np.outer(w, k)) @ to_complex_coefficients(r)) @ V.T
        # Attained at the spectral endpoint, up to round-off in the matrix products.
        assert np.linalg.norm(g_m - np.linalg.inv(H), 2) <= r.achieved_linf_error + 1e-13


class TestDefaultEta:
    def test_reflected(self):
        assert default_eta("exponentiation", method="reflected") == 2.0
        assert default_eta("inversion", method="refl", shifted=True, sigma_min=0.1, sigma_max=1.0) == 2.0

    def test_arcsine_taylor(self):
        assert default_eta("action", method="arcsine_taylor") == 3.0
        # Midpoint of 2 < eta < 2 mu / delta, capped at 3.
        assert default_eta("inversion", method="asin", shifted=True, sigma_min=0.1, sigma_max=1.0) == pytest.approx(
            1 + 1.1 / 0.9
        )
        assert default_eta("square_root", method="asin", shifted=True, sigma_min=0.5, sigma_max=1.0) == 3.0
        assert default_eta("inversion", method="asin", shifted=True, sigma_min=2.0, sigma_max=2.0) == 3.0

    def test_sobolev_unchanged(self):
        assert default_eta("inversion", shifted=True, sigma_min=0.1, sigma_max=1.0) == pytest.approx(1.1 / 0.9)


class TestValidation:
    def test_method_aliases(self):
        assert canonical_method_name(" Regularized ") == "sobolev"
        assert canonical_method_name("refl") == "reflected"
        assert canonical_method_name("arcsine-taylor") == "arcsine_taylor"
        with pytest.raises(ValueError, match="method must be"):
            canonical_method_name("chebyshev")

    @pytest.mark.parametrize("method", ["reflected", "arcsine_taylor"])
    def test_unshifted_inversion_rejected(self, method):
        with pytest.raises(ValueError, match="unshifted inversion"):
            solve_fe_coefficients("inversion", method=method, sigma_min=0.5, sigma_max=2.0)

    def test_eta_ranges(self):
        with pytest.raises(ValueError, match="eta >= 2"):
            solve_fe_coefficients("action", method="reflected", eta=1.5)
        with pytest.raises(ValueError, match="eta > 2"):
            solve_fe_coefficients("action", method="arcsine_taylor", eta=2.0)

    @pytest.mark.parametrize("method", ["reflected", "arcsine_taylor"])
    def test_extension_must_avoid_the_singularity(self, method):
        # 2 mu / delta = 2.1 for [0.025, 1].
        with pytest.raises(ValueError, match=r"2\|mu\|/delta = 2\.1"):
            solve_fe_coefficients("square_root", method=method, shifted=True, sigma_min=0.025, sigma_max=1.0, eta=2.7)
        r = solve_fe_coefficients("square_root", method=method, shifted=True, sigma_min=0.025, sigma_max=1.0, eta=2.05)
        assert np.isfinite(r.alpha) and r.alpha < 5.0

    @pytest.mark.parametrize("option", ["lambdas", "sobolev_w", "sobolev_r", "order_fit", "fit_oversample"])
    def test_sobolev_options_rejected(self, option):
        value = {"lambdas": [1e-3], "sobolev_w": 1.0, "sobolev_r": 0.5, "order_fit": 0, "fit_oversample": 4}[option]
        with pytest.raises(ValueError, match=option):
            solve_fe_coefficients("action", method="reflected", **{option: value})


class TestResult:
    @pytest.mark.parametrize("method", ["reflected", "arcsine_taylor"])
    def test_fields(self, method):
        r = solve_fe_coefficients(
            "exponentiation", method=method, shifted=True, sigma_min=0.1, sigma_max=2.0, target_error=1e-2
        )
        assert r.method == method
        assert r.lambda_used is None and r.sobolev_r is None and r.sobolev_w is None
        assert all(c.method == method for c in r.all_results)
        assert _max_error(r) == pytest.approx(r.achieved_linf_error, rel=1e-3)

    def test_sobolev_is_the_default(self):
        r = solve_fe_coefficients("exponentiation", sigma_min=0.1, sigma_max=2.0, target_error=1e-6, m=16)
        assert r.method == "sobolev"
        assert r.sobolev_w == 1.0

    def test_unreachable_target_returns_the_best(self):
        r = solve_fe_coefficients(
            "exponentiation", method="reflected", shifted=True, sigma_min=0.1, sigma_max=2.0, m_values=[8, 16, 32]
        )
        assert not r.converged
        assert r.m == 32

    def test_selection_smallest_m(self):
        r = solve_fe_coefficients(
            "action",
            method="arcsine_taylor",
            shifted=True,
            sigma_min=-1.0,
            sigma_max=1.0,
            target_error=1e-8,
            selection="smallest_m",
        )
        assert r.converged
        assert r.all_results[-1].m == r.m

    def test_exact_constant(self):
        r = solve_fe_coefficients("square_root", method="arcsine_taylor", shifted=True, sigma_min=4.0, sigma_max=4.0)
        assert r.m == 0 and r.a0 == 2.0
        assert r.method == "arcsine_taylor" and r.lambda_used is None

    @pytest.mark.parametrize("method", ["reflected", "arcsine_taylor"])
    def test_json_round_trip(self, method):
        r = solve_fe_coefficients("inversion", method=method, shifted=True, sigma_min=0.5, sigma_max=1.0, m=12)
        loaded = FourierExtensionResult.from_json(r.to_json())
        assert loaded.method == method
        assert loaded.lambda_used is None and loaded.sobolev_r is None
        np.testing.assert_array_equal(loaded.ak, r.ak)
