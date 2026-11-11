"""The polynomial target: f(z) = sum_j gamma_j B_j(z) with monomial, Chebyshev or Legendre B_j."""

import json

import numpy as np
import pytest

from felcu import (
    FourierExtensionResult,
    default_eta,
    default_sobolev_r,
    evaluate_fourier_series,
    evaluate_target_function,
    fitted_domain_segments,
    solve_fe_coefficients,
)

BASES = {
    "monomial": np.polynomial.Polynomial,
    "chebyshev": np.polynomial.Chebyshev,
    "legendre": np.polynomial.Legendre,
}
REAL = np.array([0.5, -1.0, 0.25, 2.0, -0.75])
COMPLEX = np.array([1.0 + 0.5j, -0.3j, 0.25, 0.4 - 0.1j, -0.2 + 0.3j])
SPECTRUM = dict(sigma_min=-0.5, sigma_max=1.5)


def _poly(coefficients, basis="monomial", **kwargs):
    return solve_fe_coefficients(
        "polynomial", polynomial_coefficients=coefficients, polynomial_basis=basis, **(SPECTRUM | kwargs)
    )


def _max_error(result, f):
    """Max |g_m - f(mu + x/tau)| on the fitted set, against an independent evaluation of f."""
    return max(
        np.max(np.abs(evaluate_fourier_series(x, result.a0, result.ak, result.bk) - f(result.mu + x / result.tau)))
        for x in fitted_domain_segments(result, num=2001)
    )


@pytest.mark.unit
class TestEvaluation:
    @pytest.mark.parametrize("basis", list(BASES))
    @pytest.mark.parametrize("coefficients", [REAL, COMPLEX])
    def test_values_and_derivatives(self, basis, coefficients):
        from felcu._coefficients import _target

        p = BASES[basis](coefficients)
        x, tau, mu = np.linspace(-1.0, 1.0, 9), 1.7, 0.4
        t = mu + x / tau
        for d in range(7):  # beyond the degree, the derivative vanishes
            expected = p.deriv(d)(t) / tau**d if d else p(t)
            got = _target(x, "polynomial", tau, mu, d, (coefficients, basis))
            np.testing.assert_allclose(got, expected, rtol=1e-13, atol=1e-13)

    def test_monomial_coefficients_are_ascending(self):
        # The example of the notes: [1, 2, 3, 4] is 1 + 2z + 3z^2 + 4z^3.
        r = _poly([1, 2, 3, 4], m=4)
        z = 1.25
        x = np.array([(z - r.mu) * r.tau])
        np.testing.assert_allclose(evaluate_target_function(x, r), [1 + 2 * z + 3 * z**2 + 4 * z**3])

    @pytest.mark.parametrize(
        "basis, value", [("chebyshev", 4 * 1.5**3 - 3 * 1.5), ("legendre", (5 * 1.5**3 - 3 * 1.5) / 2)]
    )
    def test_basis_acts_on_the_physical_argument(self, basis, value):
        # No mapping of the spectral interval onto [-1, 1]: B_3 is evaluated at z itself.
        r = _poly([0, 0, 0, 1], basis, sigma_min=1.0, sigma_max=2.0, m=4)
        x = np.array([(1.5 - r.mu) * r.tau])
        np.testing.assert_allclose(evaluate_target_function(x, r), [value], rtol=1e-14)

    @pytest.mark.parametrize("basis", list(BASES))
    def test_series_composition(self, basis):
        # Taylor coefficients of f(u(z)) for u(z) = mu + z, against numpy's basis conversion.
        from felcu._explicit import _series_polynomial

        u = np.zeros(10)
        u[:2] = 0.3, 1.0
        expected = BASES[basis](COMPLEX).convert(kind=np.polynomial.Polynomial)(np.polynomial.Polynomial(u[:2]))
        got = _series_polynomial(u, (COMPLEX, basis))
        np.testing.assert_allclose(got[: expected.coef.size], expected.coef, atol=1e-13)
        np.testing.assert_array_equal(got[expected.coef.size :], 0.0)


@pytest.mark.convergence
class TestConvergence:
    @pytest.mark.parametrize("basis", list(BASES))
    @pytest.mark.parametrize("coefficients", [REAL, COMPLEX])
    @pytest.mark.parametrize("method", ["sobolev", "arcsine_taylor"])
    def test_meets_the_target(self, basis, coefficients, method):
        # Arcsine-Taylor converges like sin(pi/3)^m, so needs larger orders, but each one is cheap.
        m_values = [16, 24, 32] if method == "sobolev" else range(64, 257, 32)
        r = _poly(coefficients, basis, method=method, target_error=1e-9, m_values=m_values)
        assert r.converged
        assert _max_error(r, BASES[basis](coefficients)) <= 1.01e-9

    @pytest.mark.parametrize("basis", list(BASES))
    def test_reflected_converges_algebraically(self, basis):
        e = [_poly(COMPLEX, basis, method="reflected", m=m).achieved_linf_error for m in (64, 256)]
        assert 0.2 < e[1] / e[0] < 0.3

    def test_matrix_function(self):
        rng = np.random.default_rng(3)
        Q, _ = np.linalg.qr(rng.standard_normal((6, 6)))
        lam = np.linspace(-0.5, 1.5, 6)
        H = Q @ np.diag(lam) @ Q.T
        r = solve_fe_coefficients(
            "polynomial",
            polynomial_coefficients=COMPLEX,
            polynomial_basis="chebyshev",
            matrix=H,
            target_error=1e-9,
            m_values=[16, 24, 32],
        )
        g = evaluate_fourier_series(r.tau * (lam - r.mu), r.a0, r.ak, r.bk)
        exact = Q @ np.diag(np.polynomial.Chebyshev(COMPLEX)(lam)) @ Q.T
        np.testing.assert_allclose(Q @ np.diag(g) @ Q.T, exact, atol=1e-8)


@pytest.mark.interface
class TestInterface:
    def test_defaults_are_the_entire_function_ones(self):
        r = _poly(REAL, m=16)
        assert r.eta == 2.0 and r.sobolev_r == 1.0 and r.basis_used == "full"
        assert default_eta("polynomial") == 2.0
        assert default_eta("polynomial", method="arcsine_taylor") == 3.0
        assert default_sobolev_r("polynomial") == 1.0
        assert _poly(REAL, method="arcsine_taylor", m=16).eta == 3.0

    def test_result_fields(self):
        r = _poly(COMPLEX, "legendre", m=16)
        assert r.problem == "polynomial" and r.polynomial_basis == "legendre"
        np.testing.assert_array_equal(r.polynomial_coefficients, COMPLEX)
        assert r.ak.dtype == np.complex128
        real = _poly([1, 2, 3], m=8)
        assert real.polynomial_coefficients.dtype == np.float64 and real.polynomial_basis == "monomial"
        assert real.ak.dtype == np.float64
        assert solve_fe_coefficients("exponentiation", m=8, **SPECTRUM).polynomial_coefficients is None

    @pytest.mark.parametrize("alias, basis", [("power", "monomial"), ("Cheb", "chebyshev"), (" LEG ", "legendre")])
    def test_basis_aliases(self, alias, basis):
        assert _poly(REAL, alias, m=4).polynomial_basis == basis

    def test_zero_width_interval_is_exact(self):
        r = _poly(COMPLEX, "chebyshev", sigma_min=0.7, sigma_max=0.7)
        assert r.m == 0
        assert r.a0 == pytest.approx(complex(np.polynomial.Chebyshev(COMPLEX)(0.7)))

    @pytest.mark.parametrize(
        "kwargs, match",
        [
            (dict(), "needs polynomial_coefficients"),
            (dict(polynomial_coefficients=[]), "non-empty"),
            (dict(polynomial_coefficients=[[1.0, 2.0]]), "non-empty"),
            (dict(polynomial_coefficients=["a"]), "numbers"),
            (dict(polynomial_coefficients=[1.0, np.inf]), "finite"),
            (dict(polynomial_coefficients=[1.0], polynomial_basis="hermite"), "polynomial_basis"),
        ],
    )
    def test_validation(self, kwargs, match):
        with pytest.raises(ValueError, match=match):
            solve_fe_coefficients("polynomial", **SPECTRUM, **kwargs)

    @pytest.mark.parametrize("option", ["polynomial_coefficients", "polynomial_basis"])
    def test_options_only_for_polynomial(self, option):
        value = {"polynomial_coefficients": [1.0], "polynomial_basis": "monomial"}[option]
        with pytest.raises(ValueError, match="only apply to problem='polynomial'"):
            solve_fe_coefficients("exponentiation", **SPECTRUM, **{option: value})

    def test_singular_values_not_supported(self):
        with pytest.raises(ValueError, match="singular values only support"):
            _poly(REAL, sigma_min=0.5, sigma_max=1.0, spectral_kind="singular_values")

    def test_json_round_trip(self):
        r = _poly(COMPLEX, "chebyshev", m_values=[8, 16], target_error=1e-6)
        data = json.loads(r.to_json())
        assert data["polynomial_basis"] == "chebyshev" and set(data["polynomial_coefficients"]) == {"real", "imag"}
        loaded = FourierExtensionResult.from_json(r.to_json())
        np.testing.assert_array_equal(loaded.polynomial_coefficients, COMPLEX)
        x = fitted_domain_segments(r)[0]
        np.testing.assert_array_equal(evaluate_target_function(x, loaded), evaluate_target_function(x, r))

    def test_files_without_the_polynomial_fields_load(self):
        data = solve_fe_coefficients("exponentiation", m=8, **SPECTRUM).to_dict()
        del data["polynomial_coefficients"], data["polynomial_basis"]
        loaded = FourierExtensionResult.from_dict(data)
        assert loaded.polynomial_coefficients is None and loaded.polynomial_basis is None
