"""Complex-valued functions on a real spectrum.

No built-in problem is complex-valued yet, so these tests swap the
exponential's target for the complex entire function f(t) = exp(c t), with
c = 0.5 + 1j, in every place the solver reads it: the fitted target and its
derivatives, and the arcsine-Taylor series composition.
"""

import json
import math

import numpy as np
import pytest

import felcu._coefficients
import felcu._explicit
import felcu.utils
from felcu import (
    FourierExtensionResult,
    evaluate_fourier_series,
    evaluate_target_function,
    fitted_domain_segments,
    solve_fe_coefficients,
)

C = 0.5 + 1j
SPECTRUM = dict(sigma_min=0.1, sigma_max=2.0)


def f(t):
    return np.exp(C * np.asarray(t))


@pytest.fixture
def complex_exponential(monkeypatch):
    original = felcu._coefficients._target

    def target(x, problem, tau, mu, d=0, polynomial=None):
        if problem != "exponentiation":
            return original(x, problem, tau, mu, d, polynomial)
        x = np.asarray(x, dtype=float)
        t = x / tau + mu
        return (C / tau) ** d * np.exp(C * t)

    monkeypatch.setattr(felcu._coefficients, "_target", target)
    monkeypatch.setattr(felcu.utils, "_target", target)
    monkeypatch.setitem(felcu._explicit._COMPOSE, "exponentiation", lambda u: felcu._explicit._series_exp(C * u))


def _max_error(result):
    """Max |g_m - g| on the fitted set, against f itself."""
    errors = []
    for x in fitted_domain_segments(result, num=2001):
        g = f(result.mu + x / result.tau)
        errors.append(np.max(np.abs(evaluate_fourier_series(x, result.a0, result.ak, result.bk) - g)))
    return max(errors)


@pytest.mark.convergence
@pytest.mark.usefixtures("complex_exponential")
class TestConvergence:
    def test_sobolev(self):
        r = solve_fe_coefficients("exponentiation", target_error=1e-10, m_values=range(8, 65, 8), **SPECTRUM)
        assert r.converged
        assert r.ak.dtype == np.complex128 and isinstance(r.a0, complex)
        assert _max_error(r) <= 1.01e-10
        np.testing.assert_allclose(evaluate_target_function(np.array([0.0]), r), f([r.mu]))

    def test_reflected_converges_algebraically(self):
        kw = dict(method="reflected", **SPECTRUM)
        e1 = solve_fe_coefficients("exponentiation", m=32, **kw).achieved_linf_error
        e4 = solve_fe_coefficients("exponentiation", m=128, **kw).achieved_linf_error
        assert 0.2 < e4 / e1 < 0.3

    def test_arcsine_taylor_converges_exponentially(self):
        kw = dict(method="arcsine_taylor", eta=3.0, **SPECTRUM)
        rho = math.sin(math.pi / 3.0)
        e40 = solve_fe_coefficients("exponentiation", m=40, **kw).achieved_linf_error
        e80 = solve_fe_coefficients("exponentiation", m=80, **kw).achieved_linf_error
        assert e80 < 2 * rho**40 * e40
        r = solve_fe_coefficients("exponentiation", method="arcsine_taylor", target_error=1e-10, **SPECTRUM)
        assert r.converged and _max_error(r) <= 1.01e-10

    def test_arcsine_taylor_is_the_sine_power_series(self):
        # The FFT conversion of sum_k d_k sin^k x must not assume c_{-k} = conj(c_k).
        from felcu._explicit import _taylor_coefficients
        from felcu.spectral import _make_spec, _spectral_data

        spectral = _spectral_data("exponentiation", None, 0.1, 2.0, None, "auto", 1e-12)
        spec = _make_spec("exponentiation", spectral, 3.0, "arcsine_taylor")
        m = 24
        d = _taylor_coefficients(spec, m)
        r = solve_fe_coefficients("exponentiation", method="arcsine_taylor", eta=3.0, m=m, **SPECTRUM)
        x = np.linspace(-np.pi, np.pi, 301)
        direct = np.polynomial.polynomial.polyval(np.sin(x), d)
        np.testing.assert_allclose(evaluate_fourier_series(x, r.a0, r.ak, r.bk), direct, rtol=0, atol=1e-13)

    @pytest.mark.parametrize("method", ["sobolev", "reflected", "arcsine_taylor"])
    def test_zero_width_interval_is_the_exact_complex_constant(self, method):
        r = solve_fe_coefficients("exponentiation", method=method, sigma_min=0.7, sigma_max=0.7)
        assert r.m == 0
        assert r.a0 == pytest.approx(complex(f(0.7)))
        assert r.alpha == pytest.approx(abs(f(0.7)))


@pytest.mark.interface
@pytest.mark.usefixtures("complex_exponential")
class TestResult:
    @pytest.fixture
    def result(self):
        return solve_fe_coefficients("exponentiation", target_error=1e-8, m_values=[16, 24], **SPECTRUM)

    def test_modes_are_not_conjugate_pairs(self, result):
        c = result.complex_coefficients
        m = result.m
        assert not np.allclose(c[:m][::-1], np.conj(c[m + 1 :]))

    def test_alpha_sums_both_signed_modes(self, result):
        assert result.alpha == pytest.approx(np.sum(np.abs(result.complex_coefficients)), rel=1e-13)

    def test_sine_cosine_conversion(self, result):
        # a0 = c_0, a_k = i (c_k - c_{-k}), b_k = c_k + c_{-k}.
        c, m = result.complex_coefficients, result.m
        pos, neg = c[m + 1 :], c[:m][::-1]
        assert result.a0 == pytest.approx(c[m])
        np.testing.assert_allclose(result.ak, 1j * (pos - neg), rtol=0, atol=1e-15)
        np.testing.assert_allclose(result.bk, pos + neg, rtol=0, atol=1e-15)

    def test_json_round_trip_keeps_the_imaginary_parts(self, result):
        data = json.loads(result.to_json())
        assert data["format_version"] == 2
        assert set(data["ak"]) == {"real", "imag"} and set(data["a0"]) == {"real", "imag"}
        loaded = FourierExtensionResult.from_json(result.to_json(include_candidates=True))
        assert loaded.a0 == result.a0
        np.testing.assert_array_equal(loaded.ak, result.ak)
        np.testing.assert_array_equal(loaded.bk, result.bk)
        np.testing.assert_array_equal(loaded.complex_coefficients, result.complex_coefficients)
        assert loaded.all_results[0].ak.dtype == np.complex128


@pytest.mark.interface
class TestRealFunctionsStayReal:
    @pytest.mark.parametrize("method", ["sobolev", "reflected", "arcsine_taylor"])
    def test_real_dtype(self, method):
        r = solve_fe_coefficients("exponentiation", method=method, m=16, **SPECTRUM)
        assert isinstance(r.a0, float)
        assert r.ak.dtype == np.float64 and r.bk.dtype == np.float64
        data = json.loads(r.to_json())
        assert isinstance(data["a0"], float) and isinstance(data["ak"], list)

    def test_reads_format_version_1(self):
        r = solve_fe_coefficients("exponentiation", m=8, **SPECTRUM)
        data = r.to_dict()
        data["format_version"] = 1
        loaded = FourierExtensionResult.from_dict(data)
        np.testing.assert_array_equal(loaded.ak, r.ak)


@pytest.mark.unit
class TestHelpers:
    def test_lstsq_solves_real_and_imaginary_parts(self):
        from felcu._linalg import _lstsq

        rng = np.random.default_rng(0)
        A = rng.standard_normal((20, 6))
        b = rng.standard_normal(20) + 1j * rng.standard_normal(20)
        expected, *_ = np.linalg.lstsq(A.astype(complex), b, rcond=None)
        np.testing.assert_allclose(_lstsq(A, b), expected, rtol=1e-12)

    def test_alpha(self):
        from felcu._linalg import _alpha
        from felcu.utils import to_complex_coefficients

        rng = np.random.default_rng(1)
        a0 = 0.3 - 0.2j
        ak = rng.standard_normal(5) + 1j * rng.standard_normal(5)
        bk = rng.standard_normal(5) + 1j * rng.standard_normal(5)
        expected = np.sum(np.abs(to_complex_coefficients(a0, ak, bk)))
        assert _alpha(a0, ak, bk) == pytest.approx(expected, rel=1e-14)
        # Real coefficients take the sqrt(a^2 + b^2) shortcut, which agrees.
        assert _alpha(0.3, ak.real, bk.real) == pytest.approx(
            np.sum(np.abs(to_complex_coefficients(0.3, ak.real, bk.real))), rel=1e-14
        )

    def test_series_are_complex_aware(self):
        from felcu._explicit import _series_exp, _series_sqrt

        u = np.array([0.2 + 0.5j, 1.0, -0.3j, 0.1, 0.0, 0.05])
        y = _series_exp(u)
        assert y.dtype == np.complex128 and y[0] == pytest.approx(np.exp(0.2 + 0.5j))
        s = _series_sqrt(np.array([2.0 + 1j, 0.5, 0.1j, 0.0]))
        # s * s = u, to the truncation order.
        np.testing.assert_allclose(np.convolve(s, s)[:4], [2.0 + 1j, 0.5, 0.1j, 0.0], atol=1e-15)
