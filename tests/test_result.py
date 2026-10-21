"""Tests for FourierExtensionResult.evaluate / apply and the ConvergenceWarning."""

import numpy as np
import pytest

from felcu import ConvergenceWarning, solve_fe_coefficients

# (problem, kwargs, f) fits whose spectral interval is [sigma_min, sigma_max].
FITS = [
    ("exponentiation", dict(sigma_min=0.1, sigma_max=2.0), np.exp),
    ("exponentiation", dict(shifted=True, sigma_min=0.1, sigma_max=2.0), np.exp),
    ("action", dict(shifted=True, sigma_min=-1.0, sigma_max=3.0), lambda x: x),
    ("inversion", dict(shifted=True, sigma_min=0.5, sigma_max=2.0), lambda x: 1.0 / x),
    ("square_root", dict(shifted=True, sigma_min=0.2, sigma_max=1.0), np.sqrt),
]


def _fit(problem, m=16, **kwargs):
    """A quick fit: fixed m, moderate target (as in test_solve.py)."""
    return solve_fe_coefficients(problem, target_error=1e-6, m=m, **kwargs)


def _random_hermitian(eigenvalues, seed=0, complex_=False):
    rng = np.random.default_rng(seed)
    n = len(eigenvalues)
    X = rng.standard_normal((n, n))
    if complex_:
        X = X + 1j * rng.standard_normal((n, n))
    Q, _ = np.linalg.qr(X)
    return (Q * eigenvalues) @ Q.conj().T


class TestEvaluate:
    @pytest.mark.parametrize("problem, kwargs, f", FITS)
    def test_matches_f_on_the_interval(self, problem, kwargs, f):
        r = _fit(problem, **kwargs)
        x = np.linspace(kwargs["sigma_min"], kwargs["sigma_max"], 201)
        np.testing.assert_allclose(r.evaluate(x), f(x), atol=1e-5)

    def test_unshifted_inversion_covers_both_signs(self):
        r = _fit("inversion", m=24, sigma_min=0.5, sigma_max=2.0)
        x = np.concatenate([np.linspace(-2.0, -0.5, 50), np.linspace(0.5, 2.0, 50)])
        np.testing.assert_allclose(r.evaluate(x), 1.0 / x, atol=1e-5)

    def test_keeps_the_input_shape(self):
        r = _fit("exponentiation", sigma_min=0.1, sigma_max=2.0)
        assert r.evaluate(np.ones((3, 4))).shape == (3, 4)
        assert r.evaluate(1.0).shape == ()

    def test_zero_width_interval(self):
        r = solve_fe_coefficients("exponentiation", shifted=True, sigma_min=1.5, sigma_max=1.5)
        np.testing.assert_allclose(r.evaluate([1.5]), [np.exp(1.5)])

    def test_mu(self):
        assert _fit("exponentiation", shifted=True, sigma_min=0.1, sigma_max=2.0).mu == 1.05
        assert _fit("exponentiation", sigma_min=0.1, sigma_max=2.0).mu == 0.0


class TestApply:
    @pytest.mark.parametrize("complex_", [False, True])
    def test_matches_the_exact_matrix_function(self, complex_):
        w = np.linspace(0.1, 2.0, 8)
        H = _random_hermitian(w, complex_=complex_)
        r = _fit("exponentiation", matrix=H)
        exact = _random_hermitian(np.exp(w), complex_=complex_)
        np.testing.assert_allclose(r.apply(H), exact, atol=1e-5)

    def test_shifted_inversion(self):
        w = np.linspace(0.5, 2.0, 6)
        H = _random_hermitian(w, seed=1)
        r = _fit("inversion", shifted=True, matrix=H)
        np.testing.assert_allclose(r.apply(H), np.linalg.inv(H), atol=1e-5)

    def test_rejects_non_hermitian(self):
        r = _fit("exponentiation", sigma_min=0.1, sigma_max=2.0)
        with pytest.raises(ValueError, match="Hermitian"):
            r.apply(np.array([[1.0, 1.0], [0.0, 1.0]]))

    def test_rejects_non_square(self):
        r = _fit("exponentiation", sigma_min=0.1, sigma_max=2.0)
        with pytest.raises(ValueError, match="square"):
            r.apply(np.ones((2, 3)))


class TestConvergenceWarning:
    def test_warns_when_the_target_is_missed(self):
        with pytest.warns(ConvergenceWarning, match="target_error=1e-12"):
            r = solve_fe_coefficients("exponentiation", sigma_min=0.1, sigma_max=2.0, m=2)
        assert not r.converged

    def test_silent_when_converged(self):
        # pyproject.toml turns an unexpected ConvergenceWarning into an error.
        r = _fit("exponentiation", sigma_min=0.1, sigma_max=2.0)
        assert r.converged
