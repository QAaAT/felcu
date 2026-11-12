"""Interface tests for FourierExtensionResult.evaluate / apply and the ConvergenceWarning."""

import numpy as np
import pytest

from felcu import ConvergenceWarning, solve_fe_coefficients

pytestmark = pytest.mark.interface

# (problem, kwargs, f) fits whose spectral interval is [sigma_min, sigma_max].
FITS = [
    ("exponentiation", dict(sigma_min=0.1, sigma_max=2.0), np.exp),
    ("action", dict(sigma_min=-1.0, sigma_max=3.0), lambda x: x),
    ("inversion", dict(sigma_min=0.5, sigma_max=2.0), lambda x: 1.0 / x),
    ("square_root", dict(sigma_min=0.2, sigma_max=1.0), np.sqrt),
]


def _fit(problem, m=16, **kwargs):
    """A quick fit: fixed m, moderate target."""
    return solve_fe_coefficients(problem, target_error=1e-6, m=m, **kwargs)


def _random_hermitian(eigenvalues, seed=0, complex_=False):
    rng = np.random.default_rng(seed)
    n = len(eigenvalues)
    X = rng.standard_normal((n, n))
    if complex_:
        X = X + 1j * rng.standard_normal((n, n))
    Q, _ = np.linalg.qr(X)
    return (Q * eigenvalues) @ Q.conj().T


def _dilation(A):
    n, k = A.shape
    return np.block([[np.zeros((n, n)), A], [A.conj().T, np.zeros((k, k))]])


class TestEvaluate:
    @pytest.mark.parametrize("problem, kwargs, f", FITS)
    def test_matches_f_on_the_interval(self, problem, kwargs, f):
        r = _fit(problem, **kwargs)
        x = np.linspace(kwargs["sigma_min"], kwargs["sigma_max"], 201)
        np.testing.assert_allclose(r.evaluate(x), f(x), atol=1e-5)

    def test_indefinite_inversion_covers_both_sides_of_the_gap(self):
        r = _fit("inversion", m=48, sigma_min=-2.0, sigma_max=2.0, gap=(-0.5, 0.5))
        x = np.concatenate([np.linspace(-2.0, -0.5, 50), np.linspace(0.5, 2.0, 50)])
        np.testing.assert_allclose(r.evaluate(x), 1.0 / x, atol=1e-5)

    def test_keeps_the_input_shape(self):
        r = _fit("exponentiation", sigma_min=0.1, sigma_max=2.0)
        assert r.evaluate(np.ones((3, 4))).shape == (3, 4)
        assert r.evaluate(1.0).shape == ()

    def test_zero_width_interval(self):
        r = solve_fe_coefficients("exponentiation", sigma_min=1.5, sigma_max=1.5)
        np.testing.assert_allclose(r.evaluate([1.5]), [np.exp(1.5)])


class TestApply:
    @pytest.mark.parametrize("complex_", [False, True])
    def test_matches_the_exact_matrix_function(self, complex_):
        w = np.linspace(0.1, 2.0, 8)
        H = _random_hermitian(w, complex_=complex_)
        r = _fit("exponentiation", matrix=H)
        exact = _random_hermitian(np.exp(w), complex_=complex_)
        np.testing.assert_allclose(r.apply(H), exact, atol=1e-5)

    def test_inversion(self):
        H = _random_hermitian(np.linspace(0.5, 2.0, 6), seed=1)
        r = _fit("inversion", matrix=H)
        np.testing.assert_allclose(r.apply(H), np.linalg.inv(H), atol=1e-5)

    def test_singular_value_fit_applies_to_the_dilation(self):
        A = np.random.default_rng(2).standard_normal((5, 5))
        r = _fit("action", m=24, matrix=A)
        assert r.spectral_kind_used == "singular_values"
        D = _dilation(A)
        np.testing.assert_allclose(r.apply(D), D, atol=1e-5)

    def test_rejects_non_hermitian(self):
        r = _fit("exponentiation", sigma_min=0.1, sigma_max=2.0)
        with pytest.raises(ValueError, match="Hermitian"):
            r.apply(np.array([[1.0, 1.0], [0.0, 1.0]]))

    def test_rejects_non_square(self):
        r = _fit("exponentiation", sigma_min=0.1, sigma_max=2.0)
        with pytest.raises(ValueError, match="square"):
            r.apply(np.ones((2, 3)))

    def test_rejects_eigenvalues_beyond_the_interval(self):
        r = _fit("exponentiation", sigma_min=0.1, sigma_max=2.0)
        with pytest.raises(ValueError, match="outside the fitted spectrum"):
            r.apply(np.diag([0.5, 2.5]))

    def test_rejects_eigenvalues_in_the_gap(self):
        r = _fit("inversion", m=48, sigma_min=-2.0, sigma_max=2.0, gap=(-0.5, 0.5))
        with pytest.raises(ValueError, match="outside the fitted spectrum"):
            r.apply(np.diag([-1.0, 0.1, 1.0]))

    def test_accepts_eigenvalues_at_the_ends_up_to_rounding(self):
        r = _fit("exponentiation", sigma_min=0.1, sigma_max=2.0)
        H = np.diag([0.1 * (1 - 1e-12), 2.0 * (1 + 1e-12)])
        np.testing.assert_allclose(np.diag(r.apply(H)), np.exp([0.1, 2.0]), atol=1e-5)

    def test_zero_width_interval(self):
        r = solve_fe_coefficients("exponentiation", sigma_min=1.5, sigma_max=1.5)
        np.testing.assert_allclose(r.apply(np.diag([1.5, 1.5])), np.exp(1.5) * np.eye(2))
        with pytest.raises(ValueError, match="outside the fitted spectrum"):
            r.apply(np.diag([1.5, 1.6]))

    @pytest.mark.parametrize("problem", ["inversion", "logarithm", "square_root"])
    def test_zero_width_interval_near_zero_rejects_zero(self, problem):
        # The slack is relative to mu, so a tiny mu does not admit the
        # eigenvalue 0, where inversion and the logarithm are singular.
        r = solve_fe_coefficients(problem, sigma_min=1e-10, sigma_max=1e-10)
        with pytest.raises(ValueError, match="outside the fitted spectrum"):
            r.apply([[0.0]])
        np.testing.assert_allclose(r.apply([[1e-10 * (1 + 1e-12)]]), [[r.a0]])

    @pytest.mark.parametrize("problem", ["identity", "inversion", "exponentiation", "square_root", "logarithm"])
    def test_accepts_the_float32_matrix_it_was_fitted_to(self, problem):
        # The fit computes eigenvalues in double precision; apply must too, or
        # float32 rounding (~1e-7) pushes them outside the 1e-8 slack.
        H = np.array([[1, 0.1], [0.1, 2]], dtype=np.float32)
        r = _fit(problem, matrix=H)
        exact = _fit(problem, matrix=H.astype(np.float64)).apply(H.astype(np.float64))
        np.testing.assert_allclose(r.apply(H), exact)


class TestConvergenceWarning:
    def test_warns_when_the_target_is_missed(self):
        with pytest.warns(ConvergenceWarning, match="target_error=1e-12"):
            r = solve_fe_coefficients("exponentiation", sigma_min=0.1, sigma_max=2.0, m=2)
        assert not r.converged

    def test_silent_when_converged(self):
        # pyproject.toml turns an unexpected ConvergenceWarning into an error.
        r = _fit("exponentiation", sigma_min=0.1, sigma_max=2.0)
        assert r.converged
