"""Convergence tests: the output is mathematically correct.

Each method meets its targets, and agrees with the paper: closed-form
coefficients, convergence rates and subnormalisation bounds, and f(H)
rebuilt from the series on real matrices.
"""

import math

import numpy as np
import pytest
from helpers import max_error

from felcu import evaluate_fourier_series, solve_fe_coefficients, to_complex_coefficients

pytestmark = pytest.mark.convergence

# (problem, kwargs, ||f(H)||) for convex endpoint-maximising f, where reflection is norm-saturating.
SATURATING = [
    ("action", dict(sigma_min=-1.0, sigma_max=3.0), 3.0),
    ("action", dict(sigma_min=-2.0, sigma_max=1.0), 2.0),
    ("exponentiation", dict(sigma_min=0.1, sigma_max=2.0), math.exp(2.0)),
    ("exponentiation", dict(sigma_min=-0.5, sigma_max=1.5), math.exp(1.5)),
    ("inversion", dict(sigma_min=0.5, sigma_max=1.0), 2.0),
    ("inversion", dict(sigma_min=-1.0, sigma_max=-0.2), 5.0),
]


# ---------------------------------------------------------------------------
# Sobolev-regularised least squares
# ---------------------------------------------------------------------------


class TestSobolev:
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


def _apply(r, H):
    """g(tau (H - mu I)) for a Hermitian H, by eigendecomposition."""
    w, V = np.linalg.eigh(H)
    return (V * evaluate_fourier_series(r.tau * (w - r.mu), r.a0, r.ak, r.bk)) @ V.conj().T


class TestSobolevIndefiniteInversion:
    common = dict(target_error=1e-6, m_values=range(16, 129, 16))

    @pytest.mark.parametrize("gap", [(-0.5, 0.5), (-0.5, 0.25)])
    def test_converges(self, gap):
        result = solve_fe_coefficients("inversion", sigma_min=-2.0, sigma_max=3.0, gap=gap, **self.common)
        assert result.converged
        assert result.x_gap is not None
        assert max_error(result) <= 1e-6 * 1.01

    def test_hermitian_matrix(self):
        # The gap is read from the eigenvalues, and the fit inverts H directly.
        rng = np.random.default_rng(3)
        Q, _ = np.linalg.qr(rng.standard_normal((6, 6)))
        lam = np.array([-2.0, -1.1, -0.4, 0.3, 1.5, 3.0])
        H = Q @ np.diag(lam) @ Q.T
        r = solve_fe_coefficients("inversion", matrix=H, **self.common)
        assert r.gap == (pytest.approx(-0.4), pytest.approx(0.3))
        assert r.mu == pytest.approx(0.5)
        np.testing.assert_allclose(_apply(r, H), np.linalg.inv(H), atol=1e-5)

    @pytest.mark.parametrize(
        "lam",
        [
            [-1.0, 1.0],  # both sides a point
            [-1.0, 1.0, 2.0, 3.0],  # the negative side a point
            [-3.0, -2.0, -1.0, 1.0],  # the positive side a point
        ],
    )
    def test_point_sides(self, lam):
        # A gap that reaches an end of the spectrum leaves that side a single point, which must still be fitted.
        H = np.diag(lam)
        r = solve_fe_coefficients("inversion", matrix=H, **self.common)
        assert r.converged
        np.testing.assert_allclose(_apply(r, H), np.linalg.inv(H), atol=1e-5)

    @pytest.mark.parametrize("metric", ["linf", "l2"])
    def test_repeated_end_eigenvalue_is_a_point(self, metric):
        # Rounding can split a repeated -1 into a sliver, as here; it must be fitted as the point it is.
        H = np.diag([-1.0, np.nextafter(np.nextafter(-1.0, 0.0), 0.0), 1.0, 2.0])
        r = solve_fe_coefficients("inversion", matrix=H, error_metric=metric, **self.common)
        assert r.x_gap[0] == -r.x_max
        np.testing.assert_allclose(_apply(r, H), np.linalg.inv(H), atol=1e-5)

    def test_equal_singular_values(self):
        # A scaled orthogonal matrix: both sides of the dilation's spectrum are points.
        A = 2.0 * np.array([[0.0, 1.0], [-1.0, 0.0]])
        r = solve_fe_coefficients("inversion", matrix=A, **self.common)
        assert r.converged
        D = np.block([[np.zeros((2, 2)), A], [A.T, np.zeros((2, 2))]])
        np.testing.assert_allclose(_apply(r, D)[2:, :2], np.linalg.inv(A), atol=1e-5)

    def test_general_matrix_through_its_dilation(self):
        # An odd g of the dilation [[0, A], [A^T, 0]] is [[0, g(A)-block], ...]; for 1/x it is the inverse.
        rng = np.random.default_rng(4)
        A = rng.standard_normal((4, 4)) + 2 * np.eye(4)
        r = solve_fe_coefficients("inversion", matrix=A, **self.common)
        assert r.spectral_kind_used == "singular_values"
        assert r.mu == 0.0
        D = np.block([[np.zeros((4, 4)), A], [A.T, np.zeros((4, 4))]])
        np.testing.assert_allclose(_apply(r, D)[4:, :4], np.linalg.inv(A), atol=1e-5)


class TestSobolevSquareRoot:
    common = dict(sigma_min=0.15, sigma_max=1.0, target_error=1e-6, m_values=range(8, 65, 8))

    @pytest.mark.parametrize("problem", ["square_root"])
    def test_converges(self, problem):
        result = solve_fe_coefficients(problem, **self.common)
        assert result.converged
        assert result.basis_used == "full"
        assert max_error(result, n=500) <= 1e-6 * 1.01

    @pytest.mark.parametrize("problem, f", [("sqrt", np.sqrt)])
    def test_matrix_function(self, problem, f):
        rng = np.random.default_rng(5)
        Q, _ = np.linalg.qr(rng.standard_normal((6, 6)))
        lam = np.linspace(0.3, 2.0, 6)
        H = Q @ np.diag(lam) @ Q.T
        r = solve_fe_coefficients(problem, matrix=H, target_error=1e-8, m_values=range(8, 65, 8))
        mu = r.mu
        g = evaluate_fourier_series(r.tau * (lam - mu), r.a0, r.ak, r.bk)
        np.testing.assert_allclose(Q @ np.diag(g) @ Q.T, Q @ np.diag(f(lam)) @ Q.T, atol=1e-7)


# ---------------------------------------------------------------------------
# Reflected extensions
# ---------------------------------------------------------------------------


class TestReflected:
    @pytest.mark.parametrize("problem, kwargs, norm", SATURATING)
    def test_saturates_the_norm_bound(self, problem, kwargs, norm):
        # Optimal subnormalisation: alpha_m = ||f(H)|| - epsilon_m, with the error attained at an endpoint.
        for m in (8, 33, 128):
            r = solve_fe_coefficients(problem, method="reflected", m=m, **kwargs)
            assert r.eta == 2.0
            assert r.alpha + r.achieved_linf_error == pytest.approx(norm, rel=1e-10)

    def test_algebraic_convergence(self):
        kw = dict(method="reflected", sigma_min=0.1, sigma_max=2.0)
        e1 = solve_fe_coefficients("exponentiation", m=32, **kw).achieved_linf_error
        e4 = solve_fe_coefficients("exponentiation", m=128, **kw).achieved_linf_error
        assert 0.2 < e4 / e1 < 0.3

    def test_identity_is_the_triangle_wave(self):
        # Sec. III A: a_k = (4 / (pi k^2)) sin(k pi / 2) / tau, with the constant mu exact.
        r = solve_fe_coefficients("action", method="reflected", sigma_min=1.0, sigma_max=4.0, m=9)
        k = np.arange(1, 10)
        np.testing.assert_allclose(r.ak, 4 / (np.pi * k**2) * np.sin(k * np.pi / 2) / r.tau, atol=1e-14)
        assert r.a0 == pytest.approx(2.5, abs=1e-14)
        assert r.basis_used == "const_plus_sine"

    def test_square_root_is_bounded(self):
        kw = dict(method="reflected", sigma_min=0.15, sigma_max=1.0)
        alphas = [solve_fe_coefficients("square_root", m=m, **kw).alpha for m in (16, 64, 256)]
        assert alphas == sorted(alphas)
        assert alphas[-1] < 1.1

    def test_larger_eta_moves_the_endpoint_inside(self):
        kw = dict(method="reflected", sigma_min=0.1, sigma_max=2.0, m=64)
        r2 = solve_fe_coefficients("exponentiation", **kw)
        r3 = solve_fe_coefficients("exponentiation", eta=3.0, **kw)
        assert r3.achieved_linf_error < r2.achieved_linf_error / 2
        assert r3.alpha > r2.alpha


# ---------------------------------------------------------------------------
# Arcsine-Taylor
# ---------------------------------------------------------------------------


class TestArcsineTaylor:
    def test_identity_coefficients_match_the_paper(self):
        # Eq. (arcsin_lcu_coefficients) with m = 2T + 1, scaled by 1/tau.
        T = 6
        r = solve_fe_coefficients("action", method="arcsine_taylor", sigma_min=-1.0, sigma_max=3.0, m=2 * T + 1)
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
            ("action", dict(sigma_min=-1.0, sigma_max=3.0), 3.0),
            ("exponentiation", dict(sigma_min=0.1, sigma_max=2.0), 3.0),
            ("inversion", dict(sigma_min=0.5, sigma_max=1.0), 2.5),
            ("square_root", dict(sigma_min=0.15, sigma_max=1.0), 2.7),
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
            "exponentiation", method="arcsine_taylor", sigma_min=0.1, sigma_max=2.0, target_error=1e-12
        )
        assert r.converged
        assert max_error(r) <= 1.01e-12

    def test_block_encoding_error(self):
        rng = np.random.default_rng(0)
        Q, _ = np.linalg.qr(rng.standard_normal((6, 6)))
        lam = np.array([0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
        H = Q @ np.diag(lam) @ Q.T
        r = solve_fe_coefficients("inversion", method="arcsine_taylor", matrix=H, m=60)
        mu = r.mu
        w, V = np.linalg.eigh(r.tau * (H - mu * np.eye(6)))
        k = np.arange(-r.m, r.m + 1)
        g_m = V @ np.diag(np.exp(1j * np.outer(w, k)) @ to_complex_coefficients(r)) @ V.T
        # Attained at the spectral endpoint, up to round-off in the matrix products.
        assert np.linalg.norm(g_m - np.linalg.inv(H), 2) <= r.achieved_linf_error + 1e-13
