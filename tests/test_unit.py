"""Unit tests: one helper or internal component in isolation.

Most of these never run the solver; the linear-solver test checks one
internal step against an independent reference.
"""

import numpy as np
import pytest

from felcu import canonical_method_name, default_eta, default_sobolev_r
from felcu.utils import (
    canonical_error_metric,
    canonical_problem_name,
    default_lambda_grid,
    default_m_grid,
    evaluate_fourier_series,
    to_complex_coefficients,
)

pytestmark = pytest.mark.unit

# ---------------------------------------------------------------------------
# canonical_problem_name
# ---------------------------------------------------------------------------


class TestCanonicalProblemName:
    def test_aliases_resolve(self):
        assert canonical_problem_name("action") == "action"
        assert canonical_problem_name("identity") == "action"
        assert canonical_problem_name("matrix-action") == "action"
        assert canonical_problem_name("inversion") == "inversion"
        assert canonical_problem_name("inverse") == "inversion"
        assert canonical_problem_name("exp") == "exponentiation"
        assert canonical_problem_name("exponentiation") == "exponentiation"
        assert canonical_problem_name("exponential") == "exponentiation"
        assert canonical_problem_name("sqrt") == "square_root"
        assert canonical_problem_name("square-root") == "square_root"

    def test_case_and_whitespace_insensitive(self):
        assert canonical_problem_name("  ACTION  ") == "action"
        assert canonical_problem_name("Inversion") == "inversion"

    def test_unknown_raises(self):
        with pytest.raises(ValueError, match="problem must be"):
            canonical_problem_name("eigenvalue")


# ---------------------------------------------------------------------------
# canonical_error_metric
# ---------------------------------------------------------------------------


class TestCanonicalErrorMetric:
    def test_aliases_resolve(self):
        assert canonical_error_metric("l2") == "l2"
        assert canonical_error_metric("2") == "l2"
        assert canonical_error_metric("linf") == "linf"
        assert canonical_error_metric("max") == "linf"
        assert canonical_error_metric("infinity") == "linf"

    def test_unknown_raises(self):
        with pytest.raises(ValueError, match="error_metric must be"):
            canonical_error_metric("l1")


# ---------------------------------------------------------------------------
# default_m_grid
# ---------------------------------------------------------------------------


class TestDefaultMGrid:
    def test_returns_integers(self):
        grid = default_m_grid()
        assert grid.dtype == int

    def test_bounds_respected(self):
        grid = default_m_grid(m_min=5, m_max=50)
        assert grid[0] >= 5
        assert grid[-1] <= 50

    def test_single_count_returns_max(self):
        grid = default_m_grid(m_min=4, m_max=100, count=1)
        assert list(grid) == [100]

    def test_invalid_args_raise(self):
        with pytest.raises(ValueError):
            default_m_grid(m_min=-1)
        with pytest.raises(ValueError):
            default_m_grid(m_min=50, m_max=10)


# ---------------------------------------------------------------------------
# default_lambda_grid
# ---------------------------------------------------------------------------


class TestDefaultLambdaGrid:
    def test_decreasing(self):
        grid = default_lambda_grid()
        assert np.all(np.diff(grid) < 0)

    def test_bounds(self):
        grid = default_lambda_grid(lambda_max=1.0, lambda_min=1e-6)
        assert pytest.approx(grid[0]) == 1.0
        assert pytest.approx(grid[-1]) == 1e-6

    def test_invalid_raises(self):
        with pytest.raises(ValueError):
            default_lambda_grid(lambda_min=0.0)
        with pytest.raises(ValueError):
            default_lambda_grid(count=1)


# ---------------------------------------------------------------------------
# evaluate_fourier_series
# ---------------------------------------------------------------------------


class TestEvaluateFourierSeries:
    def test_constant(self):
        x = np.linspace(-np.pi, np.pi, 100)
        y = evaluate_fourier_series(x, a0=3.0, ak=[], bk=[])
        np.testing.assert_allclose(y, 3.0)

    def test_sine_term(self):
        x = np.array([0.0, np.pi / 2, np.pi])
        # g(x) = sin(x)  →  a0=0, ak=[1], bk=[0]
        y = evaluate_fourier_series(x, a0=0.0, ak=[1.0], bk=[0.0])
        np.testing.assert_allclose(y, np.sin(x), atol=1e-14)

    def test_cosine_term(self):
        x = np.array([0.0, np.pi / 2, np.pi])
        # g(x) = cos(x)  →  a0=0, ak=[0], bk=[1]
        y = evaluate_fourier_series(x, a0=0.0, ak=[0.0], bk=[1.0])
        np.testing.assert_allclose(y, np.cos(x), atol=1e-14)

    def test_shape_preserved(self):
        x = np.zeros((3, 4))
        y = evaluate_fourier_series(x, 1.0, [0.5], [0.5])
        assert y.shape == (3, 4)

    def test_mismatched_ak_bk_raises(self):
        with pytest.raises(ValueError):
            evaluate_fourier_series(np.array([0.0]), 0.0, [1.0, 2.0], [1.0])


# ---------------------------------------------------------------------------
# to_complex_coefficients
# ---------------------------------------------------------------------------


class TestToComplexCoefficients:
    def test_constant_only(self):
        c = to_complex_coefficients(2.0, ak=[], bk=[])
        assert c.shape == (1,)
        assert c[0] == pytest.approx(2.0 + 0j)

    def test_symmetry(self):
        # c_{-k} = conj(c_k) for a real series
        ak = np.array([1.0, 0.5])
        bk = np.array([0.3, 0.7])
        c = to_complex_coefficients(0.0, ak=ak, bk=bk)
        m = len(ak)
        # c has length 2m+1; c[m] = c0, c[m+k] = c_k, c[m-k] = c_{-k}
        for k in range(1, m + 1):
            np.testing.assert_allclose(c[m - k], np.conj(c[m + k]), atol=1e-15)

    def test_missing_bk_raises(self):
        with pytest.raises(ValueError):
            to_complex_coefficients(0.0, ak=[1.0])


# ---------------------------------------------------------------------------
# canonical_method_name
# ---------------------------------------------------------------------------


class TestCanonicalMethodName:
    def test_method_aliases(self):
        assert canonical_method_name(" Regularized ") == "sobolev"
        assert canonical_method_name("refl") == "reflected"
        assert canonical_method_name("arcsine-taylor") == "arcsine_taylor"
        with pytest.raises(ValueError, match="method must be"):
            canonical_method_name("chebyshev")


# ---------------------------------------------------------------------------
# Target function
# ---------------------------------------------------------------------------


class TestTargetFunction:
    @pytest.mark.parametrize("problem", ["square_root"])
    def test_target_derivatives(self, problem):
        from felcu._coefficients import _target

        x, h = np.linspace(-0.9, 0.9, 7), 1e-5
        for d in (1, 2):
            fd = (_target(x + h, problem, 1.3, 2.0, d - 1) - _target(x - h, problem, 1.3, 2.0, d - 1)) / (2 * h)
            np.testing.assert_allclose(_target(x, problem, 1.3, 2.0, d), fd, rtol=1e-6)


# ---------------------------------------------------------------------------
# default_eta
# ---------------------------------------------------------------------------


class TestDefaultEta:
    def test_reflected(self):
        assert default_eta("exponentiation", method="reflected") == 2.0
        assert default_eta("inversion", method="refl", sigma_min=0.1, sigma_max=1.0) == 2.0

    def test_arcsine_taylor(self):
        assert default_eta("action", method="arcsine_taylor") == 3.0
        # Midpoint of 2 < eta < 2 mu / delta, capped at 3.
        assert default_eta("inversion", method="asin", sigma_min=0.1, sigma_max=1.0) == pytest.approx(1 + 1.1 / 0.9)
        assert default_eta("square_root", method="asin", sigma_min=0.5, sigma_max=1.0) == 3.0
        assert default_eta("inversion", method="asin", sigma_min=2.0, sigma_max=2.0) == 3.0

    def test_sobolev_unchanged(self):
        assert default_eta("inversion", sigma_min=0.1, sigma_max=1.0) == pytest.approx(1.1 / 0.9)

    def test_two_interval_inversion(self):
        # The gap and the free part of the period are equally wide: 1 + (b - a) / (2 delta).
        sv = dict(sigma_min=0.1, sigma_max=1.0, spectral_kind="singular_values")
        assert default_eta("inversion", **sv) == pytest.approx(1 + 1 / 10)
        assert default_eta("inversion", sigma_min=-1.0, sigma_max=1.0, gap=(-0.1, 0.1)) == pytest.approx(1.1)
        assert default_eta("inversion", sigma_min=-2.0, sigma_max=3.0, gap=(-0.5, 0.4)) == pytest.approx(1 + 0.9 / 5)

    def test_gap_read_from_matrix(self):
        H = np.diag([-2.0, -0.5, 0.4, 3.0])
        assert default_eta("inversion", matrix=H) == default_eta(
            "inversion", sigma_min=-2.0, sigma_max=3.0, gap=(-0.5, 0.4)
        )

    @pytest.mark.parametrize("problem", ["square_root"])
    def test_default_eta_puts_branch_point_on_boundary(self, problem):
        kappa = 1.0 / 0.15
        assert default_eta(problem, sigma_min=0.15, sigma_max=1.0) == pytest.approx((kappa + 1) / (kappa - 1))


# ---------------------------------------------------------------------------
# default_sobolev_r
# ---------------------------------------------------------------------------


class TestDefaultSobolevR:
    @pytest.mark.parametrize(
        "problem, kwargs",
        [
            ("action", {}),
            ("action", dict(sigma_min=-1.0, sigma_max=2.0)),
            ("inversion", dict(sigma_min=0.5, sigma_max=3.0)),
            ("inversion", dict(sigma_min=-2.0, sigma_max=3.0, gap=(-0.5, 0.4))),
            ("exponentiation", dict(sigma_min=1.0, sigma_max=3.0)),
        ],
    )
    def test_default_r_in_range(self, problem, kwargs):
        assert 0 < default_sobolev_r(problem, **kwargs) <= 1

    def test_default_r_exponentiation(self):
        # r_exp = 1 / (1 + eta delta / pi), with delta = 1
        kwargs = dict(sigma_min=1.0, sigma_max=3.0)
        assert default_sobolev_r("exponentiation", **kwargs) == pytest.approx(1 / (1 + 2.0 / np.pi))
        assert default_sobolev_r("exponentiation", eta=4.0, **kwargs) == pytest.approx(1 / (1 + 4.0 / np.pi))

    def test_default_r_inversion(self):
        # At the default eta, r_def = min(1, 2 pi / (kappa + 1)) and r_indef = min(1, pi / (kappa + 1)).
        sv = dict(spectral_kind="singular_values")
        assert default_sobolev_r("inversion", sigma_min=1.0, sigma_max=20.0) == pytest.approx(2 * np.pi / 21)
        assert default_sobolev_r("inversion", sigma_min=0.05, sigma_max=1.0, **sv) == pytest.approx(np.pi / 21)
        assert default_sobolev_r("inversion", sigma_min=0.5, sigma_max=1.0, **sv) == 1.0
        assert default_sobolev_r("inversion", sigma_min=-1.0, sigma_max=1.0, gap=(-0.05, 0.05)) == pytest.approx(
            np.pi / 21
        )

    def test_default_r_inversion_is_the_distance_to_the_pole(self):
        # min(1, tau * (distance from the spectrum to 0)), for any eta.
        kwargs = dict(sigma_min=-2.0, sigma_max=3.0, gap=(-0.5, 0.4), eta=1.5)
        tau = np.pi / (1.5 * 2.5)
        assert default_sobolev_r("inversion", **kwargs) == pytest.approx(0.4 * tau)
        assert default_sobolev_r("inversion", sigma_min=1.0, sigma_max=20.0, eta=4.0) == pytest.approx(
            np.pi / (4.0 * 9.5)
        )

    def test_default_r(self):
        kwargs = dict(sigma_min=0.04, sigma_max=1.0)  # kappa = 25
        assert default_sobolev_r("square_root", **kwargs) == pytest.approx(25**-0.5)
        assert default_sobolev_r("square_root", sobolev_w=0.75, **kwargs) == pytest.approx(25 ** (-1 / 3))


# ---------------------------------------------------------------------------
# Linear solver
# ---------------------------------------------------------------------------


class TestLinearSolver:
    @pytest.mark.parametrize("svd_fails", [False, True])
    @pytest.mark.parametrize("lam", [1e-6, 1e-3, 1e-1])
    def test_matches_augmented_lstsq(self, lam, svd_fails, monkeypatch):
        from felcu._linalg import _penalty
        from felcu._solver import _fit_system, _make_fit_data, _make_linear_solver
        from felcu.spectral import _make_spec, _spectral_data

        if svd_fails:
            # Whether LAPACK's SVD converges depends on the BLAS build and CPU,
            # so force the failure to test the QR fallback everywhere.
            def svd(*args, **kwargs):
                raise np.linalg.LinAlgError("SVD did not converge")

            monkeypatch.setattr(np.linalg, "svd", svd)

        spectral = _spectral_data("exponentiation", None, 0.1, 2.0, None, "auto", 1e-12)
        spec = _make_spec("exponentiation", spectral, None)
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
