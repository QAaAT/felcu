"""Order-aware selection: the score J_p, its order p, and the readout norm tau J_1."""

import json

import numpy as np
import pytest

from felcu import FourierExtensionResult, solve_fe_coefficients, to_complex_coefficients

EXP = dict(sigma_min=0.1, sigma_max=2.0)
# J_2 is not monotone in m here, so it and alpha pick different orders.
SQRT = dict(sigma_min=0.1, sigma_max=1.0, sobolev_w=2.0, target_error=1e-6, m_values=range(8, 57, 8))


def _j(result, p):
    """J_p straight from the definition, over the signed modes c_k."""
    c = to_complex_coefficients(result)
    k = np.arange(-result.m, result.m + 1)
    if p == 0:
        return np.sum(np.abs(c))
    return np.sum(np.abs(k[k != 0]) ** p * np.abs(c[k != 0]))


@pytest.mark.unit
class TestCoefficientNorm:
    @pytest.mark.parametrize("p", [0.0, 0.5, 1.0, 2.0])
    @pytest.mark.parametrize("dtype", [float, complex])
    def test_matches_the_definition(self, p, dtype):
        from felcu._linalg import _coefficient_norm

        rng = np.random.default_rng(1)
        ak, bk = rng.standard_normal((2, 7)).astype(dtype)
        if dtype is complex:
            ak, bk = ak + 1j * rng.standard_normal(7), bk - 0.5j * rng.standard_normal(7)
        a0 = dtype(0.3)
        c = to_complex_coefficients(a0, ak, bk)
        k = np.arange(-7, 8)
        expected = np.sum(np.abs(c)) if p == 0 else np.sum(np.abs(k[k != 0]) ** p * np.abs(c[k != 0]))
        assert _coefficient_norm(a0, ak, bk, p) == pytest.approx(expected, rel=1e-14)


@pytest.mark.interface
class TestSelectionScore:
    @pytest.mark.parametrize(
        "kwargs, p",
        [
            (dict(), 0.0),
            (dict(sobolev_w=0.75), 0.0),
            (dict(sobolev_w=2.5), 1.5),
            (dict(method="reflected"), 0.0),
            (dict(method="reflected", reflection_order=1), 1.0),
            (dict(method="arcsine_taylor"), 0.0),
        ],
    )
    def test_order_and_fields(self, kwargs, p):
        r = solve_fe_coefficients("exponentiation", m=16, **EXP, **kwargs)
        assert r.selection_order == p
        assert r.selection_score == pytest.approx(_j(r, p), rel=1e-12)
        assert r.alpha == pytest.approx(_j(r, 0), rel=1e-12)  # alpha is still J_0
        assert r.readout_norm == pytest.approx(r.tau * _j(r, 1), rel=1e-12)

    def test_order_zero_score_is_alpha(self):
        r = solve_fe_coefficients("exponentiation", m=16, **EXP)
        assert r.selection_score == r.alpha

    def test_constant_series(self):
        r = solve_fe_coefficients("exponentiation", sigma_min=0.7, sigma_max=0.7)
        assert r.m == 0 and r.readout_norm == 0.0 and r.selection_score == pytest.approx(np.exp(0.7))

    def test_json_round_trip(self):
        r = solve_fe_coefficients("exponentiation", sobolev_w=2.0, m=16, **EXP)
        data = json.loads(r.to_json())
        assert data["selection_order"] == 1.0
        loaded = FourierExtensionResult.from_json(r.to_json())
        assert (loaded.selection_score, loaded.readout_norm) == (r.selection_score, r.readout_norm)

    def test_files_without_the_score_load(self):
        data = solve_fe_coefficients("exponentiation", m=8, **EXP).to_dict()
        del data["selection_order"], data["selection_score"], data["readout_norm"]
        loaded = FourierExtensionResult.from_dict(data)
        assert loaded.selection_order is None and loaded.selection_score is None


@pytest.mark.interface
class TestSmallestNorm:
    def test_is_the_default_and_minimises_the_score(self):
        r = solve_fe_coefficients("square_root", **SQRT)
        feasible = [c for c in r.all_results if c.converged]
        assert r.selection_score == min(c.selection_score for c in feasible)
        assert r.m == 48

    def test_differs_from_smallest_alpha(self):
        r = solve_fe_coefficients("square_root", selection="smallest_alpha", **SQRT)
        feasible = [c for c in r.all_results if c.converged]
        assert r.alpha == min(c.alpha for c in feasible)
        assert r.m == 56

    def test_same_as_smallest_alpha_at_order_zero(self):
        common = dict(target_error=1e-8, m_values=range(8, 41, 4), **EXP)
        a = solve_fe_coefficients("exponentiation", **common)
        b = solve_fe_coefficients("exponentiation", selection="smallest_alpha", **common)
        assert a.m == b.m and a.alpha == b.alpha

    def test_reflected_derivative(self):
        r = solve_fe_coefficients(
            "exponentiation", method="reflected", reflection_order=1, target_error=1e-3, m_values=[48, 64, 96], **EXP
        )
        feasible = [c for c in r.all_results if c.converged]
        assert r.selection_score == min(c.selection_score for c in feasible)

    def test_rejects_unknown(self):
        with pytest.raises(ValueError, match="'smallest_norm', 'smallest_alpha' or 'smallest_m'"):
            solve_fe_coefficients("exponentiation", selection="largest", m=8, **EXP)
