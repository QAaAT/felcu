"""Tests for saving and loading FourierExtensionResult as JSON."""

import dataclasses
import json

import numpy as np
import pytest

from felcu import FourierExtensionResult, solve_fe_coefficients

pytestmark = pytest.mark.interface


def _assert_same(a, b):
    for field in dataclasses.fields(FourierExtensionResult):
        if field.name == "all_results":
            continue
        x, y = getattr(a, field.name), getattr(b, field.name)
        if isinstance(x, np.ndarray):
            assert x.dtype == y.dtype, field.name
            np.testing.assert_array_equal(x, y, err_msg=field.name)
        else:
            assert x == y or (x != x and y != y), field.name


@pytest.fixture(scope="module")
def result():
    return solve_fe_coefficients("exponentiation", sigma_min=0.1, sigma_max=2.0, target_error=1e-8, m_values=[8, 16])


@pytest.mark.parametrize(
    "problem, kwargs",
    [
        ("action", {}),
        ("inversion", dict(sigma_min=0.5, sigma_max=2.0)),
        ("exponentiation", dict(shifted=True, sigma_min=1.0, sigma_max=3.0)),
        ("square_root", dict(shifted=True, sigma_min=0.2, sigma_max=1.0)),
    ],
)
def test_round_trip_is_exact(tmp_path, problem, kwargs):
    original = solve_fe_coefficients(problem, target_error=1e-6, m_values=[16, 24], **kwargs)
    path = tmp_path / "result.json"
    original.to_json(path)
    _assert_same(original, FourierExtensionResult.from_json(path))


def test_matrix_metadata_round_trip():
    original = solve_fe_coefficients("exponentiation", matrix=np.diag([0.5, 1.0, 1.5]), target_error=1e-4, m=12)
    loaded = FourierExtensionResult.from_json(original.to_json())
    assert loaded.matrix_shape == (3, 3)
    assert loaded.matrix_is_hermitian is True
    _assert_same(original, loaded)


def test_infinite_tau_is_standard_json():
    original = solve_fe_coefficients("exponentiation", shifted=True, sigma_min=1.0, sigma_max=1.0)
    assert original.tau == np.inf
    text = original.to_json()

    def reject(constant):
        raise AssertionError(f"non-standard JSON constant {constant}")

    json.loads(text, parse_constant=reject)
    assert FourierExtensionResult.from_json(text).tau == np.inf


def test_layout(result):
    data = json.loads(result.to_json())
    assert data["format"] == "felcu.FourierExtensionResult"
    assert data["format_version"] == 1
    assert data["method"] == "sobolev"
    assert data["problem"] == "exponentiation"
    assert len(data["ak"]) == len(data["bk"]) == result.m
    c = data["complex_coefficients"]
    assert c["k"] == list(range(-result.m, result.m + 1))
    np.testing.assert_array_equal(np.array(c["real"]) + 1j * np.array(c["imag"]), result.complex_coefficients)
    assert "all_results" not in data


def test_candidates_optional(result):
    loaded = FourierExtensionResult.from_json(result.to_json(include_candidates=True))
    assert [c.m for c in loaded.all_results] == [c.m for c in result.all_results]
    for a, b in zip(loaded.all_results, result.all_results):
        _assert_same(a, b)
    assert FourierExtensionResult.from_json(result.to_json()).all_results == ()


def test_dict_round_trip(result):
    _assert_same(result, FourierExtensionResult.from_dict(result.to_dict()))


def test_rejects_other_json():
    with pytest.raises(ValueError, match="Not a saved felcu result"):
        FourierExtensionResult.from_json('{"a": 1}')


def test_rejects_future_version(result):
    data = result.to_dict()
    data["format_version"] = 99
    with pytest.raises(ValueError, match="format_version"):
        FourierExtensionResult.from_dict(data)


@pytest.mark.parametrize("method", ["reflected", "arcsine_taylor"])
def test_explicit_method_round_trip(method):
    original = solve_fe_coefficients(
        "exponentiation", method=method, shifted=True, sigma_min=0.1, sigma_max=2.0, m_values=[8, 16]
    )
    loaded = FourierExtensionResult.from_json(original.to_json(include_candidates=True))
    _assert_same(original, loaded)
    for a, b in zip(loaded.all_results, original.all_results):
        _assert_same(a, b)


def test_rejects_missing_field(result):
    data = result.to_dict()
    del data["tau"]
    with pytest.raises(ValueError, match="tau"):
        FourierExtensionResult.from_dict(data)


@pytest.mark.parametrize("method", ["reflected", "arcsine_taylor"])
def test_json_round_trip(method):
    r = solve_fe_coefficients("inversion", method=method, shifted=True, sigma_min=0.5, sigma_max=1.0, m=12)
    loaded = FourierExtensionResult.from_json(r.to_json())
    assert loaded.method == method
    assert loaded.lambda_used is None and loaded.sobolev_r is None
    np.testing.assert_array_equal(loaded.ak, r.ak)
