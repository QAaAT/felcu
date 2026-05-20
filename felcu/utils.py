"""Public utility functions for felcu."""

from __future__ import annotations

from typing import Sequence

import numpy as np

from felcu.types import ArrayLike, FourierExtensionResult


def canonical_problem_name(name: str) -> str:
    """Normalise a problem name to one of 'action', 'inversion', 'exponentiation'."""
    aliases = {
        "action": "action",
        "identity": "action",
        "matrix-action": "action",
        "inversion": "inversion",
        "inverse": "inversion",
        "exp": "exponentiation",
        "exponentiation": "exponentiation",
        "exponential": "exponentiation",
    }
    key = name.strip().lower()
    if key not in aliases:
        raise ValueError("problem must be 'action', 'inversion', or 'exponentiation'.")
    return aliases[key]


def canonical_error_metric(name: str) -> str:
    """Normalise an error-metric name to 'l2' or 'linf'."""
    aliases = {
        "l2": "l2",
        "2": "l2",
        "linf": "linf",
        "l_inf": "linf",
        "l-infinity": "linf",
        "infinity": "linf",
        "inf": "linf",
        "max": "linf",
    }
    key = name.strip().lower()
    if key not in aliases:
        raise ValueError("error_metric must be 'l2' or 'linf'.")
    return aliases[key]


def default_m_grid(
    m_min: int = 4,
    m_max: int = 256,
    count: int = 32,
    power: float = 2.0,
) -> np.ndarray:
    """Return a grid of candidate Fourier truncation orders."""
    if m_min < 0 or m_max < m_min or count <= 0:
        raise ValueError("Require 0 <= m_min <= m_max and count > 0.")
    if count == 1:
        return np.array([m_max], dtype=int)
    grid = m_min + (m_max - m_min) * np.linspace(0.0, 1.0, count) ** power
    return np.unique(grid.astype(int))


def default_lambda_grid(
    lambda_max: float = 1.0,
    lambda_min: float = 1e-16,
    count: int = 12,
) -> np.ndarray:
    """Return a log-spaced grid of regularisation parameters."""
    if lambda_min <= 0 or lambda_max <= 0 or lambda_min > lambda_max or count < 2:
        raise ValueError("Require 0 < lambda_min <= lambda_max and count >= 2.")
    return np.geomspace(lambda_max, lambda_min, count)


def evaluate_fourier_series(
    x: ArrayLike,
    a0: float | complex,
    ak: Sequence,
    bk: Sequence,
) -> np.ndarray:
    """Evaluate the real Fourier series g(x) = a0 + sum_k ak sin(kx) + bk cos(kx)."""
    x = np.asarray(x)
    x_flat = x.reshape(-1)
    ak = np.asarray(ak).reshape(-1)
    bk = np.asarray(bk).reshape(-1)
    if ak.shape != bk.shape:
        raise ValueError("ak and bk must have the same length.")

    dtype = np.result_type(x, a0, ak, bk, float)
    y = np.full(x_flat.shape, a0, dtype=dtype)
    if ak.size:
        k = np.arange(1, ak.size + 1, dtype=float)
        angles = np.outer(x_flat.astype(float), k)
        y = y + np.sin(angles) @ ak + np.cos(angles) @ bk
    return y.reshape(x.shape)


def to_complex_coefficients(
    a0_or_result: FourierExtensionResult | float | complex,
    ak: Sequence | None = None,
    bk: Sequence | None = None,
) -> np.ndarray:
    r"""Convert a0, ak, bk to [c_{-m}, ..., c_0, ..., c_m]."""
    if isinstance(a0_or_result, FourierExtensionResult):
        a0 = a0_or_result.a0
        ak = a0_or_result.ak
        bk = a0_or_result.bk
    elif ak is None or bk is None:
        raise ValueError("Provide both ak and bk, or pass a FourierExtensionResult.")
    else:
        a0 = a0_or_result

    ak = np.asarray(ak, dtype=np.complex128).reshape(-1)
    bk = np.asarray(bk, dtype=np.complex128).reshape(-1)
    if ak.shape != bk.shape:
        raise ValueError("ak and bk must have the same length.")

    c0 = np.array([complex(a0)], dtype=np.complex128)
    if ak.size == 0:
        return c0
    c_pos = 0.5 * (bk - 1j * ak)
    c_neg = 0.5 * (bk[::-1] + 1j * ak[::-1])
    return np.concatenate([c_neg, c0, c_pos])


def fitted_domain_segments(
    result: FourierExtensionResult,
    num: int = 2001,
) -> list[np.ndarray]:
    """Return x-arrays covering the fitted domain of *result*."""
    if num < 2:
        raise ValueError("num must be at least 2.")
    if result.domain == "single":
        return [np.linspace(-result.x_max, result.x_max, num)]
    if result.x_min is None:
        raise ValueError("A double-interval result requires x_min.")
    n = max(num // 2, 2)
    return [
        np.linspace(-result.x_max, -result.x_min, n),
        np.linspace(result.x_min, result.x_max, n),
    ]


def evaluate_target_function(
    x: ArrayLike,
    result: FourierExtensionResult,
) -> np.ndarray:
    """Evaluate the exact target function for *result* at points *x*."""
    x = np.asarray(x, dtype=float)
    p = result.problem

    if not result.matrix_scale_provided:
        if p == "action":
            return x
        if p == "exponentiation":
            return np.exp(x)
        if p == "inversion":
            return 1.0 / x
        raise ValueError(f"Unknown problem '{p}'.")

    tau = result.tau
    if result.shifted:
        if result.sigma_min is None or result.sigma_max is None:
            raise ValueError("Shifted results require sigma_min and sigma_max.")
        mu = 0.5 * (result.sigma_min + result.sigma_max)

    if p == "action":
        if result.shifted:
            return np.full_like(x, mu) if not np.isfinite(tau) else mu + x / tau
        return np.zeros_like(x) if not np.isfinite(tau) else x / tau

    if p == "inversion":
        if result.shifted:
            return np.full_like(x, 1.0 / mu) if not np.isfinite(tau) else tau / (x + tau * mu)
        return tau / x

    if p == "exponentiation":
        if result.shifted:
            return np.full_like(x, np.exp(mu)) if not np.isfinite(tau) else np.exp(mu + x / tau)
        return np.ones_like(x) if not np.isfinite(tau) else np.exp(x / tau)

    raise ValueError(f"Unknown problem '{p}'.")
