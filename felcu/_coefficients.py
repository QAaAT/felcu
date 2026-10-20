"""Internal coefficient and basis-matrix helpers for felcu."""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np

# ---------------------------------------------------------------------------
# Basis utilities
# ---------------------------------------------------------------------------


def _basis_size(basis: str, m: int) -> int:
    if basis == "odd_sine":
        return m
    if basis == "const_plus_sine":
        return m + 1
    return 2 * m + 1


def _basis_matrix(x: np.ndarray, m: int, basis: str) -> np.ndarray:
    k = np.arange(1, m + 1, dtype=float)
    if basis == "odd_sine":
        return np.sin(np.outer(x, k))
    ones = np.ones((x.size, 1), dtype=float)
    if basis == "const_plus_sine":
        return np.hstack([ones, np.sin(np.outer(x, k))])
    return np.hstack([ones, np.sin(np.outer(x, k)), np.cos(np.outer(x, k))])


def _basis_derivative_matrix(x: np.ndarray, m: int, basis: str, d: int) -> np.ndarray:
    """Compute the d-th derivative of the basis matrix."""
    if d == 0:
        return _basis_matrix(x, m, basis)

    k = np.arange(1, m + 1, dtype=float)
    phase = d * np.pi / 2
    scale = k**d

    if basis == "odd_sine":
        return scale * np.sin(np.outer(x, k) + phase)

    zeros = np.zeros((x.size, 1), dtype=float)
    if basis == "const_plus_sine":
        return np.hstack([zeros, scale * np.sin(np.outer(x, k) + phase)])

    return np.hstack(
        [
            zeros,
            scale * np.sin(np.outer(x, k) + phase),
            scale * np.cos(np.outer(x, k) + phase),
        ]
    )


def _target_derivative(x: np.ndarray, spec: dict, d: int) -> np.ndarray:
    """Compute the analytical d-th derivative of the configured target function."""
    if d == 0:
        return spec["func"](x) - spec["baseline"]

    p = spec["problem"]
    tau = spec["tau"]
    spectral = spec["spectral"]

    if spectral is not None:
        lo = spectral["sigma_min"]
        hi = spectral["sigma_max"]
        mu = 0.5 * (lo + hi)

    # Unscaled cases
    if not spec["matrix_scale_provided"]:
        if p == "action":
            return np.ones_like(x) if d == 1 else np.zeros_like(x)
        if p == "exponentiation":
            return np.exp(x)
        if p == "inversion":
            return ((-1.0) ** d) * math.factorial(d) / (x ** (d + 1))
        raise ValueError(f"Unknown problem '{p}'")

    # Scaled / shifted cases
    if p == "action":
        if d == 1:
            return np.zeros_like(x) if not np.isfinite(tau) else np.ones_like(x) / tau
        return np.zeros_like(x)

    if p == "inversion":
        if spec["shifted"]:
            if not np.isfinite(tau):
                return np.zeros_like(x)
            denom = x + tau * mu
            return tau * ((-1.0) ** d) * math.factorial(d) / (denom ** (d + 1))
        return tau * ((-1.0) ** d) * math.factorial(d) / (x ** (d + 1))

    if p == "exponentiation":
        if spec["shifted"]:
            if not np.isfinite(tau):
                return np.zeros_like(x)
            return (1.0 / (tau**d)) * np.exp(mu + x / tau)
        if not np.isfinite(tau):
            return np.zeros_like(x)
        return (1.0 / (tau**d)) * np.exp(x / tau)

    raise ValueError(f"Unknown problem '{p}'")


# ---------------------------------------------------------------------------
# Coefficient packing/unpacking
# ---------------------------------------------------------------------------


def _pack_coefficients(a0: float, ak: Sequence, bk: Sequence, basis: str) -> np.ndarray:
    ak = np.asarray(ak, dtype=float).reshape(-1)
    bk = np.asarray(bk, dtype=float).reshape(-1)
    if ak.shape != bk.shape:
        raise ValueError("ak and bk must have the same length.")
    if basis == "odd_sine":
        return ak.copy()
    if basis == "const_plus_sine":
        return np.concatenate([[a0], ak])
    return np.concatenate([[a0], ak, bk])


def _add_baseline(c: np.ndarray, basis: str, baseline: float) -> np.ndarray:
    c = np.array(c, dtype=float, copy=True)
    if baseline != 0.0:
        if basis == "odd_sine":
            raise ValueError("A nonzero baseline requires a constant basis function.")
        c[0] += baseline
    return c


def _unpack_coefficients(c: np.ndarray, basis: str, m: int) -> tuple[float, np.ndarray, np.ndarray]:
    if basis == "odd_sine":
        return 0.0, c.copy(), np.zeros(m)
    if basis == "const_plus_sine":
        return float(c[0]), c[1:].copy(), np.zeros(m)
    return float(c[0]), c[1 : m + 1].copy(), c[m + 1 :].copy()
