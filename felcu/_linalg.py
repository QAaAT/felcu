"""Internal numerical linear-algebra helpers for felcu."""

from __future__ import annotations

import math

import numpy as np


def _trapz(y: np.ndarray, x: np.ndarray) -> float:
    if hasattr(np, "trapezoid"):
        return float(np.trapezoid(y, x))
    return float(np.trapz(y, x))  # noqa: NPY201  (NumPy < 2 fallback)


def _gauss_interval(a: float, b: float, n: int) -> tuple[np.ndarray, np.ndarray]:
    """Gauss-Legendre nodes and weights on [a, b], split into panels for large n."""
    n = max(int(n), 2)
    max_panel_n = 120
    panels = max(1, math.ceil(n / max_panel_n))
    n_panel = max(2, math.ceil(n / panels))
    z, w = np.polynomial.legendre.leggauss(n_panel)
    xs, ws = [], []
    for j in range(panels):
        left = a + (b - a) * j / panels
        right = a + (b - a) * (j + 1) / panels
        xs.append(0.5 * (right - left) * z + 0.5 * (right + left))
        ws.append(0.5 * (right - left) * w)
    return np.concatenate(xs), np.concatenate(ws)


def _penalty(m: int, basis: str, lam: float, r: float, w: float) -> np.ndarray:
    k = np.arange(1, m + 1, dtype=float)
    p = lam * (1.0 + r * k) ** w
    if basis == "odd_sine":
        return p
    if basis == "const_plus_sine":
        return np.concatenate([[lam], p])
    return np.concatenate([[lam], p, p])


def _alpha(a0: float, ak: np.ndarray, bk: np.ndarray) -> float:
    return float(abs(a0) + np.sum(np.sqrt(ak**2 + bk**2)))


def _validation_errors(
    c: np.ndarray,
    val: list[tuple[np.ndarray, np.ndarray, np.ndarray]],
) -> tuple[float, float]:
    integral = 0.0
    linf = 0.0
    for x, A, y in val:
        r = np.abs(A @ c - y)
        integral += _trapz(r**2, x)
        if r.size:
            linf = max(linf, float(np.max(r)))
    return math.sqrt(max(integral, 0.0)), linf


def _qr_solve(A: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Least-squares solve via LAPACK (numpy.linalg.lstsq)."""
    x, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
    return x
