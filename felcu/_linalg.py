"""Internal numerical linear-algebra helpers for felcu."""

from __future__ import annotations

import math

import numpy as np

from felcu._coefficients import _assemble


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
    """Diagonal D of the H_r^w penalty, so that ||D c||^2 = lam^2 ||h||_{H_r^w}^2 / (2 pi).

    In the complex basis the norm is 2 pi sum_k (1 + (r|k|)^(2w)) |c_k|^2.  With
    c_0 = a0 and |c_k|^2 + |c_-k|^2 = (|a_k|^2 + |b_k|^2) / 2, each real sine or
    cosine coefficient gets weight (1 + (r k)^(2w)) / 2 and the constant weight 1.
    """
    k = np.arange(1, m + 1, dtype=float)
    p = lam * np.sqrt(0.5 * (1.0 + (r * k) ** (2.0 * w)))
    return _assemble(basis, [lam], p, p)


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
        integral += float(np.trapezoid(r**2, x))
        if r.size:
            linf = max(linf, float(np.max(r)))
    return math.sqrt(max(integral, 0.0)), linf


def _lstsq(A: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Least-squares solve via LAPACK (numpy.linalg.lstsq)."""
    x, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
    return x


def _ridge_qr(A: np.ndarray, b: np.ndarray, lam: float) -> np.ndarray:
    """Solve ``min ||A y - b||^2 + lam^2 ||y||^2`` for ``lam > 0`` by QR.

    Householder QR of the stacked matrix ``[A; lam I]`` is not iterative, so
    unlike an SVD it cannot fail to converge, and every singular value of the
    stacked matrix is at least ``lam``, so its ``R`` factor is invertible.
    """
    n = A.shape[1]
    Q, R = np.linalg.qr(np.vstack([A, lam * np.eye(n)]))
    return np.linalg.solve(R, Q[: A.shape[0]].T @ b)
