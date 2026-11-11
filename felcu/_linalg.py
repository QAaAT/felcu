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


def _log_sobolev_weight(z: np.ndarray, w: float) -> np.ndarray:
    """log P_w(z), the log of the H^w Sobolev weight at ``z = r|k| >= 0``.

    ``P_w(z) = sum_{j=0}^{w} z^(2j)`` for integer w, and in general
    ``(1 - z^(2(w+1))) / (1 - z^2)``, which agrees with the sum for integer w,
    with ``P_w(0) = 1`` and ``P_w(1) = w + 1``.  With ``q = 2 ln z`` it is
    evaluated as ``log(expm1((w+1) q) / expm1(q))`` for ``z < 1`` and
    ``w q + log(expm1(-(w+1) q) / expm1(-q))`` for ``z > 1``, which stays
    accurate near ``z = 1`` and does not overflow for large ``z``.
    """
    z = np.asarray(z, dtype=float)
    out = np.zeros(z.shape)
    out[z == 1.0] = math.log(w + 1.0)
    below = (z > 0.0) & (z < 1.0)
    q = 2.0 * np.log(z[below])
    out[below] = np.log(np.expm1((w + 1.0) * q) / np.expm1(q))
    above = z > 1.0
    q = 2.0 * np.log(z[above])
    out[above] = w * q + np.log(np.expm1(-(w + 1.0) * q) / np.expm1(-q))
    return out


def _penalty(m: int, basis: str, lam: float, r: float, w: float) -> np.ndarray:
    """Diagonal D of the H_r^w penalty, so that ||D c||^2 = lam^2 ||h||_{H_r^w}^2 / (2 pi).

    In the complex basis the norm is 2 pi sum_k P_w(r|k|) |c_k|^2 (see
    :func:`_log_sobolev_weight`).  With c_0 = a0 and
    |c_k|^2 + |c_-k|^2 = (|a_k|^2 + |b_k|^2) / 2, which holds for complex
    coefficients too, each sine or cosine coefficient gets weight
    P_w(r k) / 2 and the constant weight P_w(0) = 1.
    """
    k = np.arange(1, m + 1, dtype=float)
    p = lam * np.exp(0.5 * (_log_sobolev_weight(r * k, w) - math.log(2.0)))
    return _assemble(basis, [lam], p, p)


def _alpha(a0: float, ak: np.ndarray, bk: np.ndarray) -> float:
    return float(abs(a0) + np.sum(np.sqrt(ak**2 + bk**2)))


def _validation_errors(
    c: np.ndarray,
    val: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]],
) -> tuple[float, float]:
    integral = 0.0
    linf = 0.0
    for _, A, y, w in val:
        r = np.abs(A @ c - y)
        integral += float(np.dot(w, r**2))
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
