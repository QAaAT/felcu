"""Explicit Fourier extensions: reflected extensions and arcsine-Taylor.

These are the coefficient-selection strategies of Secs. III and IV of the
paper.  Both prescribe the periodic extension of g outright, so every order m
has a closed-form series and there is no regularisation search.  Both
evaluate g on [-pi/2, pi/2]; the spec checks keep that interval clear of the
pole or branch point at 0.
"""

from __future__ import annotations

import math

import numpy as np

from felcu._coefficients import _pack_coefficients, _spec_target
from felcu._linalg import _gauss_interval
from felcu._solver import _meets_target, _summarise, _validation_data

HALF_PI = 0.5 * math.pi


# ---------------------------------------------------------------------------
# Reflected extension (Sec. III)
# ---------------------------------------------------------------------------


def _reflected_coefficients(spec: dict, m: int) -> tuple[float, np.ndarray, np.ndarray]:
    """Fourier coefficients of the endpoint reflection F(x) = g(arcsin(sin x)).

    F reflects g about x = +-pi/2, so F(pi - x) = F(x): only the odd sine and
    even cosine modes survive, each twice its integral over [-pi/2, pi/2].
    """
    # Generous fixed floor: g may be sharply peaked near a spectral endpoint.
    x, w = _gauss_interval(-HALF_PI, HALF_PI, max(8 * (m + 1) + 64, 600))
    wg = w * _spec_target(x, spec)
    k = np.arange(1, m + 1)
    kx = np.outer(k, x)
    a0 = float(np.sum(wg)) / math.pi
    ak = np.where(k % 2 == 1, (2.0 / math.pi) * (np.sin(kx) @ wg), 0.0)
    bk = np.where(k % 2 == 0, (2.0 / math.pi) * (np.cos(kx) @ wg), 0.0)
    return a0, ak, bk


# ---------------------------------------------------------------------------
# Arcsine-Taylor (Sec. IV)
# ---------------------------------------------------------------------------
#
# g_m(x) = sum_{k=0}^m d_k sin^k x, where d_k are the Taylor coefficients of
# g(arcsin z) = F(mu + arcsin(z) / tau).  They come from truncated power-series
# arithmetic on u(z) = mu + arcsin(z) / tau, which is exact up to z^m.


def _arcsin_series(n: int) -> np.ndarray:
    """Taylor coefficients of arcsin z up to z^n: binom(2j, j) / (4^j (2j + 1)) at z^(2j+1)."""
    s = np.zeros(n + 1)
    term = 1.0  # binom(2j, j) / 4^j
    for j in range((n + 1) // 2):
        s[2 * j + 1] = term / (2 * j + 1)
        term *= (2 * j + 1) / (2 * j + 2)
    return s


def _series_exp(u: np.ndarray) -> np.ndarray:
    """exp(u) as a power series, from y' = u' y."""
    y = np.zeros_like(u)
    y[0] = math.exp(u[0])
    ju = np.arange(u.size) * u
    for k in range(1, u.size):
        y[k] = (ju[1 : k + 1] @ y[k - 1 :: -1]) / k
    return y


def _series_reciprocal(u: np.ndarray) -> np.ndarray:
    """1 / u as a power series, from u y = 1."""
    y = np.zeros_like(u)
    y[0] = 1.0 / u[0]
    for k in range(1, u.size):
        y[k] = -(u[1 : k + 1] @ y[k - 1 :: -1]) / u[0]
    return y


def _series_sqrt(u: np.ndarray) -> np.ndarray:
    """sqrt(u) as a power series, from y^2 = u."""
    y = np.zeros_like(u)
    y[0] = math.sqrt(u[0])
    for k in range(1, u.size):
        y[k] = (u[k] - y[1:k] @ y[k - 1 : 0 : -1]) / (2.0 * y[0])
    return y


_COMPOSE = {
    "action": lambda u: u,
    "inversion": _series_reciprocal,
    "exponentiation": _series_exp,
    "square_root": _series_sqrt,
}


def _taylor_coefficients(spec: dict, m: int) -> np.ndarray:
    """d_0, ..., d_m: the Taylor coefficients of g(arcsin z)."""
    u = _arcsin_series(m) / spec["tau"]
    u[0] = spec["mu"]
    return _COMPOSE[spec["problem"]](u)


def _arcsine_taylor_coefficients(spec: dict, m: int) -> tuple[float, np.ndarray, np.ndarray]:
    """Fourier coefficients of g_m(x) = sum_{k<=m} d_k sin^k x."""
    d = _taylor_coefficients(spec, m)
    # g_m is a trigonometric polynomial of degree m, so 2m + 2 equispaced
    # samples (Horner in sin x) give its coefficients exactly.
    n = 2 * m + 2
    s = np.sin(2.0 * math.pi * np.arange(n) / n)
    y = np.zeros(n)
    for dk in d[::-1]:
        y = y * s + dk
    c = np.fft.rfft(y) / n  # c[k] is the complex coefficient c_k
    return float(c[0].real), -2.0 * c[1 : m + 1].imag, 2.0 * c[1 : m + 1].real


# ---------------------------------------------------------------------------
# Candidate for one m
# ---------------------------------------------------------------------------

_COEFFICIENTS = {
    "reflected": _reflected_coefficients,
    "arcsine_taylor": _arcsine_taylor_coefficients,
}


def _explicit_candidate(spec: dict, m: int, target: float, metric: str, val_n: int) -> dict:
    """The closed-form series of order *m* for ``spec['method']``, in the format of ``_candidate_for_m``."""
    a0, ak, bk = _COEFFICIENTS[spec["method"]](spec, m)
    # Packing drops the modes the basis excludes, which vanish by symmetry.
    coeffs = _pack_coefficients(a0, ak, bk, spec["basis"])
    solve = _summarise(coeffs, _validation_data(spec, m, val_n), spec["basis"], m)
    return {"m": m, "lambda": None, "solve": solve, "feasible": _meets_target(solve, metric, target)}
