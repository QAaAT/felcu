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
from felcu._problems import PROBLEMS
from felcu._solver import _summarise, _validation_data

HALF_PI = 0.5 * math.pi


# ---------------------------------------------------------------------------
# Reflected extension (Sec. III)
# ---------------------------------------------------------------------------


_GRADING_RATIO = 0.25  # each panel toward the singularity is this fraction of the last
_PANEL_MIN_NODES = 24


def _reflected_quadrature(spec: dict, m: int) -> tuple[np.ndarray, np.ndarray]:
    """Gauss nodes and weights on [-pi/2, pi/2] for the reflected coefficients.

    The pole or branch point of g at x = -tau mu may lie just outside the
    interval (at the largest allowed eta, on its end), where a uniform rule
    converges slowly.  Panels shrink geometrically toward the nearer end until
    they are no wider than the distance to the singularity, so every panel
    sees it at a fixed relative distance and converges at a fixed rate.  Each
    panel also gets nodes in proportion to its width to resolve the modes up
    to *m*.
    """
    n_oscillation = 8 * (m + 1) + 64
    x_sing = -spec["tau"] * spec["mu"]
    if PROBLEMS[spec["problem"]].entire or not math.isfinite(x_sing) or abs(x_sing) <= HALF_PI:
        # Generous fixed floor: an entire g may still grow steeply.
        return _gauss_interval(-HALF_PI, HALF_PI, max(n_oscillation, 600))

    # Breakpoints by distance u from the end nearest the singularity.
    distance = abs(x_sing) - HALF_PI
    u = [math.pi]
    while u[-1] > distance:
        u.append(u[-1] * _GRADING_RATIO)
    u.append(0.0)
    xs, ws = [], []
    for lo, hi in zip(u[1:], u[:-1], strict=True):
        n = _PANEL_MIN_NODES + math.ceil(n_oscillation * (hi - lo) / math.pi)
        x, w = _gauss_interval(lo, hi, n)
        xs.append(x)
        ws.append(w)
    end = math.copysign(HALF_PI, x_sing)
    return end - math.copysign(1.0, x_sing) * np.concatenate(xs), np.concatenate(ws)


def _reflected_coefficients(spec: dict, m: int) -> tuple[float, np.ndarray, np.ndarray]:
    """Fourier coefficients of the endpoint reflection F(x) = g(arcsin(sin x)).

    F reflects g about x = +-pi/2, so F(pi - x) = F(x): only the odd sine and
    even cosine modes survive, each twice its integral over [-pi/2, pi/2].
    """
    x, w = _reflected_quadrature(spec, m)
    wg = w * _spec_target(x, spec)
    a0 = float(np.sum(wg)) / math.pi
    ak, bk = np.zeros(m), np.zeros(m)
    odd, even = np.arange(1, m + 1, 2), np.arange(2, m + 1, 2)
    ak[odd - 1] = (2.0 / math.pi) * (np.sin(np.outer(odd, x)) @ wg)
    bk[even - 1] = (2.0 / math.pi) * (np.cos(np.outer(even, x)) @ wg)
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


def _taylor_coefficients(spec: dict, m: int) -> np.ndarray:
    """d_0, ..., d_m: the Taylor coefficients of g(arcsin z)."""
    u = _arcsin_series(m) / spec["tau"]
    u[0] = spec["mu"]
    return PROBLEMS[spec["problem"]].series(u)


def _arcsine_taylor_coefficients(spec: dict, m: int) -> tuple[float, np.ndarray, np.ndarray]:
    """Fourier coefficients of g_m(x) = sum_{k<=m} d_k sin^k x."""
    d = _taylor_coefficients(spec, m)
    # g_m is a trigonometric polynomial of degree m, so 2m + 2 equispaced
    # samples (Horner in sin x) give its coefficients exactly: the same c_k
    # as the binomial expansion of each sin^k x, since every alias c_{k+-n}
    # has |k +- n| > m and vanishes (see docs/guide/method.md).
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


def _explicit_candidate(spec: dict, m: int, val_n: int) -> dict:
    """The closed-form series of order *m* for ``spec['method']``, in the format of ``_candidate_for_m``."""
    a0, ak, bk = _COEFFICIENTS[spec["method"]](spec, m)
    # Packing drops the modes the basis excludes, which vanish by symmetry.
    coeffs = _pack_coefficients(a0, ak, bk, spec["basis"])
    solve = _summarise(coeffs, _validation_data(spec, m, val_n), spec["basis"], m)
    return {"m": m, "lambda": None, "solve": solve}
