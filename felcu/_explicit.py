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

from felcu._coefficients import _pack_coefficients, _scalar, _spec_target
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
    a0 = _scalar(np.sum(wg)) / math.pi
    ak = np.where(k % 2 == 1, (2.0 / math.pi) * (np.sin(kx) @ wg), 0.0)
    bk = np.where(k % 2 == 0, (2.0 / math.pi) * (np.cos(kx) @ wg), 0.0)
    return a0, ak, bk


# ---------------------------------------------------------------------------
# Quadratic continuation of a reflection (eta > 2)
# ---------------------------------------------------------------------------
#
# For eta > 2 the fitted interval [-pi/eta, pi/eta] is shorter than half a
# period.  Measure y from the endpoint x* where |g| is larger, into the
# interval (q = +1 at the left end, -1 at the right), and rotate by the phase
# zeta of g(x*): h(y) = conj(zeta) g(x* + q y) on [0, theta], theta = 2 pi/eta.
# On (theta, pi] the profile continues as the quadratic
#
#     p(s) = h(theta) - (3/2) [h(theta) + (int_0^theta h - pi beta_0) / l] (2s - s^2),
#
# with s = (y - theta) / l and l = pi - theta, which is continuous at theta,
# flat at pi, and gives the completed profile E the mean beta_0.  E is
# extended evenly and periodically, so g_m(x) = zeta E_m(q (x - x*)) with
#
#     c_0 = zeta beta_0,   c_{+-k} = zeta beta_k e^{-+ i k x*} / 2,
#     beta_k = (2/pi) int_0^pi E(y) cos(k y) dy,
#
# i.e. a_k = zeta beta_k sin(k x*) and b_k = zeta beta_k cos(k x*).  Every
# beta_k is affine in beta_0, the closure mean.


def _continuation_profile(spec: dict, m: int) -> dict:
    """The parts of the continuation that do not depend on the closure mean.

    Returns ``x_star``, ``zeta``, ``beta`` (beta_1..beta_m at beta_0 = 0) and
    ``dbeta`` (their derivative with respect to beta_0), and whether the
    profile is ``real``.
    """
    eta = spec["eta"]
    x_max = math.pi / eta
    theta = 2.0 * x_max
    ell = math.pi - theta
    g_left, g_right = (complex(v) for v in _spec_target(np.array([-x_max, x_max]), spec))
    # The endpoint with the larger |g|; q = +1 at the left end, -1 at the right.
    if abs(g_left) >= abs(g_right):
        x_star, q, g_star, g_far = -x_max, 1.0, g_left, g_right
    else:
        x_star, q, g_star, g_far = x_max, -1.0, g_right, g_left
    zeta = g_star / abs(g_star) if g_star != 0 else 1.0

    n = max(8 * (m + 1) + 64, 600)
    y1, w1 = _gauss_interval(0.0, theta, n)
    y2, w2 = _gauss_interval(theta, math.pi, n)
    h1 = np.conj(zeta) * _spec_target(x_star + q * y1, spec)
    h_theta = np.conj(zeta) * g_far
    real = bool(np.max(np.abs(np.imag(h1)), initial=0.0) <= 1e-14 * max(np.max(np.abs(h1)), 1e-300))
    real = real and abs(np.imag(h_theta)) <= 1e-14 * max(abs(h_theta), 1e-300)
    if real:
        h1, h_theta = np.real(h1), float(np.real(h_theta))
        zeta = float(np.real(zeta))

    s = (y2 - theta) / ell
    bump = 2.0 * s - s**2
    p0 = h_theta - 1.5 * (h_theta + np.dot(w1, h1) / ell) * bump  # p at beta_0 = 0
    dp = (1.5 * math.pi / ell) * bump  # dp / d beta_0
    k = np.arange(1, m + 1)
    c1, c2 = np.cos(np.outer(k, y1)), np.cos(np.outer(k, y2))
    beta = (2.0 / math.pi) * (c1 @ (w1 * h1) + c2 @ (w2 * p0))
    dbeta = (2.0 / math.pi) * (c2 @ (w2 * dp))
    return {"x_star": x_star, "zeta": zeta, "beta": beta, "dbeta": dbeta, "real": real}


def _continuation_series(profile: dict, beta0: complex) -> tuple[float | complex, np.ndarray, np.ndarray]:
    """``(a0, ak, bk)`` of the continuation with closure mean *beta0*."""
    zeta, x_star = profile["zeta"], profile["x_star"]
    beta = profile["beta"] + beta0 * profile["dbeta"]
    k = np.arange(1, beta.size + 1)
    return _scalar(zeta * beta0), zeta * beta * np.sin(k * x_star), zeta * beta * np.cos(k * x_star)


def _optimal_closure_mean(profile: dict, val: list) -> float | complex:
    """The closure mean minimising sum_k |c_k| without raising the error of beta_0 = 0.

    With eps_0 the largest error on the validation grid at beta_0 = 0, solve

        min_{beta_0}  |beta_0| + sum_{k>=1} |beta_k(beta_0)|
        subject to    |g_m(x; beta_0) - g(x)| <= eps_0  on the grid,

    which is sum_k |c_k| up to the common factor |zeta|.  Both sides are
    affine in beta_0, so the problem is convex: for a real profile a
    weighted median on an interval, solved exactly; for a complex one a
    weighted Fermat-Weber problem on an intersection of discs, solved by
    nested golden-section search.
    """
    zeta, x_star, beta, dbeta = profile["zeta"], profile["x_star"], profile["beta"], profile["dbeta"]
    k = np.arange(1, beta.size + 1)
    a0, ak, bk = _continuation_series(profile, 0.0)
    da0, dak, dbk = 1.0, dbeta * np.sin(k * x_star), dbeta * np.cos(k * x_star)  # per unit zeta * beta_0
    c0 = _pack_coefficients(a0, ak, bk, "full")
    dc = _pack_coefficients(da0, dak, dbk, "full")
    # Residual e(x) + beta_0 D(x), rotated by conj(zeta): |conj(zeta) e + beta_0 D| <= eps_0.
    e = np.concatenate([np.conj(zeta) * (A @ c0 - y) for _, A, y, _ in val])
    D = np.concatenate([A @ dc for _, A, _, _ in val])
    eps0 = float(np.max(np.abs(e)))
    # Points and weights of the objective: |beta_0 - 0| and |dbeta_k| |beta_0 + beta_k / dbeta_k|.
    active = dbeta != 0
    points = np.concatenate([[0.0], -beta[active] / dbeta[active]])
    weights = np.concatenate([[1.0], np.abs(dbeta[active])])
    constant = np.sum(np.abs(beta[~active]))

    def objective(b):
        return constant + np.sum(weights * np.abs(b - points))

    # Each grid point with D != 0 confines beta_0 to a disc (an interval when real).
    big = np.abs(D) > 1e-14 * max(np.max(np.abs(D)), 1e-300)
    centres, radii = -e[big] / D[big], eps0 / np.abs(D[big])
    if profile["real"]:
        lo = float(np.max(centres.real - radii, initial=-np.inf))
        hi = float(np.min(centres.real + radii, initial=np.inf))
        lo, hi = min(lo, 0.0), max(hi, 0.0)  # beta_0 = 0 is feasible by construction
        candidates = np.concatenate([[0.0], np.clip(points.real, lo, hi)])
        values = np.array([objective(b) for b in candidates])
        best = values.min()
        tie = values <= best * (1 + 1e-14) + 1e-300
        return float(candidates[tie][np.argmin(np.abs(candidates[tie]))])

    # Complex profile: minimise over the real part u, each step minimising over
    # the feasible interval of the imaginary part v.
    span = float(np.max(np.abs(points))) + 1.0
    u_lo = max(float(np.max(centres.real - radii, initial=-np.inf)), -2.0 * span)
    u_hi = min(float(np.min(centres.real + radii, initial=np.inf)), 2.0 * span)
    u_lo, u_hi = min(u_lo, 0.0), max(u_hi, 0.0)

    def v_range(u):
        half = np.sqrt(np.maximum(radii**2 - (u - centres.real) ** 2, 0.0))
        lo = float(np.max(centres.imag - half, initial=-2.0 * span))
        hi = float(np.min(centres.imag + half, initial=2.0 * span))
        return (lo, hi) if lo <= hi else (0.5 * (lo + hi),) * 2

    def best_v(u):
        lo, hi = v_range(u)
        v = _golden_minimum(lambda v: objective(u + 1j * v), max(lo, -2.0 * span), min(hi, 2.0 * span))
        return v, objective(u + 1j * v)

    u = _golden_minimum(lambda u: best_v(u)[1], u_lo, u_hi)
    v, value = best_v(u)
    if objective(0.0) <= value:  # keep beta_0 = 0 unless strictly better
        return 0.0
    return complex(u, v)


def _golden_minimum(fun, lo: float, hi: float) -> float:
    """The minimiser of a convex *fun* on [lo, hi], by golden-section search."""
    if hi - lo <= 0.0:
        return lo
    ratio = (math.sqrt(5.0) - 1.0) / 2.0
    tol = 1e-13 * max(1.0, abs(lo), abs(hi))
    a, b = lo, hi
    c, d = b - ratio * (b - a), a + ratio * (b - a)
    fc, fd = fun(c), fun(d)
    while b - a > tol:
        if fc <= fd:
            b, d, fd = d, c, fc
            c = b - ratio * (b - a)
            fc = fun(c)
        else:
            a, c, fc = c, d, fd
            d = a + ratio * (b - a)
            fd = fun(d)
    candidates = [lo, hi, 0.5 * (a + b)]
    return min(candidates, key=fun)


def _continuation_coefficients(
    spec: dict, m: int, val: list
) -> tuple[float | complex, np.ndarray, np.ndarray, float | complex]:
    """The quadratic continuation of order *m*, and the closure mean it used."""
    profile = _continuation_profile(spec, m)
    beta0 = spec.get("closure_mean")
    if beta0 is None:
        beta0 = _optimal_closure_mean(profile, val)
    a0, ak, bk = _continuation_series(profile, beta0)
    return a0, ak, bk, beta0


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


# The series below work for real or complex u.


def _series_exp(u: np.ndarray) -> np.ndarray:
    """exp(u) as a power series, from y' = u' y."""
    y = np.zeros_like(u)
    y[0] = np.exp(u[0])
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
    y[0] = np.sqrt(u[0])
    for k in range(1, u.size):
        y[k] = (u[k] - y[1:k] @ y[k - 1 : 0 : -1]) / (2.0 * y[0])
    return y


def _series_polynomial(u: np.ndarray, polynomial: tuple[np.ndarray, str]) -> np.ndarray:
    """sum_j gamma_j B_j(u) as a power series, by the basis's three-term recurrence.

    The recurrence runs on truncated power series, so Chebyshev and Legendre
    coefficients are never converted to the (badly conditioned) monomial basis.
    """
    coefficients, basis = polynomial
    n = u.size
    dtype = np.result_type(u, coefficients, float)

    def times_u(a: np.ndarray) -> np.ndarray:
        return np.convolve(a, u)[:n]

    if basis == "monomial":
        # Horner's rule.
        y = np.zeros(n, dtype=dtype)
        for gamma in coefficients[::-1]:
            y = times_u(y)
            y[0] += gamma
        return y

    one = np.zeros(n, dtype=dtype)
    one[0] = 1.0
    previous, current = one, u.astype(dtype)  # B_0 = 1 and B_1 = u in both bases
    y = coefficients[0] * previous
    if coefficients.size > 1:
        y = y + coefficients[1] * current
    for j in range(1, coefficients.size - 1):
        if basis == "chebyshev":
            following = 2.0 * times_u(current) - previous  # T_{j+1} = 2u T_j - T_{j-1}
        else:
            following = ((2 * j + 1) * times_u(current) - j * previous) / (j + 1)  # Legendre
        previous, current = current, following
        y = y + coefficients[j + 1] * current
    return y


_COMPOSE = {
    "identity": lambda u: u,
    "inversion": _series_reciprocal,
    "exponentiation": _series_exp,
    "square_root": _series_sqrt,
}


def _taylor_coefficients(spec: dict, m: int) -> np.ndarray:
    """d_0, ..., d_m: the Taylor coefficients of g(arcsin z)."""
    u = _arcsin_series(m) / spec["tau"]
    u[0] = spec["mu"]
    if spec["problem"] == "polynomial":
        return _series_polynomial(u, spec["polynomial"])
    return _COMPOSE[spec["problem"]](u)


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
    if not np.iscomplexobj(y):
        c = np.fft.rfft(y) / n  # c[k] is the complex coefficient c_k, and c_{-k} = conj(c_k)
        return float(c[0].real), -2.0 * c[1 : m + 1].imag, 2.0 * c[1 : m + 1].real
    # A complex g has independent c_k and c_{-k} = c[n - k]:
    # a_k = i (c_k - c_{-k}) and b_k = c_k + c_{-k}.
    c = np.fft.fft(y) / n
    pos, neg = c[1 : m + 1], c[n - 1 : n - m - 1 : -1]
    return _scalar(c[0]), 1j * (pos - neg), pos + neg


# ---------------------------------------------------------------------------
# Candidate for one m
# ---------------------------------------------------------------------------

_COEFFICIENTS = {
    "reflected": _reflected_coefficients,
    "arcsine_taylor": _arcsine_taylor_coefficients,
}


def _explicit_candidate(spec: dict, m: int, target: float, metric: str, val_n: int) -> dict:
    """The closed-form series of order *m* for ``spec['method']``, in the format of ``_candidate_for_m``."""
    val = _validation_data(spec, m, val_n)
    closure_mean = None
    if spec["method"] == "reflected" and spec["eta"] > 2.0:
        a0, ak, bk, closure_mean = _continuation_coefficients(spec, m, val)
    else:
        a0, ak, bk = _COEFFICIENTS[spec["method"]](spec, m)
    # Packing drops the modes the basis excludes, which vanish by symmetry.
    coeffs = _pack_coefficients(a0, ak, bk, spec["basis"])
    solve = _summarise(coeffs, val, spec["basis"], m)
    return {
        "m": m,
        "lambda": None,
        "solve": solve,
        "feasible": _meets_target(solve, metric, target),
        "closure_mean": closure_mean,
    }
