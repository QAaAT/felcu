"""Internal coefficient, basis-matrix and target-function helpers for felcu."""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np

# ---------------------------------------------------------------------------
# Basis layout
# ---------------------------------------------------------------------------
#
# Coefficient vectors and basis-matrix columns are ordered
#
#     [constant] + [sin(kx), k = 1..m] + [cos(kx), k = 1..m]
#
# where 'odd_sine' has only the sine block, 'const_plus_sine' adds the
# constant, and 'full' has all three.


def _has_constant(basis: str) -> bool:
    return basis != "odd_sine"


def _has_cosine(basis: str) -> bool:
    return basis == "full"


def _basis_size(basis: str, m: int) -> int:
    return int(_has_constant(basis)) + m * (2 if _has_cosine(basis) else 1)


def _assemble(basis: str, constant, sine, cosine, axis: int = -1) -> np.ndarray:
    """Concatenate the parts of *basis* in its column order."""
    parts = [constant] if _has_constant(basis) else []
    parts.append(sine)
    if _has_cosine(basis):
        parts.append(cosine)
    return np.concatenate(parts, axis=axis)


def _basis_matrices(x: np.ndarray, m: int, basis: str, order: int = 0) -> list[np.ndarray]:
    """The basis functions and their first *order* derivatives at *x*, one column each.

    sin and cos are evaluated once: each derivative maps (sin kx, cos kx) to
    (k cos kx, -k sin kx).
    """
    k = np.arange(1, m + 1, dtype=float)
    kx = np.outer(x, k)
    sine, cosine = np.sin(kx), np.cos(kx)
    out = []
    for d in range(order + 1):
        constant = np.full((x.size, 1), 1.0 if d == 0 else 0.0)
        out.append(_assemble(basis, constant, sine, cosine, axis=1))
        if d < order:
            sine, cosine = k * cosine, -k * sine
    return out


def _basis_matrix(x: np.ndarray, m: int, basis: str) -> np.ndarray:
    """The basis functions evaluated at *x*, one column each."""
    return _basis_matrices(x, m, basis)[0]


# ---------------------------------------------------------------------------
# Target function
# ---------------------------------------------------------------------------


def _target(x: np.ndarray, problem: str, tau: float, mu: float, d: int = 0) -> np.ndarray:
    """d-th derivative of the fitted target g(x) = F(x / tau + mu).

    F is ``t``, ``1/t``, ``exp(t)``, ``sqrt(t)`` or ``log(t)`` for identity,
    inversion, exponentiation, square root and logarithm.  ``tau = inf`` (a zero-width
    interval) makes g constant.
    """
    x = np.asarray(x, dtype=float)
    t = x / tau + mu
    if problem == "identity":
        F_d = t if d == 0 else np.full_like(t, 1.0 if d == 1 else 0.0)
    elif problem == "inversion":
        if math.isfinite(tau):
            # (1 / tau^d) F^(d)(t), written without dividing x by tau.
            return tau * (-1.0) ** d * math.factorial(d) / (x + tau * mu) ** (d + 1)
        F_d = (-1.0) ** d * math.factorial(d) / t ** (d + 1)
    elif problem == "exponentiation":
        F_d = np.exp(t)
    elif problem == "square_root":
        # d-th derivative of t^(1/2) is (1/2)(1/2 - 1)...(1/2 - d + 1) t^(1/2 - d)
        F_d = math.prod(0.5 - j for j in range(d)) * t ** (0.5 - d)
    elif problem == "logarithm":
        # d-th derivative of log t is (-1)^(d-1) (d-1)! t^(-d) for d >= 1
        F_d = np.log(t) if d == 0 else (-1.0) ** (d - 1) * math.factorial(d - 1) / t**d
    else:
        raise ValueError(f"Unknown problem '{problem}'.")
    return (1.0 / tau**d) * F_d


def _value(problem: str, t: float) -> float:
    """F(t), which is also the constant g on a zero-width interval centred at t."""
    return float(_target(0.0, problem, math.inf, t))


def _spec_target(x: np.ndarray, spec: dict, d: int = 0) -> np.ndarray:
    """:func:`_target` for a problem spec."""
    return _target(x, spec["problem"], spec["tau"], spec["mu"], d)


# ---------------------------------------------------------------------------
# Coefficient packing/unpacking
# ---------------------------------------------------------------------------


def _coefficient_arrays(ak: Sequence, bk: Sequence, dtype=None) -> tuple[np.ndarray, np.ndarray]:
    """*ak* and *bk* as flat arrays of equal length."""
    ak = np.asarray(ak, dtype=dtype).reshape(-1)
    bk = np.asarray(bk, dtype=dtype).reshape(-1)
    if ak.shape != bk.shape:
        raise ValueError("ak and bk must have the same length.")
    return ak, bk


def _pack_coefficients(a0: float, ak: Sequence, bk: Sequence, basis: str) -> np.ndarray:
    ak, bk = _coefficient_arrays(ak, bk, float)
    return _assemble(basis, [a0], ak, bk)


def _add_baseline(c: np.ndarray, basis: str, baseline: float) -> np.ndarray:
    c = np.array(c, dtype=float, copy=True)
    if baseline != 0.0:
        if not _has_constant(basis):
            raise ValueError("A nonzero baseline requires a constant basis function.")
        c[0] += baseline
    return c


def _unpack_coefficients(c: np.ndarray, basis: str, m: int) -> tuple[float, np.ndarray, np.ndarray]:
    start = int(_has_constant(basis))
    a0 = float(c[0]) if start else 0.0
    ak = c[start : start + m].copy()
    bk = c[start + m :].copy() if _has_cosine(basis) else np.zeros(m)
    return a0, ak, bk


# ---------------------------------------------------------------------------
# Series evaluation and the complex and circuit forms
# ---------------------------------------------------------------------------

# Trig entries _evaluate_series builds at once (about 8 MB each for sin and cos).
_EVAL_BLOCK = 1 << 20


def _evaluate_series(x, a0, ak, bk) -> np.ndarray:
    """g(x) = a0 + sum_k ak sin(kx) + bk cos(kx); see :func:`~felcu.evaluate_fourier_series`."""
    x = np.asarray(x)
    x_flat = x.reshape(-1).astype(float)
    ak, bk = _coefficient_arrays(ak, bk)

    dtype = np.result_type(x, a0, ak, bk, float)
    y = np.full(x_flat.shape, a0, dtype=dtype)
    if ak.size:
        k = np.arange(1, ak.size + 1, dtype=float)
        # In blocks of points, so the (points x m) trig arrays stay small for a dense x.
        step = max(1, _EVAL_BLOCK // ak.size)
        for i in range(0, x_flat.size, step):
            angles = np.outer(x_flat[i : i + step], k)
            y[i : i + step] += np.sin(angles) @ ak + np.cos(angles) @ bk
    return y.reshape(x.shape)


def _complex_coefficients(a0, ak, bk) -> np.ndarray:
    """[c_{-m}, ..., c_m]; see :func:`~felcu.to_complex_coefficients`."""
    ak, bk = _coefficient_arrays(ak, bk, np.complex128)
    c0 = np.array([complex(a0)], dtype=np.complex128)
    if ak.size == 0:
        return c0
    c_pos = 0.5 * (bk - 1j * ak)
    c_neg = 0.5 * (bk[::-1] + 1j * ak[::-1])
    return np.concatenate([c_neg, c0, c_pos])


def _circuit_coefficients(a0, ak, bk) -> np.ndarray:
    """[c_0, 0, c_1, c_{-1}, ..., c_m, c_{-m}]; see :func:`~felcu.to_circuit_coefficients`."""
    c = _complex_coefficients(a0, ak, bk)
    m = (c.size - 1) // 2
    out = np.zeros(2 * m + 2, dtype=np.complex128)
    out[0] = c[m]
    out[2::2] = c[m + 1 :]  # c_1, ..., c_m
    out[3::2] = c[m - 1 :: -1] if m else c[:0]  # c_-1, ..., c_-m
    return out


def _format_circuit_coefficients(a0, ak, bk) -> str:
    """The circuit coefficients as a Python list literal that reproduces them exactly."""
    return "[" + ", ".join(repr(complex(ck)) for ck in _circuit_coefficients(a0, ak, bk)) + "]"
