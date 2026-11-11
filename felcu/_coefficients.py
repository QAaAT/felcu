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


def _basis_matrix(x: np.ndarray, m: int, basis: str, d: int = 0) -> np.ndarray:
    """The basis functions (or their d-th derivatives) evaluated at *x*, one column each."""
    k = np.arange(1, m + 1, dtype=float)
    kx = np.outer(x, k) + d * np.pi / 2  # d-th derivative of sin/cos is a phase shift
    scale = k**d
    constant = np.full((x.size, 1), 1.0 if d == 0 else 0.0)
    return _assemble(basis, constant, scale * np.sin(kx), scale * np.cos(kx), axis=1)


# ---------------------------------------------------------------------------
# Target function
# ---------------------------------------------------------------------------


def _target(x: np.ndarray, problem: str, tau: float, mu: float, d: int = 0) -> np.ndarray:
    """d-th derivative of the fitted target g(x) = F(x / tau + mu).

    F is ``t``, ``1/t``, ``exp(t)`` or ``sqrt(t)`` for identity,
    inversion, exponentiation and square root.  ``tau = inf`` (a zero-width
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
    else:
        raise ValueError(f"Unknown problem '{problem}'.")
    return (1.0 / tau**d) * F_d


def _scalar(value) -> float | complex:
    """*value* as a Python float, or as a complex if it has a nonzero imaginary part."""
    value = complex(value)
    return value.real if value.imag == 0.0 else value


def _value(problem: str, t: float) -> float | complex:
    """F(t), which is also the constant g on a zero-width interval centred at t."""
    return _scalar(_target(0.0, problem, math.inf, t))


def _spec_target(x: np.ndarray, spec: dict, d: int = 0) -> np.ndarray:
    """:func:`_target` for a problem spec."""
    return _target(x, spec["problem"], spec["tau"], spec["mu"], d)


# ---------------------------------------------------------------------------
# Coefficient packing/unpacking
# ---------------------------------------------------------------------------
#
# A real-valued g has real a0, ak, bk, stored as float64.  A complex-valued g
# has complex ones (complex128): its exponential coefficients satisfy
# c_{-k} = conj(c_k) only when g is real, so nothing below assumes it.


def _coefficient_dtype(*values) -> np.dtype:
    """float64 for real coefficients, complex128 as soon as any is complex."""
    return np.result_type(float, *(np.asarray(v) for v in values))


def _pack_coefficients(a0: float | complex, ak: Sequence, bk: Sequence, basis: str) -> np.ndarray:
    dtype = _coefficient_dtype(a0, ak, bk)
    ak = np.asarray(ak, dtype=dtype).reshape(-1)
    bk = np.asarray(bk, dtype=dtype).reshape(-1)
    if ak.shape != bk.shape:
        raise ValueError("ak and bk must have the same length.")
    return _assemble(basis, [a0], ak, bk)


def _add_baseline(c: np.ndarray, basis: str, baseline: float | complex) -> np.ndarray:
    c = np.array(c, dtype=_coefficient_dtype(c, baseline), copy=True)
    if baseline != 0.0:
        if not _has_constant(basis):
            raise ValueError("A nonzero baseline requires a constant basis function.")
        c[0] += baseline
    return c


def _unpack_coefficients(c: np.ndarray, basis: str, m: int) -> tuple[float | complex, np.ndarray, np.ndarray]:
    """``(a0, ak, bk)`` from packed coefficients, complex if *c* is complex."""
    start = int(_has_constant(basis))
    a0 = c.dtype.type(c[0]).item() if start else c.dtype.type(0).item()
    ak = c[start : start + m].copy()
    bk = c[start + m :].copy() if _has_cosine(basis) else np.zeros(m, dtype=c.dtype)
    return a0, ak, bk
