"""Type definitions for felcu."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np


ArrayLike = np.ndarray | Sequence[float] | float


@dataclass(frozen=True)
class FourierExtensionResult:
    """Result of a Fourier-extension coefficient computation."""

    problem: str
    shifted: bool
    basis_used: str
    m: int
    eta: float
    tau: float
    alpha: float
    target_error: float
    error_metric: str
    achieved_error: float
    achieved_l2_error: float
    achieved_linf_error: float
    converged: bool
    lambda_used: float
    a0: float
    ak: np.ndarray
    bk: np.ndarray
    complex_coefficients: np.ndarray
    k: np.ndarray
    domain: str
    x_max: float
    x_min: Optional[float]
    sigma_min: Optional[float]
    sigma_max: Optional[float]
    spectral_kind_used: Optional[str]
    matrix_shape: Optional[tuple[int, int]]
    matrix_is_hermitian: Optional[bool]
    matrix_scale_provided: bool
    all_results: tuple["FourierExtensionResult", ...] = ()

    @property
    def coefficients(self) -> np.ndarray:
        from felcu._coefficients import _pack_coefficients
        return _pack_coefficients(self.a0, self.ak, self.bk, self.basis_used)

    @property
    def achieved_inf_error(self) -> float:
        return self.achieved_linf_error
