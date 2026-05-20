# felcu

**Fourier-extension coefficients for LCU block encodings.**

`felcu` computes coefficients for representing matrix functions via Linear
Combination of Unitaries (LCU), using a Fourier-extension approach.  It
supports the three standard problems:

| Problem | Function |
|---|---|
| `action` | f(H) = H |
| `inversion` | f(H) = H⁻¹ |
| `exponentiation` | f(H) = exp(H) |

Both shifted (centred) and unshifted constructions are available.

## Installation

```bash
pip install .
```

Or, for development:

```bash
pip install -e ".[dev]"
```

## Quick start

```python
from felcu import solve_fe_coefficients, evaluate_fourier_series, fitted_domain_segments
import numpy as np

# Fit exp(H) for a Hermitian matrix with eigenvalues in [0.1, 2.0]
result = solve_fe_coefficients(
    "exponentiation",
    sigma_min=0.1,
    sigma_max=2.0,
    target_error=1e-10,
)

print(f"Converged: {result.converged}")
print(f"Truncation order m = {result.m}")
print(f"Achieved L2 error: {result.achieved_l2_error:.2e}")

# Evaluate the fitted series
for xs in fitted_domain_segments(result):
    y_fit = evaluate_fourier_series(xs, result.a0, result.ak, result.bk)
```

## Package layout

```
felcu/
├── __init__.py        # Public API
├── types.py           # FourierExtensionResult dataclass
├── utils.py           # Public utility functions
├── spectral.py        # Spectral data extraction & problem-spec builder
├── solve.py           # Public solver (solve_fe_coefficients, compute_fe_coefficients)
├── _coefficients.py   # Internal: basis matrices, coefficient packing
└── _linalg.py         # Internal: QR solver, quadrature, penalties
```

## API reference

### `solve_fe_coefficients`

```python
solve_fe_coefficients(
    problem,          # 'action' | 'inversion' | 'exponentiation'
    *,
    shifted=False,
    matrix=None,      # square ndarray  (mutually exclusive with sigma_*)
    sigma_min=None,
    sigma_max=None,
    spectral_kind="auto",
    eta=None,
    target_error=1e-12,
    error_metric="l2",   # 'l2' | 'linf'
    m=None,
    m_values=None,
    lambdas=None,
    k_order=1,
    order_fit=0,
    fit_oversample=8,
    val_n=4001,
    selection="smallest_m",
    hermitian_tol=1e-12,
) -> FourierExtensionResult
```

### `compute_fe_coefficients`

Thin wrapper around `solve_fe_coefficients` that returns `(a0, ak, bk)` directly.

### `FourierExtensionResult`

Key attributes:

| Attribute | Description |
|---|---|
| `a0`, `ak`, `bk` | Fourier coefficients |
| `complex_coefficients` | Complex form `[c_{-m}, …, c_m]` |
| `m` | Truncation order used |
| `converged` | Whether `target_error` was met |
| `achieved_l2_error` | Achieved L2 error |
| `achieved_linf_error` | Achieved L∞ error |
| `all_results` | All candidate results (sorted by `m`) |
