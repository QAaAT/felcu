# felcu

**Sobolev-regularised Fourier-extension coefficients for LCU block encodings.**

`felcu` computes coefficients for representing matrix functions via Linear
Combination of Unitaries (LCU), using Sobolev-regularised Fourier extensions.
It supports the three standard problems:

| Problem | Function |
|---|---|
| `action` | f(H) = H |
| `inversion` | f(H) = H⁻¹ |
| `exponentiation` | f(H) = exp(H) |

Both shifted (centred) and unshifted constructions are available.

## Method

For each candidate truncation order `m`, `felcu` fits a truncated Fourier
series to f on the spectral interval by weighted least squares on a
Gauss–Legendre grid (optionally also matching the first `order_fit`
derivatives).  The fit is regularised with a Sobolev penalty on the
coefficients,

```
λ · (1 + k²)^(k_order / 2)
```

for the k-th harmonic, which damps high frequencies.  For each `m`, `felcu`
searches the `lambdas` grid and then bisects to find the largest λ that still
meets `target_error`.  Stronger regularisation gives smaller coefficients and
so a smaller LCU normalisation α = |a₀| + Σₖ √(aₖ² + bₖ²).

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
print(f"Achieved L∞ error: {result.achieved_linf_error:.2e}")

# Evaluate the fitted series
for xs in fitted_domain_segments(result):
    y_fit = evaluate_fourier_series(xs, result.a0, result.ak, result.bk)
```

## Examples

Small, self-contained scripts live in `examples/`, grouped by problem.  Run
any of them from the project root, e.g. `python examples/action/01_basic_interval.py`.

| Script | Shows |
|---|---|
| `action/01_basic_interval.py` | f(H) = H on [-1, 1]; odd sine basis |
| `action/02_no_scale.py` | Fitting g(x) = x with no matrix-scale information |
| `action/03_shifted.py` | Shifted vs unshifted on an off-centre interval |
| `action/04_hermitian_matrix.py` | From a Hermitian matrix, checked via eigendecomposition |
| `action/05_non_hermitian_matrix.py` | Non-Hermitian matrix via singular values |
| `inversion/01_basic_interval.py` | H⁻¹ on a two-interval domain avoiding the pole |
| `inversion/02_condition_number.py` | How m and α grow with κ |
| `inversion/03_non_hermitian_matrix.py` | Inverting a non-Hermitian matrix, checked against `np.linalg.inv` |
| `inversion/04_shifted_definite.py` | Shifted inversion of positive- and negative-definite H |
| `exponentiation/01_no_scale.py` | Fitting exp(x) on [-π/2, π/2] |
| `exponentiation/02_target_sweep.py` | Truncation order m vs target error |
| `exponentiation/03_shifted_vs_unshifted.py` | Effect of centring on α |
| `exponentiation/04_matrix_exponential.py` | exp(H) from a matrix, checked against the exact exponential |
| `exponentiation/05_lcu_coefficients.py` | Complex coefficients c₋ₘ…cₘ for an LCU circuit |
| `exponentiation/06_sobolev_regularisation.py` | Effect of λ and `k_order` on α |
| `exponentiation/07_selection_and_candidates.py` | Inspecting `all_results`; `smallest_m` vs `smallest_alpha` |

## Package layout

```
felcu/
├── __init__.py        # Public API
├── types.py           # FourierExtensionResult dataclass
├── utils.py           # Public utility functions
├── spectral.py        # Spectral data extraction & problem-spec builder
├── solve.py           # Public solver (solve_fe_coefficients, compute_fe_coefficients)
├── _solver.py         # Internal: Fourier-extension fitting & regularisation search
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
    error_metric="linf", # 'l2' | 'linf'
    m=None,
    m_values=None,
    lambdas=None,        # Sobolev regularisation strengths λ to search
    k_order=1,           # Sobolev order of the penalty
    order_fit=1,
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
