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

## Installation

```bash
pip install .
```

Or, for development (editable install plus test dependencies):

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

## Documentation

Full documentation is at **https://thomashowarth.github.io/felcu/**:

- [Problems and constructions](https://thomashowarth.github.io/felcu/guide/problems/): shifted vs unshifted, scaling, bases
- [Method](https://thomashowarth.github.io/felcu/guide/method/): the Sobolev-regularised Fourier extension
- [Tuning the solver](https://thomashowarth.github.io/felcu/guide/parameters/): every parameter
- [Examples](https://thomashowarth.github.io/felcu/examples/): the scripts in [`examples/`](examples/)
- [API reference](https://thomashowarth.github.io/felcu/api/)

To build it locally:

```bash
pip install -e ".[docs]"
mkdocs serve
```
