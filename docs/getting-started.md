# Getting started

## Installation

`felcu` needs Python 3.11+ and NumPy.  Install from a clone of the repository:

```bash
pip install .
```

For development (tests, plus an editable install so changes take effect
immediately):

```bash
pip install -e ".[dev]"
```

## A first fit

Fit \(e^{H}\) for a Hermitian \(H\) with eigenvalues in \([0.1, 2]\) to an
\(L^\infty\) error of \(10^{-10}\):

```python
from felcu import solve_fe_coefficients

result = solve_fe_coefficients(
    "exponentiation",
    sigma_min=0.1,
    sigma_max=2.0,
    target_error=1e-10,
)

print(result.converged)            # True if target_error was met
print(result.m)                    # truncation order chosen
print(result.achieved_linf_error)  # error on the fitted interval
print(result.alpha)                # LCU normalisation
```

The solver tries a range of truncation orders \(m\) and, among those that
meet the target, returns the one with the smallest LCU normalisation
\(\alpha\).  Pass `selection="smallest_m"` for the shortest series instead.
Every candidate it tried is kept in `result.all_results`.

!!! tip "Choosing the range of m"
    By default the solver sweeps \(m\) up to 256.  Because \(\alpha\) usually
    keeps falling as \(m\) grows, the default `smallest_alpha` selection
    often picks the top of the range.  If circuit depth matters, cap the
    range, e.g. `m_values=range(4, 41, 2)`.  See
    [Tuning the solver](guide/parameters.md#truncation-order).

## Applying the result to a matrix

The fit is done in a scaled variable \(x\), so the operator you implement
is \(g\) evaluated at a scaled copy of \(H\):

| Construction | Operator |
|---|---|
| unshifted | \(f(H) \approx g(\tau H)\) |
| shifted | \(f(H) \approx g\bigl(\tau (H - \mu I)\bigr)\), with \(\mu = (\sigma_{\min} + \sigma_{\max})/2\) |

`result.tau` holds \(\tau\).  For a Hermitian matrix you can check the
approximation classically through an eigendecomposition:

```python
import numpy as np
from felcu import evaluate_fourier_series, solve_fe_coefficients

H = ...  # Hermitian matrix
result = solve_fe_coefficients("exponentiation", matrix=H, target_error=1e-8)

w, V = np.linalg.eigh(H)
g = evaluate_fourier_series(result.tau * w, result.a0, result.ak, result.bk)
approx = V @ np.diag(g) @ V.conj().T     # ~= expm(H)
```

Passing `matrix=` instead of `sigma_min`/`sigma_max` lets `felcu` read the
spectral interval from the matrix itself.

## Coefficients for an LCU circuit

The real coefficients `a0`, `ak`, `bk` describe \(g\) in the sine/cosine basis.
For a circuit you usually want the exponential form

\[
g(x) = \sum_{k=-m}^{m} c_k e^{ikx},
\]

which is available as `result.complex_coefficients`, ordered
\([c_{-m}, \dots, c_0, \dots, c_m]\).  You can also convert any
`(a0, ak, bk)` triple with
[`to_complex_coefficients`](api.md#felcu.to_complex_coefficients).
