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

Fit \(e^{\mathcal H}\) for a Hermitian \(\mathcal H\) with
\(\operatorname{spec}(\mathcal H) \subseteq [0.1, 2]\) to a target tolerance
\(\epsilon_{\rm t} = 10^{-10}\):

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
print(result.achieved_linf_error)  # epsilon_m on the fitted set
print(result.alpha)                # subnormalisation alpha_m
```

The solver tries a range of truncation orders \(m\) and, among those that
meet the target, returns the one with the smallest subnormalisation
\(\alpha_m\).  Pass `selection="smallest_m"` for the shortest series instead.
Every candidate it tried is kept in `result.all_results`.

!!! tip "Choosing the range of m"
    By default the solver sweeps \(m\) up to 256.  Because \(\alpha\) usually
    keeps falling as \(m\) grows, the default `smallest_alpha` selection
    often picks the top of the range.  If circuit depth matters, cap the
    range, e.g. `m_values=range(4, 41, 2)`.  See
    [Tuning the solver](guide/parameters.md#truncation-order).

This uses the default Sobolev-regularised fit.  Pass
`method="reflected"` or `method="arcsine_taylor"` for the paper's explicit
extensions, which trade convergence against subnormalisation differently;
see [Method](guide/method.md).

## Applying the result to a matrix

The fit is done in a scaled variable \(x\), so the operator you implement
is the series \(g_m\) evaluated at the scaled matrix \(\mathcal G\):

\[
f(\mathcal H) \approx g_m(\mathcal G), \qquad \mathcal G = \tau(\mathcal H - \mu I),
\]

with \(\mu\) the centre of the spectrum (`result.mu`) and \(\tau\) the scale
(`result.tau`).  For singular values, \(\mathcal H\) is the Hermitian
dilation of \(A\) and \(\mu = 0\).  For a Hermitian matrix you can check the
approximation classically through an eigendecomposition:

```python
import numpy as np
from felcu import evaluate_fourier_series, solve_fe_coefficients

H = ...  # Hermitian matrix
result = solve_fe_coefficients("exponentiation", matrix=H, target_error=1e-8)

w, V = np.linalg.eigh(H)
g = evaluate_fourier_series(result.tau * (w - result.mu), result.a0, result.ak, result.bk)
approx = V @ np.diag(g) @ V.conj().T     # ~= expm(H)
```

Passing `matrix=` instead of `sigma_min`/`sigma_max` lets `felcu` read the
spectral interval from the matrix itself.  One or the other is required.

## Coefficients for an LCU circuit

The real coefficients `a0`, `ak`, `bk` describe \(g_m\) in the sine/cosine
basis.  For a circuit you usually want the exponential form

\[
g_m(x) = \sum_{k=-m}^{m} c_k e^{ikx}, \qquad c_0 = a_0,\quad c_{\pm k} = \frac{b_k \mp i a_k}{2},
\]

which is available as `result.complex_coefficients`, ordered
\([c_{-m}, \dots, c_0, \dots, c_m]\).  You can also convert any
`(a0, ak, bk)` triple with
[`to_complex_coefficients`](api.md#felcu.to_complex_coefficients).

## Saving and loading results

A result can be written to JSON and read back exactly:

```python
from felcu import FourierExtensionResult

result.to_json("exp_coefficients.json")
loaded = FourierExtensionResult.from_json("exp_coefficients.json")
```

The file is standard JSON, so it can also be read outside Python.  It
holds every field of the result: the parameters (`problem`, `method`,
`eta`, `tau`, `sobolev_r`, ...), the errors and \(\alpha_m\), the real coefficients
`a0`, `ak`, `bk`, and the complex coefficients as

```json
"complex_coefficients": {"k": [-m, ..., m], "real": [...], "imag": [...]}
```

Floats are written with enough digits to round-trip exactly; an infinite
\(\tau\) (a zero-width interval) is stored as the string `"inf"`, and the
Sobolev-only fields (`lambda_used`, `sobolev_r`, `sobolev_w`) are `null`
for the explicit methods.  Pass
`include_candidates=True` to also keep `all_results`, every order tried
in the sweep.  [`to_dict`](api.md#felcu.FourierExtensionResult.to_dict)
and [`from_dict`](api.md#felcu.FourierExtensionResult.from_dict) give the
same data as a Python dictionary.
