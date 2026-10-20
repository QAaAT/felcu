# felcu

**Sobolev-regularised Fourier-extension coefficients for LCU block encodings.**

`felcu` computes the coefficients needed to implement a matrix function
\(f(H)\) as a Linear Combination of Unitaries (LCU).  It approximates the
scalar function \(f\) on the spectrum of \(H\) by a truncated Fourier series

\[
g(x) = a_0 + \sum_{k=1}^{m} \bigl( a_k \sin(kx) + b_k \cos(kx) \bigr),
\]

fitted by a Sobolev-regularised Fourier extension.  Each term
\(e^{\pm ikx}\) becomes a controlled power of a unitary in the LCU circuit,
and the regularisation keeps the LCU normalisation \(\alpha\) small.

Three problems are supported:

| Problem | Function | Spectral data |
|---|---|---|
| `action` | \(f(H) = H\) | eigenvalues (Hermitian) or singular values |
| `inversion` | \(f(H) = H^{-1}\) | singular values (unshifted) or a definite eigenvalue interval (shifted) |
| `exponentiation` | \(f(H) = e^{H}\) | eigenvalues |

Each comes in an *unshifted* form and a *shifted* (centred) form; see
[Problems and constructions](guide/problems.md).

## At a glance

```python
from felcu import solve_fe_coefficients

result = solve_fe_coefficients(
    "exponentiation",
    sigma_min=0.1,
    sigma_max=2.0,
    target_error=1e-10,
)

result.m            # truncation order
result.a0, result.ak, result.bk
result.complex_coefficients   # [c_{-m}, ..., c_m] for the LCU circuit
result.alpha        # LCU normalisation
```

## Where next

- [Getting started](getting-started.md): install, a first fit, and how to
  apply the result to a matrix.
- [Method](guide/method.md): what the solver actually computes.
- [Tuning the solver](guide/parameters.md): every parameter and when to change it.
- [Examples](examples.md): sixteen short scripts, one idea each.
- [API reference](api.md).
