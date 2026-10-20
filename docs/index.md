# felcu

**Sobolev-regularised Fourier-extension coefficients for LCU block encodings.**

`felcu` computes the coefficients needed to block encode a matrix function
\(f(\mathcal H)\) as a Linear Combination of Unitaries (LCU).  It rescales
the spectrum to \(\mathcal G = \tau(\mathcal H - \mu I)\) and approximates
the fitted function \(g(x) = f(\mu + x/\tau)\) on the fitted set by a
Fourier extension

\[
g_m(x) = \sum_{k=-m}^{m} c_k e^{ikx} = a_0 + \sum_{k=1}^{m} \bigl[ a_k \sin(kx) + b_k \cos(kx) \bigr],
\]

so that \(f(\mathcal H) \approx g_m(\mathcal G) = \sum_k c_k e^{ik\mathcal G}\),
a linear combination of Hamiltonian simulations.  The coefficients come from
a Sobolev-regularised fit, which converges near-exponentially down to the
target tolerance while keeping the subnormalisation
\(\alpha_m = \sum_k |c_k|\) at the optimal scale \(O(\|f(\mathcal H)\|)\).

The method is that of P. Brearley, T. L. Howarth and B. Adcock, *Fourier
extensions for matrix-function block encodings with error-independent
subnormalization bounds*; these docs follow its [notation](guide/notation.md).

Three problems are supported:

| Problem | Function | Spectral data |
|---|---|---|
| `action` | \(f(\mathcal H) = \mathcal H\) | eigenvalues (Hermitian) or singular values |
| `inversion` | \(f(\mathcal H) = \mathcal H^{-1}\) | singular values (unshifted) or a definite eigenvalue interval (shifted) |
| `exponentiation` | \(f(\mathcal H) = e^{\mathcal H}\) | eigenvalues |

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
result.alpha        # subnormalisation alpha_m
```

## Where next

- [Getting started](getting-started.md): install, a first fit, and how to
  apply the result to a matrix.
- [Notation](guide/notation.md): the paper's symbols and their names in the code.
- [Method](guide/method.md): what the solver actually computes.
- [Tuning the solver](guide/parameters.md): every parameter and when to change it.
- [Examples](examples.md): sixteen short scripts, one idea each.
- [API reference](api.md).
