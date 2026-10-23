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

The method is that of P. Brearley, T. L. Howarth and B. Adcock, [*Fourier extensions for matrix-function block encodings with error-independent subnormalization bounds*](https://arxiv.org/abs/2601.18024), arXiv:2601.18024; these docs follow its
[notation](guide/notation.md).

Four problems are supported:

| Problem | Function | Spectral data |
|---|---|---|
| `action` | \(f(\mathcal H) = \mathcal H\) | eigenvalues (Hermitian) or singular values |
| `inversion` | \(f(\mathcal H) = \mathcal H^{-1}\) | singular values (unshifted) or a definite eigenvalue interval (shifted) |
| `exponentiation` | \(f(\mathcal H) = e^{\mathcal H}\) | eigenvalues |
| `square_root` | \(f(\mathcal H) = \sqrt{\mathcal H}\) | positive-definite eigenvalue interval (shifted only) |

A matrix or spectral bounds are required.  Eigenvalue intervals use the
*shifted* (centred) construction; the *unshifted* construction is kept for
singular values, so that action and inversion work for non-Hermitian
matrices.  See [Problems and constructions](guide/problems.md).

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
- [Problems and constructions](guide/problems.md): shifted vs unshifted,
  scaling and bases.
- [Method](guide/method.md): what the solver actually computes.
- [Tuning the solver](guide/parameters.md): every parameter and when to change it.
- [Examples](examples.md): short scripts, one idea each, plus
  plots.
- [API reference](api.md).

## Citation

If you use `felcu` in your work, please cite the paper:

```bibtex
@misc{brearley2026fourier,
  title         = {Fourier extensions for matrix-function block encodings
                   with error-independent subnormalization bounds},
  author        = {Brearley, Peter and Howarth, Thomas L. and Adcock, Ben},
  year          = {2026},
  eprint        = {2601.18024},
  archivePrefix = {arXiv},
  url           = {https://arxiv.org/abs/2601.18024},
}
```

The repository's
[`CITATION.cff`](https://github.com/QAaAT/felcu/blob/main/CITATION.cff)
holds the same reference, and GitHub's "Cite this repository" button
exports it in other formats.

## License

`felcu` is released under the
[MIT License](https://github.com/QAaAT/felcu/blob/main/LICENSE).
