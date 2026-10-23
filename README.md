# felcu

**Sobolev-regularised Fourier-extension coefficients for LCU block encodings.**

`felcu` computes coefficients for representing matrix functions via Linear
Combination of Unitaries (LCU), using the Sobolev-regularised Fourier
extensions of P. Brearley, T. L. Howarth and B. Adcock, [*Fourier extensions for matrix-function block encodings with error-independent subnormalization bounds*](https://arxiv.org/abs/2601.18024), arXiv:2601.18024.
It supports four problems:

| Problem | Function |
|---|---|
| `action` | f(H) = H |
| `inversion` | f(H) = H⁻¹ |
| `exponentiation` | f(H) = exp(H) |
| `square_root` | f(H) = √H (H positive definite) |

A matrix or spectral bounds are required.  Eigenvalue intervals use the
shifted (centred) construction; action and inversion also have an unshifted
construction on singular values, for non-Hermitian matrices.

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

Full documentation is at **https://qaaat.github.io/felcu/**:

- [Notation](https://qaaat.github.io/felcu/guide/notation/): the paper's symbols and their names in the code
- [Problems and constructions](https://qaaat.github.io/felcu/guide/problems/): shifted vs unshifted, scaling, bases
- [Method](https://qaaat.github.io/felcu/guide/method/): the Sobolev-regularised Fourier extension
- [Tuning the solver](https://qaaat.github.io/felcu/guide/parameters/): every parameter
- [Examples](https://qaaat.github.io/felcu/examples/): the scripts in [`examples/`](examples/)
- [API reference](https://qaaat.github.io/felcu/api/)

To build it locally:

```bash
pip install -e ".[docs]"
mkdocs serve
```

## Citation

If you use `felcu` in your work, please cite the paper:

> P. Brearley, T. L. Howarth and B. Adcock, *Fourier extensions for
> matrix-function block encodings with error-independent subnormalization
> bounds*, arXiv:2601.18024 (2026).

GitHub's "Cite this repository" button gives the same reference in BibTeX and
APA formats, from [`CITATION.cff`](CITATION.cff).

## License

`felcu` is released under the [MIT License](LICENSE).
