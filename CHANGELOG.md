# Changelog

All notable changes to `felcu` are recorded here.  The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and `felcu` uses
[semantic versioning](https://semver.org/spec/v2.0.0.html): from 1.0.0 on,
the public API (everything in `felcu.__all__`) and the JSON result format
change incompatibly only in a new major version.

## [Unreleased]

## [1.0.0] - YYYY-MM-DD

First stable release, accompanying P. Brearley, T. L. Howarth and B. Adcock,
*Fourier extensions for matrix-function block encodings with
error-independent subnormalization bounds*,
[arXiv:2601.18024](https://arxiv.org/abs/2601.18024).

### Problems

- `identity` (alias `action`), `inversion`, `exponentiation`, `square_root`
  and `logarithm`, from a matrix or from spectral bounds.
- Every fit is centred on the spectrum, `f(H) = g(tau (H - mu I))`.
- Identity and inversion on singular values, for non-Hermitian matrices,
  through the Hermitian dilation.
- Inversion on definite and indefinite spectra.

### Methods

- `sobolev` (default): least squares regularised with the full `H_r^w`
  Sobolev norm, near-exponential convergence down to the target, and a
  subnormalisation of order `||f(H)||`.  Per-problem default Sobolev scale
  `r` and extension parameter `eta`, as in the paper.
- `reflected`: reflected extensions, with optimal subnormalisation for
  convex endpoint-maximising `f`.
- `arcsine_taylor`: arcsine-Taylor expansions, converging exponentially.

### Results

- `solve_fe_coefficients` sweeps the truncation order and returns a
  `FourierExtensionResult` with the coefficients in the real basis
  (`a0`, `ak`, `bk`) and the complex basis (`complex_coefficients`), the
  achieved L2 and L∞ errors, the subnormalisation `alpha`, and every
  candidate tried.
- `result.evaluate(x)` and `result.apply(H)` evaluate the approximation in
  the original variable and on a Hermitian matrix.  `apply` rejects
  eigenvalues outside the domain of `f` (zero for inversion, non-positive
  for the square root and logarithm) and accepts single-precision input.
- `result.circuit_coefficients` and `to_circuit_coefficients` give the
  coefficients in the order the compressed LCU circuit consumes them,
  `[c_0, 0, c_1, c_-1, ..., c_m, c_-m]`; `format_circuit_coefficients`
  gives them as a Python list literal.
- `result.method_details` holds the values specific to the method, such as
  the Sobolev `lambda_used`, `sobolev_r` and `sobolev_w`, which are also
  available as attributes.
- `evaluate_fourier_series` works through long inputs in blocks, so memory
  does not grow with the number of points times the order.
- `ConvergenceWarning` when no order meets `target_error`; the best
  candidate is still returned, with `converged=False`.
- JSON save and load (`to_json` / `from_json`, `to_dict` / `from_dict`),
  format version 1.  Problem, method and metric names are checked on
  loading, and aliases load as their canonical names.

### Inputs

- Problems, methods and error metrics accept aliases, ignoring case
  (`canonical_problem_name`, `canonical_method_name`,
  `canonical_error_metric`).
- Inputs are checked up front: `target_error` must be non-negative,
  `m`, `m_values` and `order_fit` non-negative integers, and `lambdas`
  non-negative; options of another method raise an error naming the method
  they belong to.

[Unreleased]: https://github.com/QAaAT/felcu/compare/v1.0.0...HEAD
[1.0.0]: https://github.com/QAaAT/felcu/releases/tag/v1.0.0
