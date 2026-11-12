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
  the original variable and on a Hermitian matrix.
- `ConvergenceWarning` when no order meets `target_error`; the best
  candidate is still returned, with `converged=False`.
- JSON save and load (`to_json` / `from_json`, `to_dict` / `from_dict`),
  format version 1.

[Unreleased]: https://github.com/QAaAT/felcu/compare/v1.0.0...HEAD
[1.0.0]: https://github.com/QAaAT/felcu/releases/tag/v1.0.0
