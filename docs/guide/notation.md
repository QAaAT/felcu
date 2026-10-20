# Notation

`felcu` implements the Sobolev-regularised Fourier extensions of

> P. Brearley, T. L. Howarth and B. Adcock, [*Fourier extensions for matrix-function block encodings with error-independent subnormalization bounds*](https://arxiv.org/abs/2601.18024), arXiv:2601.18024.

These docs use the paper's notation.  The table maps each symbol to the
corresponding argument or result field.

## Matrices and spectra

| Symbol | Meaning | In `felcu` |
|---|---|---|
| \(\mathcal H\) | Hermitian input matrix | `matrix=` |
| \(A = U\Sigma V^\dagger\) | general input matrix (acted on through its singular values) | `matrix=` |
| \(\lambda_{\min}, \lambda_{\max}\) | eigenvalue endpoints, \(\operatorname{spec}(\mathcal H) \subseteq [\lambda_{\min}, \lambda_{\max}]\) | `sigma_min`, `sigma_max` with eigenvalues |
| \(\Sigma_{\min}, \Sigma_{\max}\) | singular-value endpoints | `sigma_min`, `sigma_max` with singular values |
| \(\mu = (\lambda_{\max} + \lambda_{\min})/2\) | spectral midpoint | derived; used by shifted fits |
| \(\delta = (\lambda_{\max} - \lambda_{\min})/2\) | spectral half-width | derived; used by shifted fits |
| \(\kappa\) | condition number, ratio of largest to smallest \(|\lambda|\) or \(\Sigma\) | derived |

`sigma_min` and `sigma_max` hold whichever endpoints the construction uses
(see [Spectral information](problems.md#spectral-information));
`result.spectral_kind_used` says which.

## Scaling and the fitted function

| Symbol | Meaning | In `felcu` |
|---|---|---|
| \(\mathbb T = [-\pi, \pi)\) | one period | |
| \(\eta > 1\) | domain extension factor | `eta`, `result.eta` |
| \(\Omega_\eta = [-\pi/\eta, \pi/\eta]\) | fitted interval | `result.x_max` \(= \pi/\eta\) |
| \(\Omega^{\rm indef}_{\eta,\kappa} = [-x_{\max}, -x_{\min}] \cup [x_{\min}, x_{\max}]\) | two-interval fitted set (unshifted inversion) | `result.x_min`, `result.x_max`; `result.domain == "double"` |
| \(\Omega\) | the fitted set in general | `fitted_domain_segments(result)` |
| \(\tau\) | scale, \(\mathcal G = \tau(\mathcal H - \mu I)\) | `result.tau` |
| \(\mathcal G\) | scaled matrix with \(\operatorname{spec}(\mathcal G) \subseteq \Omega\) | |
| \(f\) | the matrix function: \(x\), \(x^{-1}\), \(e^x\), \(\sqrt x\) or \(\log x\) | `problem=` |
| \(g(x) = f(\mu + x/\tau)\) | fitted (scaled) target, \(f(\mathcal H) = g(\mathcal G)\) | `evaluate_target_function(x, result)` |

Unshifted constructions take \(\mu = 0\), so \(\mathcal G = \tau\mathcal H\)
and \(g(x) = f(x/\tau)\).

## The series

| Symbol | Meaning | In `felcu` |
|---|---|---|
| \(m\) | truncation order | `m`, `m_values`, `result.m` |
| \(g_m(x) = \sum_{k=-m}^{m} c_k e^{ikx}\) | the Fourier extension | `evaluate_fourier_series(x, result.a0, result.ak, result.bk)` |
| \(a_0, a_k, b_k\) | real coefficients, \(g_m = a_0 + \sum_k [a_k \sin kx + b_k \cos kx]\) | `result.a0`, `result.ak`, `result.bk` |
| \(c_k\), with \(c_0 = a_0\), \(c_{\pm k} = (b_k \mp i a_k)/2\) | complex (LCU) coefficients | `result.complex_coefficients` \(= [c_{-m}, \dots, c_m]\) |
| \(\alpha_m = \sum_k |c_k|\) | subnormalisation | `result.alpha` |
| \(\epsilon_m = \|g - g_m\|_{L^\infty(\Omega)}\) | approximation (and spectral) error | `result.achieved_linf_error` |
| \(\epsilon_{\rm t}\) | target tolerance | `target_error` |

## Regularisation

| Symbol | Meaning | In `felcu` |
|---|---|---|
| \(H^w_r(\mathbb T)\) | weighted fractional Sobolev norm, weight \(1 + (r|k|)^{2w}\) | |
| \(0 < r \le 1\) | Sobolev scale | `sobolev_r`, `result.sobolev_r` |
| \(w > 1/2\) | Sobolev order | `sobolev_w`, `result.sobolev_w` |
| \(\gamma > 0\) | regularisation weight | `lambdas`, `result.lambda_used` hold \(\sqrt{2\pi\gamma}\) |
| \(H^1(\Omega)\) | fitting norm | `order_fit=1` (default); `order_fit=0` for \(L^2(\Omega)\) |

!!! note "`lambda` is not an eigenvalue"
    For historical reasons the regularisation strength is called
    `lambda` in the code (`lambdas`, `lambda_used`).  It is
    \(\sqrt{2\pi\gamma}\), the parameter that enters the augmented
    least-squares system; it is unrelated to the eigenvalues
    \(\lambda_j\).
