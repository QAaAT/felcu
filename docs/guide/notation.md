# Notation

`felcu` implements the Sobolev-regularised, reflected and arcsine-Taylor
Fourier extensions of

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
| \(a < 0 < b\) | eigenvalues either side of 0, \(\operatorname{spec}(\mathcal H) \subseteq [\lambda_{\min}, a] \cup [b, \lambda_{\max}]\) (indefinite inversion) | `gap=(a, b)`, `result.gap` |
| \(\mu = (\lambda_{\max} + \lambda_{\min})/2\) | spectral midpoint; \(0\) for singular values | `result.mu` |
| \(\delta = (\lambda_{\max} - \lambda_{\min})/2\) | spectral half-width; \(\Sigma_{\max}\) for singular values | derived |
| \(\kappa\) | condition number, ratio of largest to smallest \(\lvert\lambda\rvert\) or \(\Sigma\) | derived |

`sigma_min` and `sigma_max` hold whichever endpoints the construction uses
(see [Spectral information](problems.md#spectral-information));
`result.spectral_kind_used` says which.

## Scaling and the fitted function

| Symbol | Meaning | In `felcu` |
|---|---|---|
| \(\mathbb T = [-\pi, \pi)\) | one period | |
| \(\eta > 1\) | domain extension factor | `eta`, `result.eta` |
| \(\Omega_\eta = [-\pi/\eta, \pi/\eta]\) | fitted interval | `result.x_max` \(= \pi/\eta\) |
| \([-x_{\max}, \tau(a - \mu)] \cup [\tau(b - \mu), x_{\max}]\) | two-interval fitted set (indefinite or singular-value inversion); \(\Omega^{\rm indef}_{\eta,\kappa}\) when symmetric | `result.x_gap`, `result.x_max`; `result.domain == "double"` |
| \(\Omega\) | the fitted set in general | `fitted_domain_segments(result)` |
| \(\tau\) | scale, \(\mathcal G = \tau(\mathcal H - \mu I)\) | `result.tau` |
| \(\mathcal G\) | scaled matrix with \(\operatorname{spec}(\mathcal G) \subseteq \Omega\) | |
| \(f\) | the matrix function: \(x\), \(x^{-1}\), \(e^x\) or \(\sqrt x\) | `problem=` |
| \(g(x) = f(\mu + x/\tau)\) | fitted (scaled) target, \(f(\mathcal H) = g(\mathcal G)\) | `evaluate_target_function(x, result)` |

For singular values, \(\mathcal H\) is the Hermitian dilation of \(A\), whose
spectrum \(\pm\operatorname{sv}(A)\) is centred at \(\mu = 0\), so
\(g(x) = f(x/\tau)\) and the series is odd.

## The series

| Symbol | Meaning | In `felcu` |
|---|---|---|
| \(m\) | truncation order | `m`, `m_values`, `result.m` |
| \(g_m(x) = \sum_{k=-m}^{m} c_k e^{ikx}\) | the Fourier extension | `evaluate_fourier_series(x, result.a0, result.ak, result.bk)` |
| \(a_0, a_k, b_k\) | real coefficients, \(g_m = a_0 + \sum_k [a_k \sin kx + b_k \cos kx]\) | `result.a0`, `result.ak`, `result.bk` |
| \(c_k\), with \(c_0 = a_0\), \(c_{\pm k} = (b_k \mp i a_k)/2\) | complex (LCU) coefficients | `result.complex_coefficients` \(= [c_{-m}, \dots, c_m]\) |
| \(\alpha_m = \sum_k \lvert c_k\rvert\) | subnormalisation | `result.alpha` |
| \(\epsilon_m = \|g - g_m\|_{L^\infty(\Omega)}\) | approximation (and spectral) error | `result.achieved_linf_error` |
| \(\epsilon_{\rm t}\) | target tolerance | `target_error` |
| refl, asin, reg | reflected, arcsine-Taylor and Sobolev-regularised strategies | `method="reflected"`, `"arcsine_taylor"`, `"sobolev"`; `result.method` |

## Explicit methods

| Symbol | Meaning | In `felcu` |
|---|---|---|
| \(F(x) = g(\arcsin(\sin x))\) | reflected extension of \(g\) about \(\pm\pi/2\) | `method="reflected"` |
| \(d_k\) | Taylor coefficients, \(g(\arcsin z) = \sum_k d_k z^k\) | `method="arcsine_taylor"` |
| \(\rho_\eta = \sin(\pi/\eta)\) | arcsine-Taylor convergence factor | |
| \(\Omega^{\rm asin}_\eta = [\mu - \eta\delta/2, \mu + \eta\delta/2]\) | interval on which both explicit methods evaluate \(f\) | |

## Regularisation

These apply to `method="sobolev"` only; the explicit methods report them as
`None`.

| Symbol | Meaning | In `felcu` |
|---|---|---|
| \(H^w_r(\mathbb T)\) | weighted fractional Sobolev norm, weight \(1 + (r\lvert k\rvert)^{2w}\) | |
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
