# Tuning the solver

All options are keyword arguments to
[`solve_fe_coefficients`](../api.md#felcu.solve_fe_coefficients).  The
defaults suit most problems; this page explains when to change them.

## Accuracy

`target_error` (default `1e-12`)
:   The target tolerance \(\epsilon_{\rm t}\) the fit must meet.  Tightening it increases \(m\) roughly
    logarithmically for smooth targets such as \(e^{x}\), and much faster for
    inversion with a large condition number.

`error_metric` (default `"linf"`)
:   `"linf"` bounds \(\epsilon_m = \|g - g_m\|_{L^\infty(\Omega)}\), which
    bounds the block-encoding error \(\|f(\mathcal H) - g_m(\mathcal G)\|\).  `"l2"` bounds the
    integrated error and is looser.  Aliases: `"inf"`, `"max"`,
    `"infinity"`, `"l_inf"`, `"l-infinity"`, `"2"`.

`val_n` (default `4001`)
:   Validation points used to measure the error.  Increase it if you see the
    error rise when checked on a finer grid.

## Truncation order

`m`
:   Fix a single order and skip the sweep.

`m_values`
:   Candidate orders to try, as a sequence or a single integer.  Defaults to
    [`default_m_grid()`](../api.md#felcu.default_m_grid): 32 values from 4 to
    256.

`selection` (default `"smallest_alpha"`)
:   `"smallest_alpha"` or `"smallest_m"`; see
    [Choosing m](method.md#choosing-m).  `"smallest_m"` stops the sweep at
    the first feasible order, so it is usually faster.

!!! tip "The sweep sets both the cost and the answer"
    With `smallest_alpha`, every order in `m_values` is solved; the default
    sweep takes a few seconds.  `smallest_m` stops at the first feasible
    order and usually takes well under a second.  Because \(\alpha\) tends
    to keep falling as \(m\) grows, the upper end of `m_values` often decides
    which order `smallest_alpha` returns, so cap it if circuit depth
    matters.

## Regularisation

`lambdas`
:   Initial grid of regularisation strengths \(\lambda = \sqrt{2\pi\gamma}\),
    where \(\gamma\) is the weight of the \(H^w_r\) penalty (see
    [The regularised problem](method.md#the-regularised-problem)).  Defaults to
    [`default_lambda_grid()`](../api.md#felcu.default_lambda_grid): 12
    log-spaced values from \(10^{-16}\) to \(1\).  The solver expands beyond
    it and bisects, so the grid only needs to bracket the answer roughly.

`sobolev_w` (default `1`)
:   Order \(w\) of the \(H_r^w\) regulariser, which weights the \(k\)-th
    Fourier mode by \(1 + (r|k|)^{2w}\).  Must exceed \(1/2\); need not be
    an integer.  The paper's bounds cover \(1/2 < w \le 1\).

`sobolev_r` (default: problem-dependent)
:   Scale \(r\) of the \(H_r^w\) regulariser, with \(0 < r \le 1\).  When
    omitted it comes from
    [`default_sobolev_r`](../api.md#felcu.default_sobolev_r), the paper's
    scale-adapted choices: \(r_{\rm id} = 1\) for action,
    \(r_{\exp} = (1 + 1/\tau)^{-1}\) for exponentiation, and
    \(r_{\rm def} = \min\{1, 2\pi/(\kappa+1)\}\) or
    \(r_{\rm indef} = \min\{1, \pi/(\kappa+1)\}\) for shifted or unshifted
    inversion, and \(r_{\rm sqrt} = \kappa^{-(2w-1)/(2w)}\) for the square
    root (see
    [The Sobolev penalty](method.md#the-sobolev-penalty)).
    The value used is reported as `result.sobolev_r`.

`order_fit` (default `1`)
:   Number of derivatives matched in the fit.  `1` fits in
    \(H^1(\Omega)\), as in the paper; `0` falls back to \(L^2(\Omega)\).  See
    [The fitting norm](method.md#the-fitting-norm).

!!! note
    Passing `lambdas=[0.0]` does **not** disable regularisation: if
    \(\lambda = 0\) meets the target, the solver still expands upwards
    to find a larger feasible \(\lambda\).

## Fitting grid

`fit_oversample` (default `8`)
:   Gauss–Legendre nodes per unknown.

`eta`
:   The domain extension factor \(\eta > 1\); see
    [The extension parameter](problems.md#the-extension-parameter-eta).

## Spectral input

`spectral_kind` (default `"auto"`)
:   `"eigenvalues"`, `"singular_values"`, or `"auto"`; see
    [Spectral information](problems.md#spectral-information).

`hermitian_tol` (default `1e-12`)
:   Tolerance for deciding whether `matrix` is Hermitian.

## Logging

`felcu` logs its progress through the standard `logging` module under the
`felcu` logger:

```python
import logging

logging.basicConfig(level=logging.INFO)  # one line per m
logging.getLogger("felcu").setLevel(logging.DEBUG)  # every lambda step
```
