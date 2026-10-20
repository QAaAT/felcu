# Tuning the solver

All options are keyword arguments to
[`solve_fe_coefficients`](../api.md#felcu.solve_fe_coefficients).  The
defaults suit most problems; this page explains when to change them.

## Accuracy

`target_error` (default `1e-12`)
:   The error the fit must meet.  Tightening it increases \(m\) roughly
    logarithmically for smooth targets such as \(e^{x}\), and much faster for
    inversion with a large condition number.

`error_metric` (default `"linf"`)
:   `"linf"` bounds the worst-case pointwise error on the fitted interval,
    which is usually what an operator-norm bound needs.  `"l2"` bounds the
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

`selection` (default `"smallest_m"`)
:   `"smallest_m"` or `"smallest_alpha"`; see
    [Choosing m](method.md#choosing-m).

!!! tip "The sweep is the main cost"
    Every order in `m_values` is solved, including the \(\lambda\) search, and
    the cost grows with \(m\).  The default sweep can take tens of seconds.
    If you know roughly what order you need, a short range such as
    `m_values=range(4, 41, 2)` is many times faster.

## Regularisation

`lambdas`
:   Initial grid of regularisation strengths.  Defaults to
    [`default_lambda_grid()`](../api.md#felcu.default_lambda_grid): 12
    log-spaced values from \(10^{-16}\) to \(1\).  The solver expands beyond
    it and bisects, so the grid only needs to bracket the answer roughly.

`k_order` (default `1`)
:   Sobolev order \(s\) of the penalty \(\lambda(1 + k^2)^{s/2}\).

`order_fit` (default `1`)
:   Number of derivatives matched in the fit.

!!! note
    Passing `lambdas=[0.0]` does **not** disable regularisation: if
    \(\lambda = 0\) meets the target, the solver still expands upwards
    to find a larger feasible \(\lambda\).

## Fitting grid

`fit_oversample` (default `8`)
:   Gauss–Legendre nodes per unknown.

`eta`
:   The extension parameter; see
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
logging.basicConfig(level=logging.INFO)   # one line per m
logging.getLogger("felcu").setLevel(logging.DEBUG)  # every lambda step
```
