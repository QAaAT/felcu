# Method

`felcu` computes a **Sobolev-regularised Fourier extension**: a
least-squares Fourier fit on part of a period, with a penalty that grows
with frequency.  This page describes what the solver does for each
truncation order \(m\), and how it chooses between them.

## The series

For truncation order \(m\), the approximation is

\[
g(x) = a_0 + \sum_{k=1}^{m} \bigl( a_k \sin(kx) + b_k \cos(kx) \bigr),
\]

restricted to the basis that suits the problem (see
[Bases](problems.md#bases)).  Write \(c\) for the vector of unknown
coefficients.

## The fit

Let \(x_j\) and \(w_j\) be Gauss–Legendre nodes and weights on the fitted
domain, and \(f\) the scaled target from
[the table of constructions](problems.md#the-scaled-variable).  The
coefficients minimise

\[
\sum_{d=0}^{D} \sum_j w_j \,\bigl| g^{(d)}(x_j) - f^{(d)}(x_j) \bigr|^2
\;+\; \bigl\| P c \bigr\|^2,
\]

where \(D\) is `order_fit` and \(P\) is the diagonal Sobolev penalty
described below.

- **Fitting norm.**  Each inner sum is a quadrature approximation of the
  \(L^2\) norm on the fitted domain, so the fit term is the \(H^D\)
  Sobolev norm of the residual \(g - f\).  The default `order_fit = 1`
  fits in \(H^1\), matching first derivatives as well as values, which
  suppresses oscillation between nodes.  `order_fit = 0` falls back to
  plain \(L^2\).
- **Quadrature.**  The number of nodes is `fit_oversample` times the number
  of unknowns (with a floor of 48 for one interval, 32 per interval for two).
  Large node counts are split into panels of at most 120 points.
- **Baseline.**  A simple baseline (for example \(1\) for \(e^{x}\), \(\mu\)
  for shifted action) is subtracted before fitting and added back to
  \(a_0\) afterwards, so the penalty acts on the correction only.

The system is solved as a stacked least-squares problem with
`numpy.linalg.lstsq`.

## The Sobolev penalty

The regulariser is the \(H_r^w\) norm of the correction \(h\) (the series
minus its baseline).  By Parseval,

\[
\|h\|_{H_r^w}^2 = \sum_{k} (1 + r|k|)^{2w}\, |\hat h_k|^2,
\]

so the diagonal penalty \(P\) weights the coefficient of the \(k\)-th
harmonic (both its sine and cosine term) by

\[
p_k = \lambda\,(1 + r k)^{w},
\]

and the constant term by \(\lambda\).  Then \(\|Pc\|^2\) is, up to a
constant factor, \(\lambda^2 \|h\|_{H_r^w}^2\) over the whole period, so
the extension region is controlled as well as the fitted domain.

- \(w\) (`sobolev_w`, default \(1\), must satisfy \(w > 1/2\)) sets how
  fast the weight grows with frequency: roughly, how many derivatives are
  penalised.
- \(r\) (`sobolev_r`, must satisfy \(0 < r \le 1\)) sets the frequency
  scale: the weight stays near 1 for \(|k| \ll 1/r\) and grows like
  \((r|k|)^{w}\) beyond it.  By default it is chosen per problem by
  [`default_sobolev_r`](../api.md#felcu.default_sobolev_r).

!!! note
    `default_sobolev_r` is currently a placeholder that returns \(r = 1\)
    for every problem.  It already receives the spectral interval, and the
    half-width, centre, radius and condition number are available inside
    it for a problem-specific rule.

## Choosing \(\lambda\)

Regularisation trades accuracy for smaller coefficients.  For each \(m\),
`felcu` looks for the **largest** \(\lambda\) whose fit still meets
`target_error`:

1. Solve with \(\lambda = 0\).  If that already misses the target, the order
   \(m\) is infeasible and is skipped.
2. Scan the `lambdas` grid in increasing order, keeping the largest value
   that still meets the target and stopping at the first that does not.
3. If every grid value meets the target, keep multiplying \(\lambda\) by 10
   (up to 40 times) until one fails.
4. Bisect between the last feasible and first infeasible \(\lambda\)
   (30 steps, in log space when the lower end is positive).

The result is a fit that sits just inside the target error, with as much
smoothing as that allows.  `result.lambda_used` reports the value chosen.

## Errors

Errors are measured on a separate uniform validation grid of `val_n`
points (split between the two intervals for a double domain):

- \(L^2\): \(\bigl(\int |g - f|^2\,dx\bigr)^{1/2}\) by the trapezoid rule;
- \(L^\infty\): \(\max |g - f|\) over the grid.

`error_metric` (default `"linf"`) chooses which one is compared with
`target_error`.  Both are always reported, as `achieved_l2_error` and
`achieved_linf_error`.

## The LCU normalisation

An LCU circuit implementing \(g\) has normalisation

\[
\alpha = |a_0| + \sum_{k=1}^{m} \sqrt{a_k^2 + b_k^2},
\]

and its success probability scales as \(1/\alpha^2\), so smaller is better.
Regularisation tends to reduce \(\alpha\) by damping large, cancelling
high-frequency coefficients.  `result.alpha` reports it.

## Choosing \(m\)

The solver repeats the above for every \(m\) in `m_values` (by default a
quadratically spaced grid from 4 to 256), then selects among the orders
that meet the target:

- `selection="smallest_m"` (default): the shortest series; ties broken by
  \(\alpha\).
- `selection="smallest_alpha"`: the smallest normalisation; ties broken by
  \(m\).

If no order meets the target, the candidate with the smallest error is
returned and `result.converged` is `False`.
