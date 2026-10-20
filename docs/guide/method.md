# Method

`felcu` computes the **Sobolev-regularised Fourier extensions** of Sec. V of
the paper: a least-squares Fourier fit in \(H^1\) on the fitted set
\(\Omega\), with a weighted fractional Sobolev penalty over the whole
period.  This page describes what the solver does for each truncation
order \(m\), and how it chooses between them.  Symbols are listed on the
[Notation](notation.md) page.

## The series

For truncation order \(m\), the Fourier extension is

\[
g_m(x) = \sum_{k=-m}^{m} c_k e^{ikx} = a_0 + \sum_{k=1}^{m} \bigl[ a_k \sin(kx) + b_k \cos(kx) \bigr],
\]

with \(c_0 = a_0\) and \(c_{\pm k} = (b_k \mp i a_k)/2\), restricted to the
basis that suits the problem (see [Bases](problems.md#bases)).  It lies in
\(\mathcal F_m = \operatorname{span}\{e^{-imx}, \dots, e^{imx}\}\).

## The regularised problem

The coefficients solve the paper's regularised problem

\[
\tilde g_m \in \operatorname*{arg\,min}_{h \in \mathcal F_m}
\Bigl\{ \|g - h\|^2_{H^1(\Omega)} + \gamma \|h\|^2_{H^w_r(\mathbb T)} \Bigr\},
\]

where \(g\) is the scaled target from
[the table of constructions](problems.md#the-scaled-variable).

### The fitting norm

\(\|g - h\|^2_{H^1(\Omega)}\) is approximated by Gauss–Legendre quadrature
with nodes \(x_j\) and weights \(w_j\) on \(\Omega\):

\[
\|g - h\|^2_{H^1(\Omega)} \approx \sum_j w_j \bigl( |g(x_j) - h(x_j)|^2 + |g'(x_j) - h'(x_j)|^2 \bigr),
\]

using the analytic derivatives of both \(g\) and the Fourier modes.  The
\(H^1\) fit controls the uniform error through the Sobolev embedding
\(\|u\|_{L^\infty(\Omega_\eta)} \le C_\eta \|u\|_{H^1(\Omega_\eta)}\).
`order_fit = 0` drops the derivative term and falls back to
\(L^2(\Omega)\).

- **Quadrature.**  The number of nodes is `fit_oversample` times the number
  of unknowns (with a floor of 48 for one interval, 32 per interval for two).
  Large node counts are split into panels of at most 120 points.

### The Sobolev penalty

The regulariser is the paper's weighted fractional Sobolev norm,

\[
\|h\|^2_{H^w_r(\mathbb T)} = 2\pi \sum_{k \in \mathbb Z} \bigl(1 + (r|k|)^{2w}\bigr) |\hat h_k|^2 ,
\qquad 0 < r \le 1,\quad w > \tfrac12 .
\]

In the augmented least-squares system this is the diagonal
\(\sqrt{\gamma}\, W_{r,w}\), with
\(W_{r,w} = \sqrt{2\pi}\operatorname{diag}\bigl(\sqrt{1 + (r|k|)^{2w}}\bigr)\).
`felcu` works with the real coefficients, for which
\(|c_k|^2 + |c_{-k}|^2 = (|a_k|^2 + |b_k|^2)/2\), so each sine and cosine
coefficient is penalised by

\[
p_k = \lambda \sqrt{\tfrac12\bigl(1 + (r k)^{2w}\bigr)},
\qquad \lambda = \sqrt{2\pi\gamma},
\]

and the constant term by \(\lambda\).  The penalty acts over the whole
period, so the extension region is controlled as well as \(\Omega\);
the paper's \(\ell^1\)-boundedness lemma turns it into the bound
\(\|\mathbf c\|_1 \le C_w r^{-1/2} \|g_m\|_{H^w_r(\mathbb T)}\) on the
subnormalisation.

- \(w\) (`sobolev_w`, default \(1\), must satisfy \(w > 1/2\)) sets how fast
  the weight grows with frequency.  \(w = 1\) gives
  \(\|h\|^2_{H^1_r} = \|h\|^2_{L^2} + r^2\|h'\|^2_{L^2}\).  The paper's
  bounds cover \(1/2 < w \le 1\); smaller \(w\) weakens the derivative
  penalty at the cost of a larger constant \(C_w\).
- \(r\) (`sobolev_r`, must satisfy \(0 < r \le 1\)) should match the
  shortest length scale \(g\) has to resolve, so that the derivative
  contribution sits on the same scale as the function itself.  By default
  it is chosen per problem by
  [`default_sobolev_r`](../api.md#felcu.default_sobolev_r).

The defaults are the scale-adapted choices of Sec. VI:

| Problem | Default \(r\) |
|---|---|
| action | \(r_{\rm id} = 1\) |
| exponentiation | \(r_{\exp} = (1 + 1/\tau)^{-1}\), i.e. \((1 + \eta\delta/\pi)^{-1}\) when shifted |
| inversion, shifted (definite) | \(r_{\rm def} = \min\{1,\ 2\pi/(\kappa + 1)\}\) |
| inversion, unshifted (two intervals) | \(r_{\rm indef} = \min\{1,\ \pi/(\kappa + 1)\}\) |
| square root | \(r_{\rm sqrt} = \kappa^{-(2w-1)/(2w)}\), i.e. \(\kappa^{-1/2}\) for \(w = 1\) |
| logarithm | \(r_{\rm def} = \min\{1,\ 2\pi/(\kappa + 1)\}\) (not from the paper) |

For unshifted exponentiation \(1/\tau = \eta\|\mathcal H\|/\pi\), so the
spectral norm takes the place of the half-width \(\delta\).  The paper
does not treat the logarithm: `felcu` uses the gap to the branch point, as
for inversion, since \(\log x\) has the same \(1/x\) derivative scale
near zero.  In tests it gave subnormalisations within a few percent of the
best of several alternatives (\(r = 1\), \(\kappa^{-1/2}\), \(1/\kappa\)).
Without spectral information the default is \(r = 1\).

### The constant baseline

Before fitting, `felcu` subtracts a constant baseline \(\beta = g(0)\)
(\(\mu\), \(1/\mu\), \(e^{\mu}\), \(\sqrt{\mu}\) or \(\log\mu\) for the
shifted problems, \(1\) for unshifted exponentiation, and \(0\) for the
odd problems), solves for the correction, and adds
\(\beta\) back to \(a_0\).  The penalty therefore acts on \(h - \beta\)
rather than \(h\), so the exactly-known constant part \(c_0 I\) is not
shrunk towards zero.  For the odd problems \(\beta = 0\) and this is the
paper's problem unchanged.

## Choosing \(\gamma\)

Regularisation trades accuracy for smaller coefficients: the regularised
error converges down to an \(O(\sqrt\gamma)\) floor.  As in the paper,
`felcu` tunes \(\sqrt\gamma\) directly, through
\(\lambda = \sqrt{2\pi\gamma}\), so that the fit attains the target
tolerance \(\epsilon_{\rm t}\) (`target_error`).  For each \(m\) it looks
for the **largest** \(\lambda\) whose fit still meets \(\epsilon_{\rm t}\):

1. Solve with \(\lambda = 0\).  If that already misses the target, the order
   \(m\) is infeasible and is skipped.
2. Scan the `lambdas` grid in increasing order, keeping the largest value
   that still meets the target and stopping at the first that does not.
3. If every grid value meets the target, keep multiplying \(\lambda\) by 10
   (up to 40 times) until one fails.
4. Bisect between the last feasible and first infeasible \(\lambda\)
   (30 steps, in log space when the lower end is positive).

The result is a fit that sits just inside the target error, with as much
smoothing as that allows.  `result.lambda_used` reports the value of
\(\lambda = \sqrt{2\pi\gamma}\) chosen.

The search is cheap because the penalty is \(\lambda D\) with \(D\)
diagonal and fixed.  Substituting \(y = Dc\) turns every \(\lambda > 0\)
solve into ridge regression on \(AD^{-1}\), where \(A\) is the fit matrix.
One SVD of \(AD^{-1}\) per \(m\) then gives each solve in a few
matrix–vector products.  The \(\lambda = 0\) solve is a separate
least-squares solve on \(A\).

## Errors

Errors are measured on a separate uniform validation grid of `val_n`
points (split between the two intervals for a double domain):

- \(L^\infty\): \(\epsilon_m = \|g - g_m\|_{L^\infty(\Omega)}\), the maximum
  over the grid.  This bounds the block-encoding error,
  \(\|f(\mathcal H) - g_m(\mathcal G)\| \le \epsilon_m\);
- \(L^2\): \(\|g - g_m\|_{L^2(\Omega)}\) by the trapezoid rule.

`error_metric` (default `"linf"`) chooses which one is compared with
`target_error`.  Both are always reported, as `achieved_l2_error` and
`achieved_linf_error`.

## The subnormalisation

The LCU block encoding of \(g_m(\mathcal G)\) has subnormalisation

\[
\alpha_m = \sum_{k=-m}^{m} |c_k| = |a_0| + \sum_{k=1}^{m} \sqrt{a_k^2 + b_k^2}
\]

(for real coefficients), and its success probability scales as
\(1/\alpha_m^2\), so smaller is better.  No extension can beat
\(\alpha_m \ge \|g\|_{L^\infty(\Omega)} - \epsilon_m\), and for these
problems the Sobolev-regularised fits reach the optimal scale
\(\alpha_m = O(\|f(\mathcal H)\|)\).  `result.alpha` reports it.

## Choosing \(m\)

The solver repeats the above for the orders in `m_values` (by default a
quadratically spaced grid from 4 to 256), in increasing order, and selects
among those that meet the target:

- `selection="smallest_alpha"` (default): the smallest \(\alpha_m\); ties
  broken by \(m\).  Every order in the grid is solved.  This is the paper's
  "lowest-\(\alpha\) feasible solution obtained up to each \(m\)".
- `selection="smallest_m"`: the shortest series.  The sweep stops at the
  first order that meets the target, so `result.all_results` only holds the
  orders up to it.

\(\alpha_m\) often keeps falling slowly as \(m\) grows, so `smallest_alpha`
tends to pick a large order from the grid.  Cap `m_values` if circuit depth
(which grows with \(m\)) matters as well as \(\alpha\).

If no order meets the target, the candidate with the smallest error is
returned and `result.converged` is `False`.
