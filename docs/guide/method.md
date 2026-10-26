# Method

`felcu` implements the paper's three coefficient-selection strategies,
chosen with `method=`:

| `method` | Paper | Convergence | Subnormalisation \(\alpha_\infty\) |
|---|---|---|---|
| `"sobolev"` (default) | Sec. V | near-exponential, down to \(\epsilon_{\rm t}\) | \(O(\|f(\mathcal H)\|)\) |
| `"reflected"` | Sec. III | algebraic, \(O(m^{-1})\) | \(\|f(\mathcal H)\|\) (optimal) for convex endpoint-maximising \(f\) |
| `"arcsine_taylor"` | Sec. IV | exponential, \(O(m^{-3/2}\sin^m(\pi/\eta))\) | \(\lvert f(\mu \pm \eta\delta/2)\rvert\) for convex endpoint-maximising \(f\) |

The **Sobolev-regularised extension** is a least-squares Fourier fit in
\(H^1\) on the fitted set \(\Omega\), with a weighted fractional Sobolev
penalty over the whole period.  It is the most flexible: it handles the
two-interval fit of indefinite inversion, and reaches tight tolerances with
near-optimal \(\alpha\).  The two **explicit** methods instead prescribe
the periodic extension outright, so every order \(m\) has a closed-form
series and no search over regularisation strengths.  They are described in
[Reflected extensions](#reflected-extensions) and
[Arcsine-Taylor](#arcsine-taylor) below; the rest of this page, up to
there, describes the Sobolev-regularised solver.  Symbols are listed on
the [Notation](notation.md) page.

```python
solve_fe_coefficients("exponentiation", sigma_min=0.1, sigma_max=2.0)  # Sobolev
solve_fe_coefficients("exponentiation", method="reflected", sigma_min=0.1, sigma_max=2.0, target_error=1e-2)
solve_fe_coefficients("exponentiation", method="arcsine_taylor", sigma_min=0.1, sigma_max=2.0)
```

`result.method` records which one was used.

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
| exponentiation | \(r_{\exp} = (1 + 1/\tau)^{-1}\), i.e. \((1 + \eta\delta/\pi)^{-1}\) |
| inversion | \(\min\{1,\ \tau \min_{\lambda \in \operatorname{spec}} \lvert\lambda\rvert\}\), the distance from the fitted set to the pole |
| square root | \(r_{\rm sqrt} = \kappa^{-(2w-1)/(2w)}\), i.e. \(\kappa^{-1/2}\) for \(w = 1\) |

For inversion at the default \(\eta\) this is the paper's
\(r_{\rm def} = \min\{1,\ 2\pi/(\kappa + 1)\}\) for a definite interval and
\(r_{\rm indef} = \min\{1,\ \pi/(\kappa + 1)\}\) for singular values.

### The constant baseline

Before fitting, `felcu` subtracts a constant baseline \(\beta = g(0)\)
(\(\mu\), \(1/\mu\), \(e^{\mu}\), or \(\sqrt{\mu}\) on a single interval,
and \(0\) for singular values and two-interval fits), solves for the correction, and adds
\(\beta\) back to \(a_0\).  The penalty therefore acts on \(h - \beta\)
rather than \(h\), so the exactly-known constant part \(c_0 I\) is not
shrunk towards zero.  For \(\beta = 0\) this is the
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

The same errors, subnormalisation and choice of \(m\) apply to the explicit
methods below; only the coefficients for each \(m\) are computed
differently.

## Reflected extensions

The reflected extension (Sec. III of the paper) extends \(g\) beyond the
fitted interval by reflecting it about \(x = \pm\pi/2\):

\[
F(x) = g(\arcsin(\sin x)),
\]

a continuous \(2\pi\)-periodic function equal to \(g\) on
\([-\pi/2, \pi/2]\).  For the identity it is the triangle wave.  \(g_m\) is
the truncated Fourier series of \(F\).  Since \(F(\pi - x) = F(x)\), only
odd sine and even cosine modes survive, and `felcu` computes each from
Gauss–Legendre quadrature of \(g\) over \([-\pi/2, \pi/2]\).

When \(f\) is convex and maximised in magnitude at a spectral endpoint
(the paper's convex endpoint-reflection condition: action, exponentiation
and definite inversion),
every Fourier mode is phase-aligned at that endpoint, so the
subnormalisation is optimal,

\[
\alpha_m = \|f(\mathcal H)\| - \epsilon_m ,
\]

with the error \(\epsilon_m\) attained at the endpoint.  The price is the
kink in \(F\) at \(\pm\pi/2\): the error only falls as \(O(m^{-1})\), so the
default `target_error=1e-12` is out of reach and the sweep returns its most
accurate order with `converged=False`.  Pass a loose target such as `1e-2`.
Concave functions such as \(\sqrt x\) still get a bounded, but not optimal,
\(\alpha_m\).

- **\(\eta\).**  The default \(\eta = 2\) maps the spectral endpoints onto
  the reflection points, which is what makes \(\alpha\) optimal.
  \(\eta > 2\) leaves the endpoints inside the smooth part of \(F\), which
  speeds convergence up to \(O(m^{-2})\) but raises \(\alpha\).  \(\eta < 2\)
  is rejected.

## Arcsine-Taylor

The arcsine-Taylor construction (Sec. IV of the paper) changes variable to
\(z = \sin x\), which is invertible on \((-\pi/2, \pi/2)\), and truncates
the Taylor series

\[
g(\arcsin z) = \sum_{k=0}^{\infty} d_k z^k
\quad\Longrightarrow\quad
g_m(x) = \sum_{k=0}^{m} d_k \sin^k x .
\]

Each \(\sin^k x\) is a trigonometric polynomial of degree \(k\) with
coefficient \(\ell^1\)-norm \(1\), so \(g_m\) has degree \(m\) and

\[
\alpha_m \le \sum_{k=0}^{m} |d_k|, \qquad
\epsilon_m \le \sum_{k > m} |d_k| \rho_\eta^k, \qquad
\rho_\eta = \sin(\pi/\eta) .
\]

Both bounds are independent of the target error, and for \(\eta > 2\)
the error falls exponentially in \(m\).  For convex endpoint-maximising
\(f\), \(\alpha_m\) approaches \(|f(\mu \pm \eta\delta/2)|\): \(f\) at the
*enlarged* endpoint, since \(x = \pm\pi/2\) corresponds to
\(\lambda = \mu \pm \eta\delta/2\).  Larger \(\eta\) therefore converges
faster but costs subnormalisation.

`felcu` computes \(d_0, \dots, d_m\) exactly (up to rounding) by truncated
power-series arithmetic on \(u(z) = \mu + \arcsin(z)/\tau\): \(u\) itself
for action, and the series exponential, reciprocal or square root for the
other problems.  It then evaluates \(\sum_k d_k \sin^k x\) by Horner's rule
on \(2m + 2\) equispaced points, and reads off the Fourier coefficients with
an FFT, which is exact for a trigonometric polynomial of degree \(m\).

!!! note "Why an FFT rather than the binomial expansion"
    Expanding each power with \(\sin x = (e^{ix} - e^{-ix})/(2i)\) gives
    the coefficients in closed form,

    \[
    \sin^k x = \frac{1}{(2i)^k} \sum_{j=0}^{k} (-1)^j \binom{k}{j} e^{i(k-2j)x}
    \quad\Longrightarrow\quad
    c_n = \sum_{\substack{|n| \le k \le m \\ k \equiv n \ (\mathrm{mod}\ 2)}}
    \frac{(-1)^{(k-n)/2}}{(2i)^k} \binom{k}{(k-n)/2}\, d_k .
    \]

    The FFT returns the same numbers.  Sampling \(g_m\) at \(N = 2m + 2\)
    equispaced points and taking the discrete Fourier transform gives
    \(c_n\) plus its aliases \(c_{n \pm N}, c_{n \pm 2N}, \dots\), and these
    all vanish because \(|n \pm N| > m\).  The two routes therefore agree
    exactly in exact arithmetic and differ only in rounding, which is of
    order \(\varepsilon \sum_k |d_k|\) for both.  In tests against
    exact rational arithmetic the FFT was as accurate as the binomial sum,
    often slightly more so, and it needs no binomial table.

- **\(\eta\).**  The method requires \(\eta > 2\).  The default is \(3\)
  (\(\rho_3 \approx 0.866\)) for action and exponentiation, as in the
  paper.  For inversion and square root, \(f\) must be analytic on
  \([\mu - \eta\delta/2, \mu + \eta\delta/2]\), so
  \(2 < \eta < 2\mu/\delta = 2(\kappa+1)/(\kappa-1)\); the default is the
  midpoint \(1 + \mu/\delta\), capped at \(3\).  Near the upper bound the
  convergence factor improves but \(\alpha\) grows; for a large condition
  number \(\kappa\) the admissible range shrinks towards \(2\) and
  convergence slows, which is where the Sobolev method does better.

## Restrictions of the explicit methods

- **Two-interval inversion is not supported.**  On an indefinite interval or
  on singular values the pole lies between the fitted intervals, and neither
  extension can pass it.  Use `method="sobolev"`.
- **The extension must avoid the singularity.**  Both methods evaluate \(f\)
  on \([\mu - \eta\delta/2, \mu + \eta\delta/2]\), so for inversion and
  square root `felcu` rejects \(\eta \ge 2|\mu|/\delta\).
- **No regularisation options.**  `lambdas`, `sobolev_w`, `sobolev_r`,
  `order_fit` and `fit_oversample` only apply to the Sobolev method, and
  passing them with an explicit method raises an error.  The result's
  `lambda_used`, `sobolev_r` and `sobolev_w` are `None`.
