# Problems and constructions

Every fit is described by three choices: the **problem** (which \(f\)),
whether the construction is **shifted**, and where the **spectral interval**
comes from.

## Spectral information

A spectral interval is required.  Give it either as bounds or as a matrix:

```python
solve_fe_coefficients("inversion", sigma_min=0.5, sigma_max=3.0)
solve_fe_coefficients("inversion", matrix=A)
```

Passing neither, or both, raises an error.  With `matrix=`, `felcu`
computes eigenvalues or singular values itself.

The construction fixes which interval is read: the **shifted**
construction fits an eigenvalue interval, and the **unshifted** one a
singular-value interval.  `shifted` chooses between them:

| Problem | `shifted=None` (default) | Also available |
|---|---|---|
| `action` | `True` (eigenvalues); `False` (singular values) for a non-Hermitian matrix | the other one |
| `inversion` | `False` (singular values) | `True` (eigenvalues, must be definite) |
| `exponentiation` | `True` (eigenvalues) | none |
| `square_root` | `True` (eigenvalues, must be positive definite) | none |

`sigma_min`/`sigma_max` are read as eigenvalue or singular-value bounds to
match.  The shifted construction requires a Hermitian matrix (tested to
`hermitian_tol`).

## The scaled variable

Following Sec. II of the paper, the fit is carried out in a scaled
variable \(x\) on the fitted interval \(\Omega_\eta = [-\pi/\eta, \pi/\eta]\),
strictly inside one period \(\mathbb T = [-\pi, \pi)\).  This is what makes
it a Fourier *extension*: the series \(g_m\) only has to match the target
on part of the period, so a smooth non-periodic function can still be
fitted to high accuracy.

The scale \(\tau\) maps the spectrum onto the fitted set, giving the scaled
matrix \(\mathcal G = \tau(\mathcal H - \mu I)\) and the fitted function

\[
g(x) = f\!\left(\mu + \frac{x}{\tau}\right), \qquad f(\mathcal H) = g(\mathcal G),
\]

with spectral midpoint \(\mu\) and half-width \(\delta\) (see
[Notation](notation.md)).  The shifted constructions are exactly the
paper's \(\tau = \pi/(\eta\delta)\).  Unshifted (singular-value)
constructions take \(\mu = 0\) and scale by \(\Sigma_{\max}\) instead:

| Problem | Construction | Fitted \(g(x)\) | \(\tau\) | Basis | Fitted set |
|---|---|---|---|---|---|
| action | shifted | \(\mu + x/\tau\) | \(\pi/(\eta\delta)\) | constant + sine | \(\Omega_\eta\) |
| action | unshifted | \(x/\tau\) | \(\pi/(\eta\Sigma_{\max})\) | odd sine | \(\Omega_\eta\) |
| inversion | shifted | \((\mu + x/\tau)^{-1}\) | \(\pi/(\eta\delta)\) | full | \(\Omega_\eta\) |
| inversion | unshifted | \(\tau/x\) | \(\pi/(\eta\Sigma_{\max})\) | odd sine | \(\Omega^{\rm indef}_{\eta,\kappa}\) |
| exponentiation | shifted | \(e^{\mu + x/\tau}\) | \(\pi/(\eta\delta)\) | full | \(\Omega_\eta\) |
| square root | shifted | \(\sqrt{\mu + x/\tau}\) | \(\pi/(\eta\delta)\) | full | \(\Omega_\eta\) |

The block encoding then implements \(g_m(\mathcal G) = \sum_k c_k e^{ik\mathcal G}\),
where each \(e^{ik\mathcal G} = e^{-ik\tau\mu} e^{ik\tau\mathcal H}\) is a
Hamiltonian simulation of \(\mathcal H\) up to a known phase.

### Bases

- **odd sine**: \(a_0 = 0\) and \(b_k = 0\); only sine terms.  Used by the
  unshifted constructions, which act on singular values and so need an odd
  target.
- **constant + sine**: a constant plus sine terms.
- **full**: constant, sine and cosine terms.

Restricting the basis halves the number of unknowns and guarantees the
right symmetry exactly.

### Unshifted inversion: the two-interval fit

\(x^{-1}\) has a pole at zero, and an indefinite (or dilated, see below)
spectrum lies on both sides of it.  Unshifted inversion therefore fits
\(g(x) = \tau/x\) on the disconnected set of Sec. VI C of the paper,

\[
\Omega^{\rm indef}_{\eta,\kappa} = [-x_{\max}, -x_{\min}] \cup [x_{\min}, x_{\max}],
\qquad x_{\max} = \frac{\pi}{\eta},\quad x_{\min} = \frac{\pi}{\kappa\eta},
\]

and leaves the gap around the pole unconstrained.  `result.domain` is
`"double"` and `result.x_min` holds \(x_{\min}\).  This is the fit used
for a general \(A\) through its Hermitian dilation, whose spectrum is
\(\pm\operatorname{sv}(A)\).

## Shifted versus unshifted

The shifted construction centres the spectrum before fitting.  For an
interval far from zero, such as \([2, 5]\), it only covers the half-width
\(\delta = 1.5\), and the constant part \(c_0 I\) carries \(\mu\) exactly;
the shift only adds the known phase \(e^{-ik\tau\mu}\) to each Hamiltonian
simulation.  Every eigenvalue interval is fitted this way.

The unshifted construction is kept for singular values, where the target
must be odd and so cannot be centred.  It covers a non-Hermitian matrix,
for which there is no real eigenvalue interval, and indefinite inversion
through the two-interval fit above.  For the same interval it has to cover
\([-\Sigma_{\max}, \Sigma_{\max}]\), so it usually costs more \(m\) and
\(\alpha_m\) than a shifted fit would.

Shifted inversion requires a *definite* interval (both bounds the same sign,
neither zero); negative-definite matrices work too.

## Square root

\(\sqrt{x}\) has a branch point at zero and is only defined (as a real
function) for \(x > 0\).  It follows the paper's
treatment of the matrix square root (Sec. VI D): \(\mathcal H\) must be
positive definite, \(0 < \lambda_{\min} \le \lambda_{\max}\), and the fit
uses the shifted construction with \(\eta_{\rm def}\), which places the
branch point on the periodic boundary rather than near the fitted
endpoint.  `shifted=False` raises an error, since an unshifted interval
\([-\|\mathcal H\|, \|\mathcal H\|]\) would contain the branch point.

\(\sqrt x\) is concave, so it does not satisfy the convex
endpoint-reflection condition and the subnormalisation is bounded but not
optimal: the paper shows
\(\alpha_m = O(\kappa^{(2w-1)/(4w)}\sqrt{\lambda_{\max}})\), an extra
\(O(\kappa^{1/4})\) for \(w = 1\).

## The extension parameter \(\eta\)

\(\eta > 1\) sets how much of the period is left free.  Larger \(\eta\)
gives the series more room to turn around, but packs the spectrum into a
smaller interval.  The defaults are the paper's choices for
Sobolev-regularised extensions (Sec. VI):

| Problem | Default \(\eta\) | Reason |
|---|---|---|
| action, exponentiation | \(2\) | entire \(f\); optimal for reflected extensions |
| inversion, shifted | \(\eta_{\rm def} = (\kappa + 1)/(\kappa - 1)\) | puts the pole \(x_{\rm pole} = -\tau\mu\) at the periodic boundary |
| square root | \(\eta_{\rm def} = (\kappa + 1)/(\kappa - 1)\) | puts the branch point at the periodic boundary |
| inversion, unshifted | \(\eta_{\rm indef} = 1 + 1/\kappa\) | inner gap to the pole equals the outer gap to the boundary |

Here \(\kappa = \Sigma_{\max}/\Sigma_{\min}\) for unshifted inversion and
\(\kappa = \max|\lambda|/\min|\lambda|\) for the definite interval of shifted
inversion and square root (\(\eta = 2\) if \(\kappa = 1\)).  Override with `eta=`, or query
the default with [`default_eta`](../api.md#felcu.default_eta).

## Problem name aliases

| Canonical | Also accepted |
|---|---|
| `action` | `identity`, `matrix-action` |
| `inversion` | `inverse` |
| `exponentiation` | `exp`, `exponential` |
| `square_root` | `sqrt`, `square-root` |

Names are case-insensitive.
