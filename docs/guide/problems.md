# Problems and constructions

Every fit is described by the **problem** (which \(f\)) and the
**spectrum** it acts on: eigenvalues or singular values, and where they
come from.  A third choice, the coefficient-selection **method**, is
covered on the [Method](method.md) page.

## Spectral information

A spectral interval is required.  Give it either as bounds or as a matrix:

```python
solve_fe_coefficients("inversion", sigma_min=0.5, sigma_max=3.0)
solve_fe_coefficients("inversion", matrix=A)
```

Passing neither, or both, raises an error.  With `matrix=`, `felcu`
computes eigenvalues or singular values itself.

`spectral_kind` says which data the bounds (or the matrix) give:

| Problem | `spectral_kind='auto'` (default) | Also available |
|---|---|---|
| `action` | eigenvalues; singular values for a non-Hermitian matrix | the other one |
| `inversion` | eigenvalues; singular values for a non-Hermitian matrix | the other one |
| `exponentiation` | eigenvalues | none |
| `square_root` | eigenvalues (must be positive definite) | none |

Eigenvalues need a Hermitian matrix (tested to `hermitian_tol`).
Singular values are only available for the odd functions, action and
inversion (see [Singular values](#singular-values)).

## The scaled variable

Following Sec. II of the paper, the fit is carried out in a scaled
variable \(x\) on the fitted interval \(\Omega_\eta = [-\pi/\eta, \pi/\eta]\),
strictly inside one period \(\mathbb T = [-\pi, \pi)\).  This is what makes
it a Fourier *extension*: the series \(g_m\) only has to match the target
on part of the period, so a smooth non-periodic function can still be
fitted to high accuracy.

Every fit is centred on the spectrum.  With \(\mu\) and \(\delta\) the
midpoint and half-width of the fitted spectrum, the scale
\(\tau = \pi/(\eta\delta)\) maps it onto \(\Omega_\eta\), giving the scaled
matrix \(\mathcal G = \tau(\mathcal H - \mu I)\) and the fitted function

\[
g(x) = f\!\left(\mu + \frac{x}{\tau}\right), \qquad f(\mathcal H) = g(\mathcal G).
\]

`result.mu` and `result.tau` hold \(\mu\) and \(\tau\).  The block encoding
then implements \(g_m(\mathcal G) = \sum_k c_k e^{ik\mathcal G}\), where each
\(e^{ik\mathcal G} = e^{-ik\tau\mu} e^{ik\tau\mathcal H}\) is a Hamiltonian
simulation of \(\mathcal H\) up to a known phase.

Centring means the series only covers the half-width: for an interval far
from zero, such as \([2, 5]\), it fits \([-1.5, 1.5]\) rather than
\([-5, 5]\), and the constant part \(c_0 I\) carries \(f(\mu)\).

| Problem | Spectrum | Fitted \(g(x)\) | Basis | Fitted set |
|---|---|---|---|---|
| action | eigenvalues | \(\mu + x/\tau\) | constant + sine | \(\Omega_\eta\) |
| action | singular values | \(x/\tau\) | odd sine | \(\Omega_\eta\) |
| inversion | definite eigenvalues | \((\mu + x/\tau)^{-1}\) | full | \(\Omega_\eta\) |
| inversion | indefinite eigenvalues | \((\mu + x/\tau)^{-1}\) | full | \(\Omega_\eta\) less the gap |
| inversion | singular values | \(\tau/x\) | odd sine | \(\Omega_\eta\) less the gap |
| exponentiation | eigenvalues | \(e^{\mu + x/\tau}\) | full | \(\Omega_\eta\) |
| square root | positive eigenvalues | \(\sqrt{\mu + x/\tau}\) | full | \(\Omega_\eta\) |

### Bases

- **odd sine**: \(a_0 = 0\) and \(b_k = 0\); only sine terms.  Used for
  singular values, which need an odd series.
- **constant + sine**: a constant plus sine terms.
- **full**: constant, sine and cosine terms.

Restricting the basis halves the number of unknowns and guarantees the
right symmetry exactly.

## Singular values

A general matrix \(A\) has no real eigenvalue interval, so it is handled
through its Hermitian dilation

\[
\mathcal H = \begin{pmatrix} 0 & A \\ A^\dagger & 0 \end{pmatrix},
\qquad \operatorname{spec}(\mathcal H) = \pm\operatorname{sv}(A).
\]

The dilation's spectrum is symmetric, so its midpoint is \(\mu = 0\) and
its half-width is \(\Sigma_{\max}\): the same centred construction, with
\(\tau = \pi/(\eta\Sigma_{\max})\).  With \(A = U\Sigma V^\dagger\), an odd
\(g\) gives

\[
g(\mathcal H) = \begin{pmatrix} 0 & U g(\Sigma) V^\dagger \\ V g(\Sigma) U^\dagger & 0 \end{pmatrix},
\]

so the series is restricted to odd sine terms.  For inversion the
lower-left block is \(A^{-1}\).

## Inversion

\(x^{-1}\) has a pole at zero.  On a **definite** interval (both bounds the
same sign, neither zero; negative-definite matrices work too), the pole lies
outside the spectrum, and the default \(\eta\) puts it on the periodic
boundary.

On an **indefinite** interval the spectrum lies on both sides of the pole,
\(\operatorname{spec}(\mathcal H) \subseteq [\lambda_{\min}, a] \cup [b, \lambda_{\max}]\)
with \(a < 0 < b\).  The fit covers both parts and leaves the gap around the
pole unconstrained:

\[
\Omega = [-x_{\max}, \tau(a - \mu)] \cup [\tau(b - \mu), x_{\max}],
\qquad x_{\max} = \frac{\pi}{\eta}.
\]

`result.x_gap` holds the gap in the scaled variable.  Give the gap with
`gap=(a, b)`; with `matrix=`, it is read from the eigenvalues either side of
zero, and a matrix with an eigenvalue at zero is rejected as singular.  When
the gap reaches an end of the spectrum, as for a matrix with a single
negative eigenvalue, that side of \(\Omega\) is a single point.  It is still
fitted, weighted as much as the whole set, since as a measure it would carry
no weight:

```python
solve_fe_coefficients("inversion", sigma_min=-2.0, sigma_max=3.0, gap=(-0.5, 0.4))
solve_fe_coefficients("inversion", matrix=H)  # indefinite Hermitian H
```

Singular-value inversion is the symmetric case, \(a = -\Sigma_{\min}\) and
\(b = \Sigma_{\min}\), which is the disconnected set
\(\Omega^{\rm indef}_{\eta,\kappa}\) of Sec. VI C of the paper.  A general
indefinite interval is not centred on the pole, so \(\mu \neq 0\).

## Square root

\(\sqrt{x}\) has a branch point at zero and is only defined (as a real
function) for \(x > 0\).  It follows the paper's
treatment of the matrix square root (Sec. VI D): \(\mathcal H\) must be
positive definite, \(0 < \lambda_{\min} \le \lambda_{\max}\), and the default
\(\eta\) places the branch point on the periodic boundary rather than near
the fitted endpoint.

\(\sqrt x\) is concave, so it does not satisfy the convex
endpoint-reflection condition and the subnormalisation is bounded but not
optimal: the paper shows
\(\alpha_m = O(\kappa^{(2w-1)/(4w)}\sqrt{\lambda_{\max}})\), an extra
\(O(\kappa^{1/4})\) for \(w = 1\).

## The extension parameter \(\eta\)

\(\eta > 1\) sets how much of the period is left free.  Larger \(\eta\)
gives the series more room to turn around, but packs the spectrum into a
smaller interval.  For `method="sobolev"` the defaults are:

| Problem | Default \(\eta\) | Reason |
|---|---|---|
| action, exponentiation | \(2\) | entire \(f\); optimal for reflected extensions |
| inversion (definite), square root | \(\eta_{\rm def} = \lvert\mu\rvert/\delta = (\kappa + 1)/(\kappa - 1)\) | puts the pole or branch point \(x = -\tau\mu\) at the periodic boundary |
| inversion (indefinite or singular values) | \(1 + (b - a)/(2\delta)\) | the gap around the pole and the free part of the period are equally wide |

For singular values (or any symmetric gap) the last rule is the paper's
\(\eta_{\rm indef} = 1 + 1/\kappa\).  Here
\(\kappa = \max|\lambda|/\min|\lambda|\) (\(\eta = 2\) if \(\kappa = 1\)).
Override with `eta=`, or query the default with
[`default_eta`](../api.md#felcu.default_eta), which also takes `method=`.

The explicit methods use their own defaults and ranges (see
[Method](method.md#reflected-extensions)):

| `method` | Default \(\eta\) | Allowed |
|---|---|---|
| `"reflected"` | \(2\) | \(\eta \ge 2\) |
| `"arcsine_taylor"`, action and exponentiation | \(3\) | \(\eta > 2\) |
| `"arcsine_taylor"`, inversion and square root | \(\min\{3,\ 1 + \lvert\mu\rvert/\delta\}\) | \(2 < \eta < 2\lvert\mu\rvert/\delta\) |

Both explicit methods require \(\eta < 2|\mu|/\delta\) for inversion and
square root, and neither supports the two-interval fit of indefinite or
singular-value inversion.

## Problem name aliases

| Canonical | Also accepted |
|---|---|
| `action` | `identity`, `matrix-action` |
| `inversion` | `inverse` |
| `exponentiation` | `exp`, `exponential` |
| `square_root` | `sqrt`, `square-root` |

Names are case-insensitive.
