# Problems and constructions

Every fit is described by three choices: the **problem** (which \(f\)),
whether the construction is **shifted**, and where the **spectral interval**
comes from.

## Spectral information

Give the spectral interval either as bounds or as a matrix:

```python
solve_fe_coefficients("inversion", sigma_min=0.5, sigma_max=3.0)
solve_fe_coefficients("inversion", matrix=A)
```

Passing both raises an error.  With `matrix=`, `felcu` computes eigenvalues
or singular values itself, using `spectral_kind` to decide which:

| Problem | Unshifted | Shifted |
|---|---|---|
| `action` | eigenvalues if Hermitian, otherwise singular values | eigenvalues |
| `inversion` | singular values | eigenvalues (must be definite) |
| `exponentiation` | eigenvalues | eigenvalues |

`spectral_kind='auto'` (the default) follows this table.  Eigenvalue-based
constructions require a Hermitian matrix (tested to `hermitian_tol`).

## The scaled variable

With spectral information, the fit is carried out in a scaled variable \(x\)
on an interval \([-\pi/\eta, \pi/\eta]\) strictly inside one period
\([-\pi, \pi]\).  This is what makes it a Fourier *extension*: \(g\) only
has to match \(f\) on part of the period, so a smooth non-periodic \(f\)
can still be fitted to high accuracy.

The scale \(\tau\) maps the spectral interval onto the fitted interval.  Write
\(r\) for the spectral radius (\(\max|\lambda|\), or \(\sigma_{\max}\) for
singular values), \(\mu\) for the centre and \(h\) for the half-width of
\([\sigma_{\min}, \sigma_{\max}]\).

| Problem | Construction | Fitted \(g(x)\) | \(\tau\) | Basis | Domain |
|---|---|---|---|---|---|
| action | unshifted | \(x/\tau\) | \(\pi/(\eta r)\) | odd sine | single |
| action | shifted | \(\mu + x/\tau\) | \(\pi/(\eta h)\) | constant + sine | single |
| inversion | unshifted | \(\tau/x\) | \(\pi/(\eta\sigma_{\max})\) | odd sine | double |
| inversion | shifted | \(1/(\mu + x/\tau)\) | \(\pi/(\eta h)\) | full | single |
| exponentiation | unshifted | \(e^{x/\tau}\) | \(\pi/(\eta r)\) | full | single |
| exponentiation | shifted | \(e^{\mu + x/\tau}\) | \(\pi/(\eta h)\) | full | single |

In every case \(f(H) = g(\tau H)\) (unshifted) or
\(f(H) = g(\tau(H - \mu I))\) (shifted).

### Bases

- **odd sine**: \(a_0 = 0\) and \(b_k = 0\); only sine terms.  Used when the
  target is odd.
- **constant + sine**: a constant plus sine terms.
- **full**: constant, sine and cosine terms.

Restricting the basis halves the number of unknowns and guarantees the
right symmetry exactly.

### Unshifted inversion: the double interval

\(1/x\) has a pole at zero.  Unshifted inversion therefore fits on two
intervals,

\[
[-\pi/\eta,\ -\tau\sigma_{\min}] \cup [\tau\sigma_{\min},\ \pi/\eta],
\]

and leaves the gap around the pole unconstrained.  `result.domain` is
`"double"` and `result.x_min` holds the inner edge.

## Shifted versus unshifted

The shifted construction centres the spectrum before fitting.  For an
interval far from zero, such as \([2, 5]\), the unshifted fit has to cover
\([-5, 5]\) while the shifted fit only covers the half-width \(1.5\).  This
usually lowers both \(m\) and \(\alpha\), at the cost of implementing
\(H - \mu I\) in the circuit.  Shifted constructions always need spectral
information.

Shifted inversion requires a *definite* interval (both bounds the same sign,
neither zero); negative-definite matrices work too.

## The extension parameter \(\eta\)

\(\eta > 1\) controls how much of the period is left free.  Larger \(\eta\)
gives the series more room to turn around, but packs the spectrum into a
smaller interval.  The defaults are:

| Problem | Default \(\eta\) |
|---|---|
| action, exponentiation | \(2\) |
| inversion, unshifted | \(1 + 1/\kappa\) |
| inversion, shifted | \((\kappa + 1)/(\kappa - 1)\) |

where \(\kappa = \sigma_{\max}/\sigma_{\min}\) (for shifted inversion, the ratio
of the largest to smallest \(|\lambda|\)).  Override with `eta=`, or query
the default with [`default_eta`](../api.md#felcu.default_eta).

## No spectral information

If neither `matrix` nor bounds are given, `felcu` fits the unscaled function
on \([-\pi/\eta, \pi/\eta]\) with \(\tau = 1\):

- `action` fits \(g(x) = x\);
- `exponentiation` fits \(g(x) = e^{x}\).

You are then responsible for scaling \(H\) into that interval.  Inversion
and all shifted constructions need spectral information and raise an error
without it.

## Problem name aliases

| Canonical | Also accepted |
|---|---|
| `action` | `identity`, `matrix-action` |
| `inversion` | `inverse` |
| `exponentiation` | `exp`, `exponential` |

Names are case-insensitive.
