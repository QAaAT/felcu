# Examples

Each script in `examples/` is short and self-contained, and shows one idea.
Run any of them from the project root:

```bash
python examples/action/01_basic_interval.py
```

!!! note
    The examples pass a short `m_values` range so most run in a few seconds.
    Make sure `felcu` is installed in editable mode (`pip install -e .`) so
    they use your working copy.

## Action

### Action on a symmetric interval

Fit f(H) = H for a Hermitian H with eigenvalues in [-1, 1].

Because f is odd, felcu uses a pure sine basis: every cosine coefficient
(bk) and the constant term (a0) are zero.

```python title="examples/action/01_basic_interval.py"
--8<-- "examples/action/01_basic_interval.py"
```

### Action without matrix-scale information

If no matrix or spectral bounds are given, felcu fits the unscaled function
g(x) = x directly on [-pi/eta, pi/eta].  You are then responsible for
rescaling H so that its spectrum lies inside that interval.

```python title="examples/action/02_no_scale.py"
--8<-- "examples/action/02_no_scale.py"
```

### Shifted action on an off-centre interval

For eigenvalues in [2, 5] the unshifted fit has to cover [-5, 5].  The
shifted construction centres the interval at mu = 3.5 instead:

    g(x) = mu + x / tau,   f(H) = g(tau (H - mu I))

so the constant term carries mu and only the half-width needs fitting.

```python title="examples/action/03_shifted.py"
--8<-- "examples/action/03_shifted.py"
```

### Action from a Hermitian matrix

Pass the matrix itself: felcu reads its eigenvalue range.  We then check
the block-encoded operator g(tau H) against H using an eigendecomposition.

```python title="examples/action/04_hermitian_matrix.py"
--8<-- "examples/action/04_hermitian_matrix.py"
```

### Action on a non-Hermitian matrix

For a non-Hermitian A, felcu switches to the singular values automatically.
The odd sine series acts on the singular values (as in QSVT):

    g(tau A) = U diag(g(tau s)) V^dagger  ~=  A

```python title="examples/action/05_non_hermitian_matrix.py"
--8<-- "examples/action/05_non_hermitian_matrix.py"
```

## Inversion

### Inversion on a singular-value interval

Fit f(A) = A^{-1} for singular values in [0.5, 3].

The target 1/x has a pole at 0, so felcu fits on two intervals,
[-x_max, -x_min] and [x_min, x_max], and leaves the gap around the pole
unconstrained.

```python title="examples/inversion/01_basic_interval.py"
--8<-- "examples/inversion/01_basic_interval.py"
```

### Cost of inversion versus condition number

Fix sigma_max = 1 and shrink sigma_min, so kappa = 1 / sigma_min grows.
Both the truncation order m and the LCU normalisation alpha increase.

```python title="examples/inversion/02_condition_number.py"
--8<-- "examples/inversion/02_condition_number.py"
```

### Inverting a non-Hermitian matrix

Unshifted inversion works on singular values, so A need not be Hermitian.
The fitted g(x) = tau / x is applied to the singular values:

    V diag(g(tau s)) U^dagger  ~=  A^{-1}

```python title="examples/inversion/03_non_hermitian_matrix.py"
--8<-- "examples/inversion/03_non_hermitian_matrix.py"
```

### Shifted inversion of a definite Hermitian matrix

When H is definite (all eigenvalues the same sign), the shifted
construction fits 1/x on a single interval centred at mu:

    f(H) = g(tau (H - mu I))

This works for positive- and negative-definite H alike.

```python title="examples/inversion/04_shifted_definite.py"
--8<-- "examples/inversion/04_shifted_definite.py"
```

## Exponentiation

### Exponentiation without matrix-scale information

With no matrix or bounds, felcu fits g(x) = exp(x) on [-pi/eta, pi/eta]
(eta = 2 by default, so [-pi/2, pi/2]).

```python title="examples/exponentiation/01_no_scale.py"
--8<-- "examples/exponentiation/01_no_scale.py"
```

### Truncation order versus target error

Tighten target_error and watch how the selected order m grows.  For a
smooth target like exp, m grows roughly logarithmically in 1/error.

```python title="examples/exponentiation/02_target_sweep.py"
--8<-- "examples/exponentiation/02_target_sweep.py"
```

### Shifted versus unshifted exponentiation

For eigenvalues in [1, 3], the unshifted fit covers [-3, 3].  The shifted
fit centres at mu = 2 and only covers the half-width, using

    exp(H) = g(tau (H - mu I)),   g(x) = exp(mu + x / tau).

```python title="examples/exponentiation/03_shifted_vs_unshifted.py"
--8<-- "examples/exponentiation/03_shifted_vs_unshifted.py"
```

### Matrix exponential from a Hermitian matrix

Fit exp(H) from H directly, then compare g(tau H) with the exact matrix
exponential computed by eigendecomposition.

```python title="examples/exponentiation/04_matrix_exponential.py"
--8<-- "examples/exponentiation/04_matrix_exponential.py"
```

### Complex coefficients for an LCU circuit

An LCU circuit implements sum_k c_k e^{i k x}.  felcu returns these as
result.complex_coefficients = [c_{-m}, ..., c_0, ..., c_m].  The LCU
normalisation alpha is the sum of the magnitudes of the paired terms.

```python title="examples/exponentiation/05_lcu_coefficients.py"
--8<-- "examples/exponentiation/05_lcu_coefficients.py"
```

### Effect of the Sobolev regulariser

At fixed m, felcu regularises with the H_r^w norm, weighting the k-th
harmonic by (1 + r|k|)^(2w), and picks the largest lambda that still
meets target_error.  Vary the order w and the scale r and see how
lambda_used and the LCU normalisation alpha change.

```python title="examples/exponentiation/06_sobolev_regularisation.py"
--8<-- "examples/exponentiation/06_sobolev_regularisation.py"
```

### Choosing between candidate truncation orders

With the default selection='smallest_alpha', felcu solves for every m in
m_values, keeps all candidates in result.all_results, and picks the one
with the smallest LCU normalisation.  'smallest_m' picks the shortest
series that meets the target instead, and stops the sweep there.

```python title="examples/exponentiation/07_selection_and_candidates.py"
--8<-- "examples/exponentiation/07_selection_and_candidates.py"
```

## Plots

These scripts sweep a parameter and plot how the error and the LCU
normalisation \(\alpha\) respond.  They need matplotlib:

```bash
pip install -e ".[plots]"
python examples/plots/01_error_alpha_vs_m.py
```

Each saves a PNG to the directory given as its first argument (default: the
current directory), shows it when a display is available, and prints the
numbers behind it as a table.

### Error, alpha and lambda versus truncation order m

Sweep m for each problem at the same target and plot, per m:

- the achieved Linf error (falls until it reaches the target, then stays
  there, because lambda absorbs the remaining slack);
- the LCU normalisation alpha of each feasible order;
- the regularisation strength lambda chosen for it.

The ringed point is the order picked by the default selection
('smallest_alpha').

![Error, alpha and lambda versus truncation order m](images/plots/01_error_alpha_vs_m.png)

??? example "Code: examples/plots/01_error_alpha_vs_m.py"

    ```python
    --8<-- "examples/plots/01_error_alpha_vs_m.py"
    ```

### The error-alpha trade-off controlled by lambda

At fixed m, felcu chooses the largest lambda that still meets the target,
so sweeping target_error traces out how error and alpha depend on lambda:
more regularisation means a larger error but a smaller alpha.  Repeated
for three orders m.

![The error-alpha trade-off controlled by lambda](images/plots/02_error_alpha_vs_lambda.png)

??? example "Code: examples/plots/02_error_alpha_vs_lambda.py"

    ```python
    --8<-- "examples/plots/02_error_alpha_vs_lambda.py"
    ```

### Alpha over the Sobolev parameters r and w

The H_r^w regulariser weights the k-th harmonic by (1 + r|k|)^(2w).  At
fixed m and target, sweep 0 < r <= 1 and w > 0.5 and plot the resulting
LCU normalisation alpha, as a heatmap and as slices at four values of r.
Every fit here meets the target, so only alpha changes.

![Alpha over the Sobolev parameters r and w](images/plots/03_alpha_vs_r_w.png)

??? example "Code: examples/plots/03_alpha_vs_r_w.py"

    ```python
    --8<-- "examples/plots/03_alpha_vs_r_w.py"
    ```

### Cost of inversion versus condition number

For singular values in [1/kappa, 1], plot the shortest series that meets
the target (selection='smallest_m') and its LCU normalisation alpha as the
condition number kappa grows.

![Cost of inversion versus condition number](images/plots/04_inversion_vs_kappa.png)

??? example "Code: examples/plots/04_inversion_vs_kappa.py"

    ```python
    --8<-- "examples/plots/04_inversion_vs_kappa.py"
    ```

### The fitted series over a whole period, and its pointwise error

Top: the series g over the full period [-pi, pi] against the target f on
the fitted domain.  The shaded regions are not fitted: the extension
region, and for unshifted inversion also the gap around the pole, where g
stays bounded.  Bottom: |g - f| on the fitted domain.

![The fitted series over a whole period, and its pointwise error](images/plots/05_pointwise_error.png)

??? example "Code: examples/plots/05_pointwise_error.py"

    ```python
    --8<-- "examples/plots/05_pointwise_error.py"
    ```
