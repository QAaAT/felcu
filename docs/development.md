# Development

## Setup

```bash
git clone git@github.com:QAaAT/felcu.git
cd felcu
pip install -e ".[dev]"
```

Use an editable install (`-e`).  A regular `pip install .` copies the
package into `site-packages`, and scripts run from elsewhere (such as
`examples/`) will keep importing that copy after you change the source.

## Tests

```bash
python -m pytest tests/ -v
```

To see which lines the tests miss, add `--cov=felcu --cov-report=term-missing`.

Every test belongs to exactly one category, one file each, from the lowest
layer up:

| Category | File | What it checks |
|---|---|---|
| `unit` | `test_unit.py` | One helper or internal component in isolation: name aliases, grids, series evaluation, the default `eta` and `r` formulas, the linear solver |
| `interface` | `test_interface.py`, `test_io.py` | The public contract: input validation, defaults and options, result fields, candidate selection, saving and loading |
| `convergence` | `test_convergence.py` | Mathematical correctness: targets met, the paper's closed-form coefficients, convergence rates and subnormalisation bounds |
| `regression` | `test_regression.py` | Output unchanged from the saved references (below) |

The category is the file's `pytestmark`, and `tests/conftest.py` stops
collection if any test has no category or more than one.  A failure low down
usually explains the ones above it: a broken helper also breaks convergence
and regression, so start with the lowest failing category.  Run one category
with `-m`:

```bash
python -m pytest tests/ -m unit
```

Convergence and regression tests can look alike, but answer different
questions.  Convergence tests check that the output is *right*; regression
tests check that it has not *changed*.  Regenerating the references after a
change that breaks a method would pass the regression tests, so the
convergence tests are what catch it.

Helpers shared between test files live in `tests/helpers.py`.

### Regression tests

`tests/test_regression.py` solves a fixed set of cases, defined in
`tests/regression_cases.py`, and compares the results with references saved
as JSON in `tests/data/regression/`.  Rounding differs between BLAS/LAPACK
builds and NumPy versions, so the comparison has three tiers:

- discrete fields (`m`, `basis_used`, `converged`, ...) must match exactly,
  and fields computed in closed form from the inputs (`eta`, `tau`,
  `sobolev_r`, ...) to a relative `1e-12`;
- the series must match as a function on the fitted domain to
  `1e-3 * target_error`.  The individual coefficients are not compared:
  the fit is ill-conditioned, so they can move by more than the target
  error without changing the series;
- `alpha` must match to a relative `1e-6`, and `lambda_used` and the
  achieved errors, which come from a bisection on a flat error curve, to
  `1e-4`.

When a change is meant to alter the coefficients, regenerate the references
and commit them with the change:

```bash
python tests/regression_cases.py
```

The script refuses to run unless `felcu` is imported from the checkout (an
editable install).  To add a case, add it to `CASES` and regenerate.  Pass
a directory (`python tests/regression_cases.py OUT_DIR`) to write the
results there instead of over the references.

## Continuous integration

`.github/workflows/ci.yml` runs on every push to `main` and every pull
request.  A new push to a branch cancels its still-running checks.

| Job | What it does |
|---|---|
| `lint` | `ruff check` and `ruff format --check` |
| `test` | Each test category as its own step, with a combined coverage report, on Ubuntu, macOS and Windows with every supported Python (3.13 through 3.15) |
| `test-min-deps` | Each test category as its own step, on Python 3.13 with NumPy 2.2, the oldest supported versions |
| `install` | The whole suite against a regular (not editable) install, run from outside the checkout, so a package that builds incomplete fails |
| `examples` | Every script in `examples/`, including the plots (headless, from a scratch directory) |

The jobs run in stages: `lint` first, then `test`, `test-min-deps` and
`install` in parallel, then `examples`.  A job only starts once the stage before it has
passed, so a lint failure skips the tests and a test failure skips the
examples.

Within `test` and `test-min-deps`, the categories run lowest first, and each
step runs even when an earlier one failed, so one failure cannot hide
another and the failing step's name says which layer broke.  Every category
runs on every platform: convergence and regression in particular depend on
rounding, which differs between BLAS/LAPACK builds and NumPy versions.  When they fail,
the job uploads that platform's results as an artifact,
`regression-results-<os>-py<version>` (or `regression-results-min-deps`),
laid out like `tests/data/regression/`, so you can diff them against the
references directly.

[Dependabot](https://docs.github.com/en/code-security/dependabot)
(`.github/dependabot.yml`) opens monthly pull requests to update the GitHub
Actions versions and the ruff pin.  It leaves the supported Python and
dependency versions alone; those follow SPEC 0 (below) and are raised by
hand.

## Supported versions

`felcu` follows [SPEC 0](https://scientific-python.org/specs/spec-0000/):
each Python version is supported for 3 years after its release, and each
version of NumPy and matplotlib (the `plots` extra) for 2 years.  The
minimum of each is the oldest release still inside its window.

| Dependency | Minimum | Next raise |
|---|---|---|
| Python | 3.13 | to 3.14 on 2027-10-07 |
| NumPy | 2.2 | to 2.3 on 2026-12-08, then 2.4 on 2027-06-07 |
| matplotlib | 3.10 | to 3.11 on 2026-12-14 |

[SPEC 0's table](https://scientific-python.org/specs/spec-0000/#support-window)
lists the drop dates.  When one passes, raise the minimum in all of:

- `requires-python`, the classifiers, `dependencies` and the `plots` extra in
  `pyproject.toml`, and ruff's `target-version`;
- the `test` matrix and the `test-min-deps` job in `.github/workflows/ci.yml`,
  and the Python used by the other jobs there and in `docs.yml`;
- the requirements in `docs/getting-started.md` and the table above.

Add each new Python version to the `test` matrix when its first release
candidate is out; `allow-prereleases` lets CI install it before the final
release.

`.github/workflows/spec0.yml` checks this on the first of every month (or
when run by hand from the Actions tab), with `.github/scripts/spec0_check.py`.
When a floor is due to rise, or a released Python version is missing from the
`test` matrix or the classifiers, it opens an issue titled
"SPEC 0: raise the supported versions", or updates the one already open.
Run the check locally with `python .github/scripts/spec0_check.py`; pass
`--today YYYY-MM-DD` to see what a later date will ask for.

## Linting and formatting

Code is linted and formatted with [ruff](https://docs.astral.sh/ruff/),
configured in `pyproject.toml` (line length 120; pycodestyle, pyflakes,
import order, bugbear, pyupgrade and NumPy-deprecation rules).  Before
pushing:

```bash
ruff check --fix .
ruff format .
```

CI runs `ruff check .` and `ruff format --check .` and fails if either
reports anything.  The ruff version is pinned once, in the `dev` extra, and
CI installs it from there, so local and CI formatting agree.  Markdown files are not formatted, so code
blocks in `docs/` are left as written.

## Documentation

The docs are built with [MkDocs](https://www.mkdocs.org/) and the
[Material](https://squidfunk.github.io/mkdocs-material/) theme; the API
reference is generated from docstrings by
[mkdocstrings](https://mkdocstrings.github.io/).

```bash
pip install -e ".[docs]"
mkdocs serve        # live preview at http://127.0.0.1:8000
mkdocs build --strict
```

The plot images on the Examples page are committed under
`docs/images/plots/`, so building the docs does not need matplotlib.  After
changing a plotting example, regenerate them with:

```bash
pip install -e ".[plots]"
for f in examples/plots/0*.py; do MPLBACKEND=Agg python "$f" docs/images/plots; done
```

Pushes to `main` that change the site's sources (`docs/`, `mkdocs.yml`,
`felcu/` for the API reference, `examples/` for the Examples page,
`pyproject.toml` or the workflow itself) publish the site to GitHub Pages
(`.github/workflows/docs.yml`); pull requests that change them build it
without publishing.  To publish anyway, run the workflow by hand from the
Actions tab.  In the repository settings, under
**Pages → Build and deployment**, the source must be set to
**GitHub Actions**.

## Package layout

```
felcu/
├── __init__.py        # Public API
├── types.py           # FourierExtensionResult dataclass
├── _io.py             # Internal: JSON serialisation of results
├── utils.py           # Public utility functions
├── spectral.py        # Spectral data extraction & problem-spec builder
├── solve.py           # Public solver (solve_fe_coefficients, compute_fe_coefficients)
├── _solver.py         # Internal: Fourier-extension fitting & regularisation search
├── _explicit.py       # Internal: reflected-extension and arcsine-Taylor coefficients
├── _coefficients.py   # Internal: basis matrices, coefficient packing
├── _linalg.py         # Internal: least squares, quadrature, penalties
└── _logging.py        # Internal: package logger
```
