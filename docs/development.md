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

## Continuous integration

`.github/workflows/ci.yml` runs on every push to `main` and every pull
request.  A new push to a branch cancels its still-running checks.

| Job | What it does |
|---|---|
| `lint` | `ruff check` and `ruff format --check` |
| `test` | The test suite with a coverage report, on Ubuntu, macOS and Windows with Python 3.11 through 3.15 |
| `test-min-deps` | The test suite on Python 3.11 with NumPy 2.0, the oldest supported versions |
| `examples` | Every script in `examples/`, including the plots (headless, from a scratch directory) |

The jobs run in stages: `lint` first, then `test` and `test-min-deps` in
parallel, then `examples`.  A job only starts once the stage before it has
passed, so a lint failure skips the tests and a test failure skips the
examples.

When you raise a minimum dependency version in `pyproject.toml`, raise it in
`test-min-deps` too.

[Dependabot](https://docs.github.com/en/code-security/dependabot)
(`.github/dependabot.yml`) opens monthly pull requests to update the GitHub
Actions versions and the ruff pin.  It leaves the library's minimum
dependency versions alone; raise those by hand.

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
