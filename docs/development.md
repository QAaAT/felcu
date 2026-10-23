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

CI runs the same command on Python 3.11 through 3.15 for every push to `main`
and every pull request (`.github/workflows/ci.yml`).

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
reports anything.  The ruff version is pinned in the `dev` extra and in CI,
so local and CI formatting agree.  Markdown files are not formatted, so code
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

Pushes to `main` publish the site to GitHub Pages
(`.github/workflows/docs.yml`).  In the repository settings, under
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
