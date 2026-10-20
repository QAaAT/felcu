# Development

## Setup

```bash
git clone git@github.com:ThomasHowarth/felcu.git
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

CI runs the same command on Python 3.11 and 3.12 for every push to `main`
and every pull request (`.github/workflows/ci.yml`).

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

Pushes to `main` publish the site to GitHub Pages
(`.github/workflows/docs.yml`).  In the repository settings, under
**Pages → Build and deployment**, the source must be set to
**GitHub Actions**.

## Package layout

```
felcu/
├── __init__.py        # Public API
├── types.py           # FourierExtensionResult dataclass
├── utils.py           # Public utility functions
├── spectral.py        # Spectral data extraction & problem-spec builder
├── solve.py           # Public solver (solve_fe_coefficients, compute_fe_coefficients)
├── _solver.py         # Internal: Fourier-extension fitting & regularisation search
├── _coefficients.py   # Internal: basis matrices, coefficient packing
├── _linalg.py         # Internal: QR solver, quadrature, penalties
└── _logging.py        # Internal: package logger
```
