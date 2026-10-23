"""Cases for the regression tests, and the script that writes their references.

The references live in ``tests/data/regression/``, one JSON file per case.
After a change that is meant to alter the coefficients, regenerate them with

    python tests/regression_cases.py

and commit the new files alongside the change.
"""

from pathlib import Path

import felcu
from felcu import solve_fe_coefficients

DATA_DIR = Path(__file__).parent / "data" / "regression"

# Small m grids keep each case fast while still exercising the lambda search
# and the choice between candidates.
M_VALUES = [16, 24, 32]

CASES = {
    "action": dict(problem="action", target_error=1e-6),
    "inversion": dict(problem="inversion", sigma_min=0.5, sigma_max=2.0, target_error=1e-4),
    "inversion_shifted": dict(problem="inversion", shifted=True, sigma_min=0.5, sigma_max=2.0, target_error=1e-4),
    "exponentiation": dict(problem="exponentiation", sigma_min=0.1, sigma_max=2.0, target_error=1e-8),
    "exponentiation_shifted_l2": dict(
        problem="exponentiation", shifted=True, sigma_min=1.0, sigma_max=3.0, target_error=1e-8, error_metric="l2"
    ),
    "exponentiation_matrix": dict(problem="exponentiation", matrix=[[1.0, 0.2], [0.2, 0.5]], target_error=1e-8),
    "square_root_shifted": dict(problem="square_root", shifted=True, sigma_min=0.2, sigma_max=1.0, target_error=1e-4),
}


def solve(name):
    return solve_fe_coefficients(m_values=M_VALUES, **CASES[name])


def reference_path(name):
    return DATA_DIR / f"{name}.json"


def regenerate():
    # Running this file puts tests/ (not the repository) on sys.path, so a
    # non-editable install would silently write references from the wrong code.
    repo = Path(__file__).resolve().parents[1]
    if Path(felcu.__file__).resolve().parent != repo / "felcu":
        raise SystemExit(f"felcu is imported from {felcu.__file__}, not this checkout; run pip install -e .")
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    for old in DATA_DIR.glob("*.json"):
        old.unlink()
    for name in CASES:
        solve(name).to_json(reference_path(name))
        print(f"wrote {reference_path(name)}")


if __name__ == "__main__":
    regenerate()
