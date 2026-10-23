"""Cases for the regression tests, and the script that writes their references.

The references live in ``tests/data/regression/``, one JSON file per case.
After a change that is meant to alter the coefficients, regenerate them with

    python tests/regression_cases.py

and commit the new files alongside the change.  Pass a directory to write
there instead, e.g. to compare another platform's results with the references.
"""

import sys
from pathlib import Path

import felcu
from felcu import solve_fe_coefficients

DATA_DIR = Path(__file__).parent / "data" / "regression"

# Small m grids keep each case fast while still exercising the lambda search
# and the choice between candidates.  The explicit methods converge more
# slowly, so their cases set their own grid, each spanning orders that miss
# and meet the target.
M_VALUES = [16, 24, 32]

CASES = {
    "action": dict(problem="action", sigma_min=-1.0, sigma_max=2.0, target_error=1e-6),
    "action_singular_values": dict(problem="action", matrix=[[1.0, 0.5], [-0.2, 0.8]], target_error=1e-6),
    "inversion": dict(problem="inversion", sigma_min=0.5, sigma_max=2.0, target_error=1e-4),
    "inversion_indefinite": dict(
        problem="inversion", sigma_min=-2.0, sigma_max=3.0, gap=(-0.5, 0.4), target_error=1e-4, m_values=[32, 48, 64]
    ),
    "inversion_singular_values": dict(
        problem="inversion", sigma_min=0.5, sigma_max=2.0, spectral_kind="singular_values", target_error=1e-4
    ),
    "exponentiation": dict(problem="exponentiation", sigma_min=0.1, sigma_max=2.0, target_error=1e-8),
    "exponentiation_l2": dict(
        problem="exponentiation", sigma_min=1.0, sigma_max=3.0, target_error=1e-8, error_metric="l2"
    ),
    "exponentiation_matrix": dict(problem="exponentiation", matrix=[[1.0, 0.2], [0.2, 0.5]], target_error=1e-8),
    "square_root": dict(problem="square_root", sigma_min=0.2, sigma_max=1.0, target_error=1e-4),
    "reflected_action": dict(
        problem="action", method="reflected", sigma_min=-1.0, sigma_max=1.0, target_error=1e-2, m_values=[32, 64, 128]
    ),
    "reflected_exponentiation": dict(
        problem="exponentiation",
        method="reflected",
        sigma_min=0.1,
        sigma_max=2.0,
        target_error=5e-2,
        m_values=[32, 64, 128],
    ),
    "reflected_inversion": dict(
        problem="inversion", method="reflected", sigma_min=1.0, sigma_max=2.0, target_error=1e-2
    ),
    "arcsine_taylor_exponentiation": dict(
        problem="exponentiation",
        method="arcsine_taylor",
        sigma_min=0.1,
        sigma_max=2.0,
        target_error=1e-4,
        m_values=[32, 48, 64],
    ),
    "arcsine_taylor_inversion": dict(
        problem="inversion",
        method="arcsine_taylor",
        sigma_min=1.0,
        sigma_max=2.0,
        target_error=1e-6,
        m_values=[32, 48, 64],
    ),
    "arcsine_taylor_square_root": dict(
        problem="square_root",
        method="arcsine_taylor",
        sigma_min=0.5,
        sigma_max=1.0,
        target_error=1e-6,
        m_values=[32, 48, 64],
    ),
}


def solve(name):
    return solve_fe_coefficients(**({"m_values": M_VALUES} | CASES[name]))


def reference_path(name):
    return DATA_DIR / f"{name}.json"


def regenerate(out_dir=DATA_DIR):
    # Running this file puts tests/ (not the repository) on sys.path, so a
    # non-editable install would silently write references from the wrong code.
    repo = Path(__file__).resolve().parents[1]
    if Path(felcu.__file__).resolve().parent != repo / "felcu":
        raise SystemExit(f"felcu is imported from {felcu.__file__}, not this checkout; run pip install -e .")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for old in out_dir.glob("*.json"):
        old.unlink()
    for name in CASES:
        path = out_dir / reference_path(name).name
        solve(name).to_json(path)
        print(f"wrote {path}")


if __name__ == "__main__":
    regenerate(*sys.argv[1:2])
