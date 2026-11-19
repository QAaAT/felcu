"""Helpers shared by the test files."""

import numpy as np

from felcu import evaluate_fourier_series
from felcu._problems import PROBLEMS
from felcu.utils import evaluate_target_function, fitted_domain_segments

# Problem lists by property, so a new problem is tested wherever its properties apply.
ALL_PROBLEMS = list(PROBLEMS)
BRANCH_POINT_PROBLEMS = [name for name, p in PROBLEMS.items() if p.positive_definite]
SINGULAR_PROBLEMS = [name for name, p in PROBLEMS.items() if not p.entire]
NOT_ODD_PROBLEMS = [name for name, p in PROBLEMS.items() if not p.odd]


def max_error(result, n=3001):
    """L∞ error between the fitted series and the exact target over the fitted domain."""
    errors = []
    for xs in fitted_domain_segments(result, num=n):
        y_fit = evaluate_fourier_series(xs, result.a0, result.ak, result.bk)
        errors.append(np.max(np.abs(y_fit - evaluate_target_function(xs, result))))
    return max(errors)
