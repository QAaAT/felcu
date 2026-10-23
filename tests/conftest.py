"""Test categories.

Every test belongs to exactly one category, set by ``pytestmark`` in its file:

- ``unit``: one helper or internal component in isolation;
- ``interface``: the public contract (validation, defaults, result fields,
  selection, saving and loading);
- ``convergence``: mathematical correctness (targets met, the paper's
  convergence rates and subnormalisation bounds);
- ``regression``: output unchanged from the saved references.

CI runs each category as its own step with ``pytest -m <category>``.  The
check below fails collection if a test has no category, or more than one,
so no test can fall outside every step.
"""

import pytest

CATEGORIES = ("unit", "interface", "convergence", "regression")


def pytest_collection_modifyitems(items):
    wrong = []
    for item in items:
        found = [c for c in CATEGORIES if item.get_closest_marker(c) is not None]
        if len(found) != 1:
            wrong.append(f"  {item.nodeid}: {', '.join(found) or 'no category'}")
    if wrong:
        raise pytest.UsageError(
            f"Each test needs exactly one category marker ({', '.join(CATEGORIES)}):\n" + "\n".join(wrong)
        )
