"""Mark every test that is not a regression test as functional.

CI runs ``pytest -m functional`` and ``pytest -m regression`` as separate
steps; marking here, rather than in each file, means a new test file can
never fall outside both.
"""

import pytest


def pytest_collection_modifyitems(items):
    for item in items:
        if item.get_closest_marker("regression") is None:
            item.add_marker(pytest.mark.functional)
