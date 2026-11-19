"""Names of the problems, methods and error metrics felcu accepts, and their aliases.

A leaf module: it imports nothing from felcu, so every other module can use it.
"""

from __future__ import annotations

# Functions with a branch point at 0, fitted on a positive-definite interval
# only (Sec. VI D of the paper).
POSITIVE_DEFINITE_PROBLEMS = frozenset({"square_root", "logarithm"})

# Functions analytic on the whole plane.  The others have a pole or branch
# point at 0 that bounds how far the explicit methods may extend them.
ENTIRE_PROBLEMS = frozenset({"identity", "exponentiation"})

# Odd functions.  Only these can act on singular values, through the odd
# series that the Hermitian dilation of a general matrix needs.
ODD_PROBLEMS = frozenset({"identity", "inversion"})

_PROBLEM_ALIASES = {
    "identity": "identity",
    "action": "identity",
    "matrix-action": "identity",
    "inversion": "inversion",
    "inverse": "inversion",
    "exp": "exponentiation",
    "exponentiation": "exponentiation",
    "exponential": "exponentiation",
    "square_root": "square_root",
    "square-root": "square_root",
    "sqrt": "square_root",
    "logarithm": "logarithm",
    "log": "logarithm",
    "ln": "logarithm",
}

_METHOD_ALIASES = {
    "sobolev": "sobolev",
    "regularised": "sobolev",
    "regularized": "sobolev",
    "reg": "sobolev",
    "reflected": "reflected",
    "refl": "reflected",
    "reflected_extension": "reflected",
    "reflected-extension": "reflected",
    "arcsine_taylor": "arcsine_taylor",
    "arcsine-taylor": "arcsine_taylor",
    "arcsine": "arcsine_taylor",
    "asin": "arcsine_taylor",
}

_METRIC_ALIASES = {
    "l2": "l2",
    "2": "l2",
    "linf": "linf",
    "l_inf": "linf",
    "l-infinity": "linf",
    "infinity": "linf",
    "inf": "linf",
    "max": "linf",
}


def _canonical(name: str, aliases: dict[str, str], what: str) -> str:
    """The canonical name for *name* in *aliases*, ignoring case and surrounding whitespace."""
    key = name.strip().lower()
    if key not in aliases:
        names = list(dict.fromkeys(aliases.values()))
        listed = ", ".join(f"'{n}'" for n in names[:-1]) + (", or " if len(names) > 2 else " or ") + f"'{names[-1]}'"
        raise ValueError(f"{what} must be {listed}.")
    return aliases[key]


def canonical_problem(name: str) -> str:
    return _canonical(name, _PROBLEM_ALIASES, "problem")


def canonical_method(name: str) -> str:
    return _canonical(name, _METHOD_ALIASES, "method")


def canonical_metric(name: str) -> str:
    return _canonical(name, _METRIC_ALIASES, "error_metric")
