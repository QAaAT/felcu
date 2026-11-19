"""JSON serialisation of FourierExtensionResult."""

from __future__ import annotations

import json
import math
from collections.abc import Callable
from functools import cache
from importlib import metadata
from pathlib import Path
from typing import TYPE_CHECKING, Any, get_type_hints

import numpy as np

from felcu._fitted import canonical_metric
from felcu._methods import canonical_method
from felcu._problems import canonical_problem

if TYPE_CHECKING:
    from felcu.types import FourierExtensionResult

FORMAT = "felcu.FourierExtensionResult"
FORMAT_VERSION = 1


def _felcu_version() -> str | None:
    try:
        return metadata.version("felcu")
    except metadata.PackageNotFoundError:
        return None


# ---------------------------------------------------------------------------
# Codecs: (encode, decode) between a field's value and its JSON form
# ---------------------------------------------------------------------------


def _encode_float(value: float) -> float | str:
    # Standard JSON has no Infinity or NaN (tau is infinite for a zero-width interval).
    value = float(value)
    return value if math.isfinite(value) else str(value)


def _optional(codec: tuple[Callable, Callable]) -> tuple[Callable, Callable]:
    encode, decode = codec
    return (lambda v: None if v is None else encode(v)), (lambda v: None if v is None else decode(v))


_PLAIN = (lambda v: v, lambda v: v)
_FLOAT = (_encode_float, float)
_FLOATS = (lambda v: [_encode_float(x) for x in v], lambda v: np.asarray([float(x) for x in v], dtype=float))
_INTS = (lambda v: [int(x) for x in v], lambda v: np.asarray(v, dtype=int))
_FLOAT_PAIR = (lambda v: [_encode_float(x) for x in v], lambda v: tuple(float(x) for x in v))
_INT_PAIR = (lambda v: [int(x) for x in v], lambda v: tuple(int(x) for x in v))
_COMPLEX = (
    lambda c: {
        "k": list(range(-(c.size // 2), c.size // 2 + 1)),
        "real": [_encode_float(x) for x in c.real],
        "imag": [_encode_float(x) for x in c.imag],
    },
    lambda d: np.asarray([float(x) for x in d["real"]]) + 1j * np.asarray([float(x) for x in d["imag"]]),
)

# Codecs by type, and for arrays, whose type does not say what they hold, by name.
_BY_TYPE = {
    str: _PLAIN,
    int: _PLAIN,
    bool: _PLAIN,
    float: _FLOAT,
    float | None: _optional(_FLOAT),
    bool | None: _PLAIN,
    tuple[float, float] | None: _optional(_FLOAT_PAIR),
    tuple[int, int] | None: _optional(_INT_PAIR),
}
_ARRAYS = {"ak": _FLOATS, "bk": _FLOATS, "k": _INTS, "complex_coefficients": _COMPLEX}
# Names are checked on loading, so an unknown one fails here, and an alias loads as its canonical name.
_NAMES = {
    "problem": (_PLAIN[0], canonical_problem),
    "method": (_PLAIN[0], canonical_method),
    "error_metric": (_PLAIN[0], canonical_metric),
}
_SKIP_FIELDS = {"all_results"}  # written separately


@cache
def _codecs(cls: type) -> dict[str, tuple[Callable, Callable]]:
    """The codec of every saved field of *cls*; a field of a type without one is an error, not a guess."""
    codecs = {}
    for name, hint in get_type_hints(cls).items():
        if name in _SKIP_FIELDS:
            continue
        codec = _NAMES.get(name) or (_ARRAYS.get(name) if hint is np.ndarray else _BY_TYPE.get(hint))
        if codec is None:
            raise TypeError(f"No JSON codec for field {name!r} of type {hint}.")
        codecs[name] = codec
    return codecs


def _fields_to_dict(result: FourierExtensionResult) -> dict[str, Any]:
    return {name: encode(getattr(result, name)) for name, (encode, _) in _codecs(type(result)).items()}


def _fields_from_dict(cls: type[FourierExtensionResult], data: dict[str, Any]) -> dict[str, Any]:
    missing = [name for name in _codecs(cls) if name not in data]
    if missing:
        raise ValueError(f"Missing field '{missing[0]}' in saved result.")
    return {name: decode(data[name]) for name, (_, decode) in _codecs(cls).items()}


def to_dict(result: FourierExtensionResult, include_candidates: bool = False) -> dict[str, Any]:
    out = {"format": FORMAT, "format_version": FORMAT_VERSION, "felcu_version": _felcu_version()}
    out |= _fields_to_dict(result)
    if include_candidates:
        out["all_results"] = [_fields_to_dict(c) for c in result.all_results]
    return out


def from_dict(cls: type[FourierExtensionResult], data: dict[str, Any]) -> FourierExtensionResult:
    if data.get("format") != FORMAT:
        raise ValueError(f"Not a saved felcu result (expected format '{FORMAT}').")
    version = data.get("format_version")
    if version != FORMAT_VERSION:
        raise ValueError(f"Unsupported format_version {version!r}; this felcu reads version {FORMAT_VERSION}.")
    candidates = tuple(cls(**_fields_from_dict(cls, c)) for c in data.get("all_results", []))
    return cls(**_fields_from_dict(cls, data), all_results=candidates)


def to_json(
    result: FourierExtensionResult,
    path: str | Path | None = None,
    *,
    indent: int | None = 2,
    include_candidates: bool = False,
) -> str:
    text = json.dumps(to_dict(result, include_candidates), indent=indent, allow_nan=False)
    if path is not None:
        Path(path).write_text(text + "\n", encoding="utf-8")
    return text


def from_json(cls: type[FourierExtensionResult], source: str | Path) -> FourierExtensionResult:
    if isinstance(source, str) and source.lstrip().startswith("{"):
        text = source
    else:
        text = Path(source).read_text(encoding="utf-8")
    return from_dict(cls, json.loads(text))
