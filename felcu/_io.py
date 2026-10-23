"""JSON serialisation of FourierExtensionResult."""

from __future__ import annotations

import dataclasses
import json
import math
from importlib import metadata
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from felcu.types import FourierExtensionResult

FORMAT = "felcu.FourierExtensionResult"
FORMAT_VERSION = 1

_FLOAT_ARRAYS = {"ak", "bk"}
_FLOAT_PAIRS = {"x_gap", "gap"}
_SKIP_FIELDS = {"complex_coefficients", "all_results"}  # written separately


def _felcu_version() -> str | None:
    try:
        return metadata.version("felcu")
    except metadata.PackageNotFoundError:
        return None


def _encode_float(value: float) -> float | str:
    # Standard JSON has no Infinity or NaN (tau is infinite for a zero-width interval).
    return value if math.isfinite(value) else str(value)


def _decode_float(value: float | str) -> float:
    return float(value)


def _fields_to_dict(result: FourierExtensionResult) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for field in dataclasses.fields(result):
        name = field.name
        if name in _SKIP_FIELDS:
            continue
        value = getattr(result, name)
        if name in _FLOAT_ARRAYS:
            out[name] = [_encode_float(float(v)) for v in value]
        elif name == "k":
            out[name] = [int(v) for v in value]
        elif isinstance(value, float):
            out[name] = _encode_float(value)
        elif isinstance(value, tuple):
            out[name] = list(value)
        else:
            out[name] = value
    c = result.complex_coefficients
    out["complex_coefficients"] = {
        "k": list(range(-result.m, result.m + 1)),
        "real": [_encode_float(float(v)) for v in c.real],
        "imag": [_encode_float(float(v)) for v in c.imag],
    }
    return out


def to_dict(result: FourierExtensionResult, include_candidates: bool = False) -> dict[str, Any]:
    out = {"format": FORMAT, "format_version": FORMAT_VERSION, "felcu_version": _felcu_version()}
    out |= _fields_to_dict(result)
    if include_candidates:
        out["all_results"] = [_fields_to_dict(c) for c in result.all_results]
    return out


def _fields_from_dict(cls: type[FourierExtensionResult], data: dict[str, Any]) -> dict[str, Any]:
    kwargs: dict[str, Any] = {}
    for field in dataclasses.fields(cls):
        name = field.name
        if name in _SKIP_FIELDS:
            continue
        if name not in data:
            raise ValueError(f"Missing field '{name}' in saved result.")
        value = data[name]
        if name == "k":
            value = np.asarray(value, dtype=int)
        elif name in _FLOAT_ARRAYS:
            value = np.asarray([_decode_float(v) for v in value], dtype=float)
        elif name == "matrix_shape" and value is not None:
            value = tuple(int(v) for v in value)
        elif name in _FLOAT_PAIRS and value is not None:
            value = tuple(float(v) for v in value)
        elif isinstance(value, str) and field.type in ("float", "float | None"):
            value = _decode_float(value)
        kwargs[name] = value
    c = data["complex_coefficients"]
    real = np.asarray([_decode_float(v) for v in c["real"]], dtype=float)
    imag = np.asarray([_decode_float(v) for v in c["imag"]], dtype=float)
    kwargs["complex_coefficients"] = real + 1j * imag
    return kwargs


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
