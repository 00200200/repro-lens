"""Strict JSON parsing shared by experiment results and retained reports."""

from __future__ import annotations

import json
import math
from decimal import Decimal


def same_json(first: object, second: object) -> bool:
    """Compare metadata without Python's True == 1 or 1 == 1.0 coercions."""
    try:
        return json.dumps(first, sort_keys=True, allow_nan=False) == json.dumps(
            second, sort_keys=True, allow_nan=False
        )
    except RecursionError as exc:
        raise ValueError("JSON metadata nesting is too deep to compare") from exc


def loads(text: str, source: object) -> object:
    def reject_constant(value):
        raise ValueError(f"Nonfinite JSON value {value} in {source}")

    def finite_float(value):
        number = float(value)
        if not math.isfinite(number):
            reject_constant(value)
        # float() also turns underflowing literals such as 1e-400 into 0.0.
        if number == 0.0 and Decimal(value) != 0:
            raise ValueError(f"Underflowing JSON number {value} in {source}")
        return number

    def unique_object(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"Duplicate JSON key {key!r} in {source}")
            result[key] = value
        return result

    try:
        return json.loads(
            text,
            parse_constant=reject_constant,
            parse_float=finite_float,
            object_pairs_hook=unique_object,
        )
    except RecursionError as exc:
        raise ValueError(f"JSON nesting is too deep in {source}") from exc
