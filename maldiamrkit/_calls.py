"""Susceptibility calls marked as set by a rule.

Some datasets (MARISMa, for example) write a trailing ``*`` on a call
set by an expert rule rather than read from the MIC (``R*``, ``S*``).
These helpers drop the marker and surrounding spaces, so that such
calls match, count and encode as the plain call.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

_CALLS = frozenset({"S", "I", "R"})


def _plain_call(value: Any) -> Any:
    """Return an S/I/R call without spaces and its trailing ``*``.

    Letter case is kept (``"r*"`` gives ``"r"``). Values that are not
    S/I/R calls, with or without the marker (``"ESBL"``, ``"N/R"``,
    NaN, numbers), come back unchanged.
    """
    if not isinstance(value, str):
        return value
    text = value.strip()
    if text.endswith("*"):
        text = text[:-1].rstrip()
    return text if text.upper() in _CALLS else value


def _is_call(value: Any) -> bool:
    """Return whether *value* is an S/I/R call, with or without ``*``."""
    return isinstance(value, str) and _plain_call(value).upper() in _CALLS


def _plain_calls(values: Any) -> Any:
    """Apply :func:`_plain_call` to every element of an array or Series.

    A Series stays a Series (same index); anything else becomes a NumPy
    array of the same shape. Numeric arrays come back unchanged.
    """
    if isinstance(values, pd.Series):
        if values.dtype.kind not in "OUS":
            return values
        return values.map(_plain_call)
    arr = np.asarray(values)
    if arr.dtype.kind not in "OUS":
        return arr
    plain = [_plain_call(v) for v in arr.ravel()]
    dtype = object if arr.dtype.kind == "O" else None
    return np.array(plain, dtype=dtype).reshape(arr.shape)
