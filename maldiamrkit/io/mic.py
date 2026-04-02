"""Minimum Inhibitory Concentration (MIC) parsing utilities."""

from __future__ import annotations

import re

import numpy as np
import pandas as pd

_MIC_PATTERN = re.compile(r"^\s*([<>]=?|=)?\s*([\d]+[,.]?\d*)\s*$")


def parse_mic_column(series: pd.Series) -> pd.DataFrame:
    """
    Parse a column of MIC strings into numeric values and qualifiers.

    Handles European comma decimals (e.g. ``"0,5"`` becomes ``0.5``),
    qualifier prefixes (``"<=8"``, ``">16"``), and missing values.

    Parameters
    ----------
    series : pd.Series
        Column of MIC strings.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns ``'value'`` (float) and
        ``'qualifier'`` (str: ``"<="``, ``">="``, ``">"``, ``"<"``,
        ``"="``, or ``""`` for missing).

    Examples
    --------
    >>> import pandas as pd
    >>> from maldiamrkit.io import parse_mic_column
    >>> s = pd.Series(["<=8", ">16", "0,5", None])
    >>> parse_mic_column(s)
       value qualifier
    0    8.0       <=
    1   16.0        >
    2    0.5        =
    3    NaN
    """
    values = np.full(len(series), np.nan)
    qualifiers = np.full(len(series), "", dtype=object)

    for i, raw in enumerate(series):
        if pd.isna(raw):
            continue
        text = str(raw).strip()
        if not text:
            continue
        match = _MIC_PATTERN.match(text)
        if match is None:
            continue
        qualifier = match.group(1) or "="
        num_str = match.group(2).replace(",", ".")
        values[i] = float(num_str)
        qualifiers[i] = qualifier

    return pd.DataFrame({"value": values, "qualifier": qualifiers}, index=series.index)
