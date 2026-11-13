"""Unified duplicate spectrum handling.

Provides the :class:`DuplicateStrategy` enum and helper functions for
applying duplicate-handling strategies consistently across all layout
classes, :func:`~maldiamrkit.alignment.raw_warping.create_raw_input`,
and downstream transformers.
"""

from __future__ import annotations

import logging
from enum import Enum

import numpy as np
import pandas as pd

from .._calls import _is_call, _plain_call

logger = logging.getLogger(__name__)


class DuplicateStrategy(str, Enum):
    """Strategy for handling duplicate spectrum identifiers.

    Attributes
    ----------
    first : str
        Keep the first occurrence of each duplicate ID.
    last : str
        Keep the last occurrence of each duplicate ID.
    drop : str
        Remove **all** rows whose ID appears more than once.
    keep_all : str
        Retain every replicate, disambiguating IDs with a suffix
        (e.g. ``_rep1``, ``_rep2`` or the target-position value). The ID
        before the suffix is kept, so that the replicates of one isolate
        share a group.
    average : str
        Keep all replicates for downstream averaging.  Adds an
        ``_original_id`` column so that loaders / transformers can
        group replicates and average their spectra.
    """

    first = "first"
    last = "last"
    drop = "drop"
    keep_all = "keep_all"
    average = "average"


def apply_metadata_strategy(
    df: pd.DataFrame,
    strategy: DuplicateStrategy,
    *,
    id_col: str = "ID",
    suffix_col: str | None = None,
) -> pd.DataFrame:
    """Apply a duplicate-handling strategy to a metadata DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Metadata with an ``id_col`` column that may contain duplicates.
    strategy : DuplicateStrategy
        Which strategy to apply.
    id_col : str, default ``"ID"``
        Name of the identifier column.
    suffix_col : str or None
        When *strategy* is ``"keep_all"`` and *suffix_col* is given,
        duplicate IDs are disambiguated by appending
        ``_{suffix_col_value}`` (e.g. a target-position column).
        If ``None``, a sequential ``_rep1``, ``_rep2``, ... suffix is
        used instead.

    Returns
    -------
    pd.DataFrame
        The (possibly filtered / modified) DataFrame.
    """
    strategy = DuplicateStrategy(strategy)
    dup_mask = df.duplicated(subset=id_col, keep=False)
    n_dups = dup_mask.sum()

    if n_dups == 0:
        return df

    logger.warning(
        "%d duplicate ID(s) detected; applying strategy '%s'.",
        n_dups,
        strategy.value,
    )

    if strategy in (DuplicateStrategy.first, DuplicateStrategy.last):
        return df.drop_duplicates(subset=id_col, keep=strategy.value)

    if strategy is DuplicateStrategy.drop:
        return df.drop_duplicates(subset=id_col, keep=False)

    if strategy is DuplicateStrategy.keep_all:
        isolates = df.get("_isolate", pd.Series(np.nan, index=df.index, dtype=object))
        df = df.assign(_isolate=isolates.fillna(df[id_col]))
        return _suffix_duplicates(df, id_col, suffix_col)

    # strategy is DuplicateStrategy.average
    df = df.copy()
    df["_original_id"] = df[id_col]
    return _suffix_duplicates(df, id_col, suffix_col)


def _merge_spectrum_rows(
    df: pd.DataFrame,
    path_column: str,
    target_position_column: str,
    id_column: str = "ID",
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Merge the metadata rows that describe one spectrum into one row.

    Rows sharing ID, path and (when *df* has the column) target position
    describe the same spectrum, not replicates.
    In a column of S/I/R calls (one holding at least one call, ``R*`` and
    ``S*`` included) the rows keep their call when they agree (``R`` and
    ``R*`` agree; the first is kept) and get a missing value otherwise;
    in other columns the distinct non-empty values are joined with
    ``"; "`` in order of appearance. Empty cells are ignored. Nothing is
    merged without an ID and a path column.

    Returns the merged frame and the counts ``rows`` (in *df*),
    ``spectra`` (rows left), ``merged_spectra`` (spectra that had several
    rows) and ``conflicting_calls`` (calls set missing).
    """
    unmerged = {
        "rows": len(df),
        "spectra": len(df),
        "merged_spectra": 0,
        "conflicting_calls": 0,
    }
    if id_column not in df.columns or path_column not in df.columns:
        return df, unmerged
    key = [
        column
        for column in (id_column, target_position_column, path_column)
        if column in df.columns
    ]
    repeated = df.duplicated(subset=key, keep=False).to_numpy()
    if not repeated.any():
        return df, unmerged

    call_columns = {
        column
        for column in df.columns
        if column not in key
        and df[column].dtype.kind not in "biufcmM"
        and any(_is_call(v) for v in pd.unique(df[column].dropna()))
    }
    rows = df[repeated]
    codes = rows.groupby(key, sort=False, dropna=False).ngroup().to_numpy()
    order = np.argsort(codes, kind="stable")
    groups = np.split(order, np.flatnonzero(np.diff(codes[order])) + 1)
    cells = rows.to_numpy(dtype=object)
    kinds = [
        "key" if column in key else "call" if column in call_columns else "other"
        for column in df.columns
    ]
    merged: list[list] = []
    n_missing = 0
    for group in groups:
        block = cells[group]
        record = []
        for j, kind in enumerate(kinds):
            present = [v for v in block[:, j] if not _is_empty(v)]
            if kind == "key" or not present:
                record.append(block[0, j])
            elif kind == "call":
                if len({_plain_call(v) for v in present}) == 1:
                    record.append(present[0])
                else:
                    record.append(np.nan)
                    n_missing += 1
            else:
                distinct = list(dict.fromkeys(present))
                record.append(
                    distinct[0]
                    if len(distinct) == 1
                    else "; ".join(str(v) for v in distinct)
                )
        merged.append(record)

    positions = np.arange(len(df))
    first = positions[repeated & ~df.duplicated(subset=key, keep="first").to_numpy()]
    single = positions[~repeated]
    combined = pd.concat(
        [
            df.iloc[single],
            pd.DataFrame(merged, index=df.index[first], columns=df.columns),
        ]
    )
    order = np.argsort(np.concatenate([single, first]), kind="stable")
    logger.warning(
        "%d spectra had several metadata rows (%d rows): merged into one row "
        "each; %d susceptibility calls on which the rows disagree are now "
        "missing.",
        len(merged),
        int(repeated.sum()),
        n_missing,
    )
    result = combined.iloc[order].infer_objects()
    return result, {
        "rows": len(df),
        "spectra": len(result),
        "merged_spectra": len(merged),
        "conflicting_calls": n_missing,
    }


def _is_empty(value: object) -> bool:
    """Return whether a metadata cell is missing or blank."""
    if isinstance(value, str):
        return not value.strip()
    return bool(pd.isna(value))


def _suffix_duplicates(
    df: pd.DataFrame,
    id_col: str,
    suffix_col: str | None,
) -> pd.DataFrame:
    """Append disambiguating suffixes to the duplicated IDs only.

    If *suffix_col* is provided and present in the DataFrame, the
    suffix is ``_{value}``; otherwise a sequential ``_rep{N}`` suffix
    (counted per ID) is used. Unique IDs stay unchanged.

    Raises
    ------
    ValueError
        If a suffixed ID clashes with another ID.
    """
    df = df.copy()
    ids = df[id_col].astype(str)
    if suffix_col is not None and suffix_col in df.columns:
        suffixed = ids + "_" + df[suffix_col].astype(str)
    else:
        suffixed = ids + "_rep" + (ids.groupby(ids).cumcount() + 1).astype(str)
    duplicated = df.duplicated(subset=id_col, keep=False)
    df[id_col] = df[id_col].where(~duplicated, suffixed)
    _check_no_clash(df[id_col])
    return df


def _check_no_clash(ids: pd.Series | pd.Index) -> None:
    """Raise when suffixing produced an ID that appears twice."""
    clash = pd.Index(ids).duplicated(keep=False)
    if clash.any():
        raise ValueError(
            "Suffixing duplicate IDs produced IDs that clash with others: "
            f"{sorted(set(pd.Index(ids)[clash]))}. Rename them in the metadata."
        )


def apply_index_strategy(
    df: pd.DataFrame,
    strategy: DuplicateStrategy,
) -> pd.DataFrame:
    """Apply a duplicate-handling strategy on the DataFrame **index**.

    This is used by :func:`~maldiamrkit.alignment.raw_warping.create_raw_input`
    where sample IDs live in the index rather than a column.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame whose index may contain duplicate sample IDs.
    strategy : DuplicateStrategy
        Which strategy to apply.

    Returns
    -------
    pd.DataFrame
        The (possibly filtered / modified) DataFrame.
    """
    strategy = DuplicateStrategy(strategy)
    dup_mask = df.index.duplicated(keep=False)
    n_dups = dup_mask.sum()

    if n_dups == 0:
        return df

    logger.warning(
        "%d duplicate sample ID(s) detected; applying strategy '%s'.",
        n_dups,
        strategy.value,
    )

    if strategy is DuplicateStrategy.first:
        return df[~df.index.duplicated(keep="first")]

    if strategy is DuplicateStrategy.last:
        return df[~df.index.duplicated(keep="last")]

    if strategy is DuplicateStrategy.drop:
        return df[~dup_mask]

    if strategy is DuplicateStrategy.keep_all:
        return _suffix_index_duplicates(df)

    # strategy is DuplicateStrategy.average
    df = df.copy()
    df["_original_id"] = df.index
    return df


def _suffix_index_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Append ``_repN`` suffixes to the duplicated index entries only."""
    df = df.copy()
    ids = pd.Series(df.index.astype(str), index=df.index)
    suffixed = ids + "_rep" + (ids.groupby(ids).cumcount() + 1).astype(str)
    duplicated = df.index.duplicated(keep=False)
    df.index = pd.Index(np.where(duplicated, suffixed.to_numpy(), ids.to_numpy()))
    _check_no_clash(df.index)
    return df
