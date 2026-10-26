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
        isolates = df.get("_isolate", pd.Series(np.nan, index=df.index))
        df = df.assign(_isolate=isolates.fillna(df[id_col]))
        return _suffix_duplicates(df, id_col, suffix_col)

    # strategy is DuplicateStrategy.average
    df = df.copy()
    df["_original_id"] = df[id_col]
    return _suffix_duplicates(df, id_col, suffix_col)


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
