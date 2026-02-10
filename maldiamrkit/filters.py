"""Composable filter system for MaldiSet.

Filters can be combined with ``&`` (and), ``|`` (or) and ``~`` (invert)
to build complex predicates that select spectra from a :class:`MaldiSet`.

Examples
--------
>>> from maldiamrkit.filters import SpeciesFilter, QualityFilter, MetadataFilter
>>> f = SpeciesFilter("Escherichia coli") & QualityFilter(min_snr=5.0)
>>> filtered_ds = ds.filter(f)

>>> f = SpeciesFilter(["Klebsiella pneumoniae", "Escherichia coli"])
>>> filtered_ds = ds.filter(f)

>>> f = MetadataFilter("batch_id", lambda v: v == "batch_1")
>>> filtered_ds = ds.filter(f)
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Callable, Sequence

import pandas as pd


class SpectrumFilter(ABC):
    """Base filter with operator overloading.

    Subclasses must implement :meth:`__call__` which receives a single
    row of the metadata DataFrame (as a :class:`pandas.Series`) and
    returns ``True`` to keep the sample.
    """

    @abstractmethod
    def __call__(self, meta_row: pd.Series) -> bool:
        """Return True if the sample should be kept."""

    def __and__(self, other: SpectrumFilter) -> _AndFilter:
        return _AndFilter(self, other)

    def __or__(self, other: SpectrumFilter) -> _OrFilter:
        return _OrFilter(self, other)

    def __invert__(self) -> _NotFilter:
        return _NotFilter(self)

    def __repr__(self) -> str:
        return f"{type(self).__name__}()"


class _AndFilter(SpectrumFilter):
    """Intersection of two filters."""

    def __init__(self, left: SpectrumFilter, right: SpectrumFilter) -> None:
        self.left = left
        self.right = right

    def __call__(self, meta_row: pd.Series) -> bool:
        return self.left(meta_row) and self.right(meta_row)

    def __repr__(self) -> str:
        return f"({self.left!r} & {self.right!r})"


class _OrFilter(SpectrumFilter):
    """Union of two filters."""

    def __init__(self, left: SpectrumFilter, right: SpectrumFilter) -> None:
        self.left = left
        self.right = right

    def __call__(self, meta_row: pd.Series) -> bool:
        return self.left(meta_row) or self.right(meta_row)

    def __repr__(self) -> str:
        return f"({self.left!r} | {self.right!r})"


class _NotFilter(SpectrumFilter):
    """Negation of a filter."""

    def __init__(self, inner: SpectrumFilter) -> None:
        self.inner = inner

    def __call__(self, meta_row: pd.Series) -> bool:
        return not self.inner(meta_row)

    def __repr__(self) -> str:
        return f"(~{self.inner!r})"


class SpeciesFilter(SpectrumFilter):
    """Filter by species name(s).

    Parameters
    ----------
    species : str or list of str
        Species name(s) to keep.
    column : str, default="Species"
        Metadata column containing species information.
    """

    def __init__(self, species: str | Sequence[str], column: str = "Species") -> None:
        if isinstance(species, str):
            self._species = {species}
        else:
            self._species = set(species)
        self.column = column

    def __call__(self, meta_row: pd.Series) -> bool:
        """Return True if the row's species is in the filter set."""
        val = meta_row.get(self.column)
        return val in self._species

    def __repr__(self) -> str:
        species = sorted(self._species)
        if len(species) == 1:
            return f"SpeciesFilter({species[0]!r})"
        return f"SpeciesFilter({species!r})"


class DrugFilter(SpectrumFilter):
    """Filter by antibiotic resistance status.

    Parameters
    ----------
    drug : str
        Antibiotic column name in metadata.
    status : str or list of str, optional
        Keep only samples with this resistance status (e.g. ``"R"``, ``"S"``,
        ``"I"``).  If *None*, keeps any sample where the drug column is not
        null.

    Examples
    --------
    >>> DrugFilter("Ceftriaxone")                    # has data for this drug
    >>> DrugFilter("Ceftriaxone", status="R")        # resistant only
    >>> DrugFilter("Ceftriaxone", status=["R", "I"]) # resistant or intermediate
    """

    def __init__(self, drug: str, status: str | Sequence[str] | None = None) -> None:
        self.drug = drug
        if isinstance(status, str):
            self._status: set[str] | None = {status}
        elif status is not None:
            self._status = set(status)
        else:
            self._status = None

    def __call__(self, meta_row: pd.Series) -> bool:
        """Return True if the sample matches the drug filter criteria."""
        val = meta_row.get(self.drug)
        if val is None:
            return False
        try:
            if val != val:  # NaN check
                return False
        except (TypeError, ValueError):
            pass
        if self._status is not None:
            return val in self._status
        return True

    def __repr__(self) -> str:
        if self._status is not None:
            status = sorted(self._status)
            if len(status) == 1:
                return f"DrugFilter({self.drug!r}, status={status[0]!r})"
            return f"DrugFilter({self.drug!r}, status={status!r})"
        return f"DrugFilter({self.drug!r})"


class QualityFilter(SpectrumFilter):
    """Filter by quality metrics stored in metadata columns.

    Parameters
    ----------
    min_snr : float, optional
        Minimum signal-to-noise ratio (column ``snr``).
    min_peaks : int, optional
        Minimum number of detected peaks (column ``n_peaks``).
    max_baseline_fraction : float, optional
        Maximum fraction of intensity in the baseline (column ``baseline_fraction``).
    """

    def __init__(
        self,
        min_snr: float | None = None,
        min_peaks: int | None = None,
        max_baseline_fraction: float | None = None,
    ) -> None:
        self.min_snr = min_snr
        self.min_peaks = min_peaks
        self.max_baseline_fraction = max_baseline_fraction

    def __call__(self, meta_row: pd.Series) -> bool:
        """Return True if the row passes all quality thresholds."""
        if self.min_snr is not None:
            snr = meta_row.get("snr")
            if snr is None or snr < self.min_snr:
                return False
        if self.min_peaks is not None:
            n_peaks = meta_row.get("n_peaks")
            if n_peaks is None or n_peaks < self.min_peaks:
                return False
        if self.max_baseline_fraction is not None:
            bf = meta_row.get("baseline_fraction")
            if bf is None or bf > self.max_baseline_fraction:
                return False
        return True

    def __repr__(self) -> str:
        parts = []
        if self.min_snr is not None:
            parts.append(f"min_snr={self.min_snr}")
        if self.min_peaks is not None:
            parts.append(f"min_peaks={self.min_peaks}")
        if self.max_baseline_fraction is not None:
            parts.append(f"max_baseline_fraction={self.max_baseline_fraction}")
        return f"QualityFilter({', '.join(parts)})"


class MetadataFilter(SpectrumFilter):
    """Filter by arbitrary metadata column condition.

    Parameters
    ----------
    column : str
        Metadata column name.
    condition : callable
        Function that takes a single value and returns bool.

    Examples
    --------
    >>> MetadataFilter("batch_id", lambda v: v == "batch_1")
    >>> MetadataFilter("age", lambda v: v >= 18)
    """

    def __init__(self, column: str, condition: Callable[[Any], bool]) -> None:
        self.column = column
        self.condition = condition

    def __call__(self, meta_row: pd.Series) -> bool:
        """Apply the filter condition to a metadata row.

        Raises
        ------
        ValueError
            If the condition callable raises an exception when applied
            to the column value.
        """
        val = meta_row.get(self.column)
        if val is None:
            return False
        try:
            return bool(self.condition(val))
        except Exception as exc:
            raise ValueError(
                f"MetadataFilter condition failed on column '{self.column}' "
                f"with value {val!r}: {exc}"
            ) from exc

    def __repr__(self) -> str:
        return f"MetadataFilter({self.column!r})"
