"""Sklearn-style transformer turning MIC strings into ML targets.

Wraps :func:`~maldiamrkit.io.parse_mic_column` to produce a tidy
DataFrame with regression and (optionally) classification targets in one
pass. If a :class:`~maldiamrkit.susceptibility.BreakpointTable` is supplied,
each row is also categorised as S/I/R and flagged for ATU; otherwise only
``log2_mic`` and ``censored`` are populated.
"""

from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from ..io.mic import parse_mic_column
from .breakpoint import BreakpointTable

_OUTPUT_COLUMNS = ("log2_mic", "censored", "category", "atu", "source")


class MICEncoder(BaseEstimator, TransformerMixin):
    """Encode MIC strings into log2 numeric values and optional S/I/R labels.

    Parameters
    ----------
    breakpoints : BreakpointTable or None, default=None
        When provided, each MIC is also categorised as ``S``/``I``/``R`` and
        flagged for ATU. When ``None``, only ``log2_mic`` and ``censored``
        columns are populated; ``category`` / ``atu`` / ``source`` columns
        are present but filled with ``pd.NA``.
    mic_col : str, default="MIC"
        Name of the MIC column in the input DataFrame.
    species_col : str or None, default=None
        Name of the species column in the input DataFrame. Required when
        ``breakpoints`` is provided unless ``species`` is given as a scalar.
    drug : str or None, default=None
        Antibiotic name applied to all rows (single-drug case). Mutually
        exclusive with ``drug_col``.
    drug_col : str or None, default=None
        Name of the drug column in the input DataFrame (multi-drug case).
        Mutually exclusive with ``drug``.
    species : str or None, default=None
        Species applied to all rows (single-species case). Mutually
        exclusive with ``species_col``.

    Notes
    -----
    The censoring rule treats ``≤`` / ``<`` / ``≥`` / ``>`` qualifiers in
    the source MIC strings as censored point estimates: the parsed numeric
    is kept as ``log2_mic`` and ``censored`` is set to ``True``, so
    downstream code (e.g. censoring-aware loss functions) can choose how
    to use them.

    See Also
    --------
    BreakpointTable : Clinical breakpoint lookup consumed by this encoder.
    maldiamrkit.io.parse_mic_column : Underlying MIC string parser.
    """

    def __init__(
        self,
        breakpoints: BreakpointTable | None = None,
        *,
        mic_col: str = "MIC",
        species_col: str | None = None,
        species: str | None = None,
        drug: str | None = None,
        drug_col: str | None = None,
    ) -> None:
        self.breakpoints = breakpoints
        self.mic_col = mic_col
        self.species_col = species_col
        self.species = species
        self.drug = drug
        self.drug_col = drug_col

    def fit(self, X: pd.DataFrame, y=None, **kwargs) -> MICEncoder:
        """Validate configuration (no statistics learned).

        Parameters
        ----------
        X : pd.DataFrame
            Input frame with at least ``mic_col``. Other required columns
            depend on the chosen species/drug configuration.
        y : ignored
            Present for sklearn API compatibility.
        **kwargs
            Ignored.

        Returns
        -------
        self
        """
        self._check_columns(X)
        if self.species is not None and self.species_col is not None:
            raise ValueError("Provide either species= or species_col=, not both.")
        if self.drug is not None and self.drug_col is not None:
            raise ValueError("Provide either drug= or drug_col=, not both.")
        if self.breakpoints is not None and (
            self._species_source() is None or self._drug_source() is None
        ):
            raise ValueError(
                "When breakpoints is provided, species/species_col and "
                "drug/drug_col must be configured."
            )
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Encode MIC strings.

        Parameters
        ----------
        X : pd.DataFrame
            Input frame with ``mic_col``.

        Returns
        -------
        pd.DataFrame
            Columns ``log2_mic``, ``censored``, ``category``, ``atu``,
            ``source`` indexed like ``X``.
        """
        self._check_columns(X)
        parsed = parse_mic_column(X[self.mic_col])
        values = parsed["value"].to_numpy()
        qualifiers = parsed["qualifier"].to_numpy()

        log2_mic = np.full(len(values), np.nan)
        nonzero = ~pd.isna(values) & (values > 0)
        log2_mic[nonzero] = np.log2(values[nonzero])

        censored = np.array(
            [q in ("<", "<=", ">", ">=") for q in qualifiers], dtype=bool
        )

        out = pd.DataFrame(
            {
                "log2_mic": log2_mic,
                "censored": censored,
                "category": pd.array([pd.NA] * len(values), dtype="object"),
                "atu": pd.array([False] * len(values), dtype="boolean"),
                "source": pd.array([pd.NA] * len(values), dtype="object"),
            },
            index=X.index,
        )

        if self.breakpoints is None:
            out["atu"] = pd.array([pd.NA] * len(values), dtype="boolean")
            return out

        species_arr = self._resolve_array(X, self.species, self.species_col)
        drug_arr = self._resolve_array(X, self.drug, self.drug_col)
        batch = self.breakpoints.apply_batch(species_arr, drug_arr, values)
        out["category"] = pd.array(
            [c if c is not None else pd.NA for c in batch["category"]],
            dtype="object",
        )
        out["atu"] = pd.array(batch["atu"].tolist(), dtype="boolean")
        out["source"] = pd.array(
            [s if s is not None else pd.NA for s in batch["source"]],
            dtype="object",
        )
        return out

    def fit_transform(self, X: pd.DataFrame, y=None, **kwargs) -> pd.DataFrame:
        """Fit then transform in one step."""
        return self.fit(X, y).transform(X)

    def get_feature_names_out(
        self, input_features: Iterable[str] | None = None
    ) -> np.ndarray:
        """Return output column names for sklearn pipelines."""
        return np.array(list(_OUTPUT_COLUMNS), dtype=object)

    def _check_columns(self, X: pd.DataFrame) -> None:
        if self.mic_col not in X.columns:
            raise KeyError(
                f"MIC column {self.mic_col!r} not found in input. "
                f"Available columns: {list(X.columns)}"
            )
        if self.species_col is not None and self.species_col not in X.columns:
            raise KeyError(
                f"species_col {self.species_col!r} not found in input. "
                f"Available columns: {list(X.columns)}"
            )
        if self.drug_col is not None and self.drug_col not in X.columns:
            raise KeyError(
                f"drug_col {self.drug_col!r} not found in input. "
                f"Available columns: {list(X.columns)}"
            )

    def _species_source(self) -> str | None:
        return self.species_col if self.species_col is not None else self.species

    def _drug_source(self) -> str | None:
        return self.drug_col if self.drug_col is not None else self.drug

    @staticmethod
    def _resolve_array(
        X: pd.DataFrame, scalar: str | None, col: str | None
    ) -> np.ndarray:
        if col is not None:
            return X[col].to_numpy(dtype=object)
        return np.full(len(X), scalar, dtype=object)
