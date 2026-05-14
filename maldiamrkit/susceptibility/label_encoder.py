"""Label encoding utilities for AMR classification.

Maps clinical resistance categories (R/I/S) to binary or numeric labels.
Previously lived under ``maldiamrkit.evaluation``; moved here in v0.15 to
group together with :class:`~maldiamrkit.susceptibility.MICEncoder` and
:class:`~maldiamrkit.susceptibility.BreakpointTable`. The old import path
still works for one release with a :class:`DeprecationWarning`.

Examples
--------
>>> from maldiamrkit.susceptibility import LabelEncoder
>>> enc = LabelEncoder()
>>> enc.fit_transform(["R", "S", "I", "R", "S"])
array([1, 0, 0, 1, 0])

>>> enc = LabelEncoder(intermediate="resistant")
>>> enc.transform(["I"])
array([1])
"""

from __future__ import annotations

import warnings
from enum import Enum

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class IntermediateHandling(str, Enum):
    """Strategy for handling intermediate (I) resistance labels.

    Attributes
    ----------
    susceptible : str
        Map intermediate to susceptible (0).
    resistant : str
        Map intermediate to resistant (1).
    drop : str
        Remove intermediate samples.
    nan : str
        Map intermediate to NaN.
    """

    susceptible = "susceptible"
    resistant = "resistant"
    drop = "drop"
    nan = "nan"


class LabelEncoder(BaseEstimator, TransformerMixin):
    """Encode R/I/S resistance labels to binary (0/1).

    Supports configurable handling of intermediate (I) labels.
    Accepts both 1-D arrays (single drug) and 2-D DataFrames
    (multiple drugs).

    Parameters
    ----------
    intermediate : str, default="susceptible"
        How to handle intermediate ("I") labels:

        - ``"susceptible"``: treat I as susceptible (0) - conservative,
          avoids false resistance calls.
        - ``"resistant"``: treat I as resistant (1) - stricter, avoids
          missing resistance.
        - ``"drop"``: remove samples with I labels entirely.
          Note: this changes the output array length (samples with I
          labels are excluded) and is not compatible with sklearn
          pipelines that expect consistent sample counts.
        - ``"nan"``: map I to ``NaN``. Useful for multi-drug encoding
          where each drug is handled independently. Output dtype is
          ``float64`` (required to hold ``NaN``).

    Attributes
    ----------
    classes_ : ndarray
        Array of ``[0, 1]`` after fitting.

    Raises
    ------
    ValueError
        If ``intermediate`` is not one of the accepted values.
    """

    _RESISTANT = {"R", "r", "resistant", "Resistant"}
    _SUSCEPTIBLE = {"S", "s", "susceptible", "Susceptible"}
    _INTERMEDIATE = {"I", "i", "intermediate", "Intermediate"}

    def __init__(
        self,
        intermediate: str | IntermediateHandling = IntermediateHandling.susceptible,
    ) -> None:
        self.intermediate = IntermediateHandling(intermediate)

    def fit(self, y: np.ndarray | pd.DataFrame, **kwargs: object) -> LabelEncoder:
        """Fit the encoder (no-op, just sets ``classes_``).

        Parameters
        ----------
        y : array-like
            Labels to learn from (unused beyond validation).
        **kwargs : dict
            Additional keyword arguments (unused, accepted for sklearn
            compatibility).

        Returns
        -------
        self
        """
        self.classes_ = np.array([0, 1])
        return self

    def transform(self, y: np.ndarray | pd.DataFrame) -> np.ndarray | pd.DataFrame:
        """Transform labels to binary.

        Parameters
        ----------
        y : array-like or pd.DataFrame
            String labels (R/I/S or resistant/intermediate/susceptible).
            If a DataFrame is passed, each column is encoded independently.

        Returns
        -------
        ndarray or pd.DataFrame
            Binary encoded labels. Returns a DataFrame when the input is
            a DataFrame (or a 2-D ndarray), preserving column names and
            index. Returns a 1-D ndarray for 1-D input.
        """
        if isinstance(y, pd.DataFrame):
            return y.apply(self._encode_column)
        arr = np.asarray(y)
        if arr.ndim == 2:
            return pd.DataFrame(
                {i: self._encode_column(arr[:, i]) for i in range(arr.shape[1])}
            )
        return self._encode_column(arr)

    def _encode_column(self, y: np.ndarray) -> np.ndarray:
        """Encode a single 1-D label array."""
        y = np.asarray(y)
        result = np.full(len(y), np.nan)

        for i, label in enumerate(y):
            result[i] = self._classify_label(str(label))

        self._validate_unrecognized(y, result)

        if self.intermediate == "drop":
            mask = ~np.isnan(result)
            n_dropped = int((~mask).sum())
            if n_dropped:
                warnings.warn(
                    f"intermediate='drop' removed {n_dropped} intermediate "
                    "samples. Output length differs from input - incompatible "
                    "with sklearn pipelines that expect consistent sample counts.",
                    UserWarning,
                    stacklevel=2,
                )
            return result[mask].astype(int)
        if self.intermediate == "nan":
            return result
        return result.astype(int)

    def _classify_label(self, label_str: str) -> float:
        """Map a single label string to its numeric value (1, 0, or NaN)."""
        if label_str in self._RESISTANT:
            return 1.0
        if label_str in self._SUSCEPTIBLE:
            return 0.0
        if label_str in self._INTERMEDIATE:
            if self.intermediate == "susceptible":
                return 0.0
            if self.intermediate == "resistant":
                return 1.0
        return float("nan")

    def _validate_unrecognized(self, y: np.ndarray, result: np.ndarray) -> None:
        """Raise ValueError if any labels are unrecognized (unexpected NaN)."""
        unrecognized_mask = np.isnan(result)
        if self.intermediate in ("drop", "nan"):
            intermediate_mask = np.array(
                [str(label) in self._INTERMEDIATE for label in y]
            )
            unexpected_nan = unrecognized_mask & ~intermediate_mask
        else:
            unexpected_nan = unrecognized_mask

        if unexpected_nan.any():
            bad_labels = [str(y[i]) for i in np.where(unexpected_nan)[0]]
            raise ValueError(
                f"Unrecognized labels: {bad_labels}. "
                f"Expected one of R/S/I (or resistant/susceptible/intermediate)."
            )

    def fit_transform(self, y, **kwargs):
        """Fit the encoder and transform labels in one step.

        Parameters
        ----------
        y : array-like or pd.DataFrame
            String labels (R/I/S or resistant/intermediate/susceptible).
            If a DataFrame is passed, each column is encoded independently.
        **kwargs : dict
            Additional keyword arguments (unused, accepted for sklearn
            compatibility).

        Returns
        -------
        ndarray or pd.DataFrame
            Binary encoded labels. Returns a DataFrame when the input is
            a DataFrame, preserving column names and index.
        """
        return self.fit(y).transform(y)
