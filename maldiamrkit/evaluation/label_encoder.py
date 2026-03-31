"""Label encoding utilities for AMR classification.

Maps clinical resistance categories (R/I/S) to binary or numeric labels.

Examples
--------
>>> from maldiamrkit.evaluation import LabelEncoder
>>> enc = LabelEncoder()
>>> enc.fit_transform(["R", "S", "I", "R", "S"])
array([1, 0, 0, 1, 0])

>>> enc = LabelEncoder(intermediate="resistant")
>>> enc.transform(["I"])
array([1])
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


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

    def __init__(self, intermediate: str = "susceptible") -> None:
        if intermediate not in ("susceptible", "resistant", "drop", "nan"):
            raise ValueError(
                f"intermediate must be 'susceptible', 'resistant', 'drop', "
                f"or 'nan', got {intermediate!r}"
            )
        self.intermediate = intermediate

    def fit(self, y: np.ndarray | pd.DataFrame, **kwargs: object) -> LabelEncoder:
        """Fit the encoder (no-op, just sets ``classes_``).

        Parameters
        ----------
        y : array-like
            Labels to learn from (unused beyond validation).

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
            label_str = str(label)
            if label_str in self._RESISTANT:
                result[i] = 1
            elif label_str in self._SUSCEPTIBLE:
                result[i] = 0
            elif label_str in self._INTERMEDIATE:
                if self.intermediate == "susceptible":
                    result[i] = 0
                elif self.intermediate == "resistant":
                    result[i] = 1
                # "drop" and "nan" -> stays NaN

        # Check for unrecognized labels (still NaN after encoding)
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

        if self.intermediate == "drop":
            mask = ~np.isnan(result)
            return result[mask].astype(int)
        if self.intermediate == "nan":
            return result
        return result.astype(int)

    def fit_transform(self, y, **kwargs):
        """Fit and transform in one step."""
        return self.fit(y).transform(y)
