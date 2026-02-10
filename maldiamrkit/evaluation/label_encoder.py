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
from sklearn.base import BaseEstimator, TransformerMixin


class LabelEncoder(BaseEstimator, TransformerMixin):
    """Encode R/I/S resistance labels to binary (0/1).

    Supports configurable handling of intermediate (I) labels.

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

    Attributes
    ----------
    classes_ : ndarray
        Array of ``[0, 1]`` after fitting.

    Raises
    ------
    ValueError
        If ``intermediate`` is not one of 'susceptible', 'resistant',
        or 'drop'.
    """

    _RESISTANT = {"R", "r", "resistant", "Resistant"}
    _SUSCEPTIBLE = {"S", "s", "susceptible", "Susceptible"}
    _INTERMEDIATE = {"I", "i", "intermediate", "Intermediate"}

    def __init__(self, intermediate: str = "susceptible") -> None:
        if intermediate not in ("susceptible", "resistant", "drop"):
            raise ValueError(
                f"intermediate must be 'susceptible', 'resistant', or 'drop', "
                f"got {intermediate!r}"
            )
        self.intermediate = intermediate

    def fit(self, y, **kwargs):
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

    def transform(self, y):
        """Transform labels to binary.

        Parameters
        ----------
        y : array-like
            String labels (R/I/S or resistant/intermediate/susceptible).

        Returns
        -------
        ndarray
            Binary array: 1 for resistant, 0 for susceptible.
            If ``intermediate="drop"``, intermediate samples are
            removed from the output (the returned array will be
            shorter than the input).
        """
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
                # else: drop → stays NaN

        if self.intermediate == "drop":
            mask = ~np.isnan(result)
            return result[mask].astype(int)
        return result.astype(int)

    def fit_transform(self, y, **kwargs):
        """Fit and transform in one step."""
        return self.fit(y).transform(y)
