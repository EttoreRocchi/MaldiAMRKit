from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from scipy.signal import find_peaks


class MaldiPeakDetector(BaseEstimator, TransformerMixin):
    """
    Peak detector for MALDI-TOF spectra.

    The transformer keeps the original feature dimension; all non-peak
    positions are set to 0.  Peaks can be returned as **binary flags**
    or with their original intensities.

    Parameters
    ----------
    binary : bool, default=True
        If *True* every peak is marked with 1; otherwise its original
        intensity is kept.
    **kwargs :
        Any keyword accepted by :func:`scipy.signal.find_peaks`
        (e.g. `prominence`, `height`, `distance`, …).
    """

    def __init__(
            self,
            binary: bool = True,
            **kwargs
        ) -> MaldiPeakDetector:
        self.binary = binary
        self.kwargs = kwargs

    def fit(self, X: pd.DataFrame, y=None):
        """No learning required, just return *self*."""
        return self

    def transform(self, X: pd.DataFrame):
        """Detect peaks in *each* sample independently and mask everything else."""
        X_out = X.copy()

        for i in range(X_out.shape[0]):
            row = X_out.iloc[i].values
            peaks, _ = find_peaks(row, **self.kwargs)

            masked = np.zeros_like(row, dtype=row.dtype)
            if self.binary:
                masked[peaks] = 1
            else:
                masked[peaks] = row[peaks]
            X_out.iloc[i] = masked

        return X_out

    def fit_transform(self, X: pd.DataFrame, y=None, **fit_params):
        """Convenience shortcut."""
        return self.fit(X, y).transform(X)
