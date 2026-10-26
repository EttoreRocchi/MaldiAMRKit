"""Warping with a fractional ``max_shift``.

Only public names that exist in every release are used, so the same module
can be run against older versions.
"""

import numpy as np
import pandas as pd
import pytest

from maldiamrkit.alignment import Warping
from maldiamrkit.detection import MaldiPeakDetector


def _dataset(centres: list[int], n_bins: int = 400) -> pd.DataFrame:
    """Gaussian peaks at ``centres``, in a reference row and two shifted rows."""
    x = np.arange(n_bins)
    rows = [
        sum(np.exp(-0.5 * ((x - c - shift) / 3.0) ** 2) for c in centres)
        for shift in (0, 2, -2)
    ]
    return pd.DataFrame(rows, columns=[str(i) for i in range(n_bins)])


def _check(method: str, centres: list[int]) -> None:
    X = _dataset(centres)
    warper = Warping(
        method=method,
        max_shift=2.5,
        reference=0,
        peak_detector=MaldiPeakDetector(binary=True, prominence=0.1),
        min_reference_peaks=1,
    )
    out = warper.fit_transform(X)
    assert out.shape == X.shape
    moved = np.argmax(out.to_numpy(), axis=1) - np.argmax(X.to_numpy(), axis=1)
    assert np.all(np.abs(moved) <= 2)


@pytest.mark.parametrize("method", ["shift", "linear", "quadratic", "cubic", "lowess"])
def test_single_peak(method):
    _check(method, [200])


@pytest.mark.parametrize("method", ["piecewise", "shift", "lowess"])
def test_four_peaks(method):
    _check(method, [80, 160, 240, 320])
