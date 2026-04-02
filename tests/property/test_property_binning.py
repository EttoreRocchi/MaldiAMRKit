"""Property-based tests for spectrum binning invariants."""

from __future__ import annotations

import numpy as np
from hypothesis import given, settings
from hypothesis import strategies as st

from maldiamrkit.preprocessing.binning import _uniform_edges, bin_spectrum
from tests.conftest import _generate_synthetic_spectrum


@given(
    mz_min=st.integers(min_value=1000, max_value=5000),
    bin_width=st.integers(min_value=1, max_value=50),
)
@settings(max_examples=50)
def test_uniform_edges_monotonically_increasing(mz_min, bin_width):
    """Uniform bin edges must be strictly monotonically increasing."""
    mz_max = mz_min + 1000
    edges = _uniform_edges(mz_min, mz_max, bin_width)
    diffs = np.diff(edges)
    assert np.all(diffs > 0), "Edges are not monotonically increasing"


@given(
    mz_min=st.integers(min_value=1000, max_value=5000),
    bin_width=st.integers(min_value=1, max_value=50),
)
@settings(max_examples=50)
def test_uniform_edges_cover_range(mz_min, bin_width):
    """Uniform bin edges must cover the [mz_min, mz_max] range."""
    mz_max = mz_min + 1000
    edges = _uniform_edges(mz_min, mz_max, bin_width)
    assert edges[0] <= mz_min
    assert edges[-1] >= mz_max


@given(seed=st.integers(min_value=0, max_value=1000))
@settings(max_examples=20)
def test_bin_spectrum_output_nonneg(seed):
    """Binned spectrum intensities must be non-negative."""
    df = _generate_synthetic_spectrum(random_state=seed)
    binned, _ = bin_spectrum(df, bin_width=3, method="uniform")
    assert (binned["intensity"] >= 0).all()
