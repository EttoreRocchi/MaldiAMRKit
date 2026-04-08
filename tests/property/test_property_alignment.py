"""Property-based tests for alignment strategy invariants."""

from __future__ import annotations

import numpy as np
import pandas as pd
from hypothesis import given, settings
from hypothesis import strategies as st

from maldiamrkit.alignment.warping import Warping


def _make_shifted_dataset(seed: int, n_samples: int = 8, n_bins: int = 100):
    """Create a small binned dataset with slight random shifts."""
    rng = np.random.default_rng(seed)
    columns = [str(2000 + i * 3) for i in range(n_bins)]
    index = [f"s{i}" for i in range(n_samples)]

    # Base spectrum with peaks
    base = rng.exponential(0.01, n_bins)
    peak_pos = [20, 50, 80]
    for p in peak_pos:
        base[p] += 1.0

    # Create shifted copies
    data = np.zeros((n_samples, n_bins))
    for i in range(n_samples):
        shift = rng.integers(-2, 3)
        data[i] = np.roll(base, shift) + rng.normal(0, 0.001, n_bins)
    data = np.maximum(data, 0)

    return pd.DataFrame(data, columns=columns, index=index)


@given(seed=st.integers(min_value=0, max_value=5000))
@settings(max_examples=15, deadline=30_000)
def test_shift_preserves_shape(seed):
    """Shift alignment output must have the same shape as input."""
    X = _make_shifted_dataset(seed)
    warper = Warping(method="shift", reference="median")
    result = warper.fit_transform(X)
    assert result.shape == X.shape


@given(seed=st.integers(min_value=0, max_value=5000))
@settings(max_examples=15, deadline=30_000)
def test_linear_preserves_shape(seed):
    """Linear alignment output must have the same shape as input."""
    X = _make_shifted_dataset(seed)
    warper = Warping(method="linear", reference="median")
    result = warper.fit_transform(X)
    assert result.shape == X.shape


@given(seed=st.integers(min_value=0, max_value=5000))
@settings(max_examples=15, deadline=30_000)
def test_piecewise_preserves_shape(seed):
    """Piecewise alignment output must have the same shape as input."""
    X = _make_shifted_dataset(seed)
    warper = Warping(method="piecewise", reference="median", n_segments=3)
    result = warper.fit_transform(X)
    assert result.shape == X.shape


@given(seed=st.integers(min_value=0, max_value=5000))
@settings(max_examples=10, deadline=30_000)
def test_alignment_reference_near_identity(seed):
    """Aligning the reference to itself should produce near-identical output."""
    X = _make_shifted_dataset(seed, n_samples=5)
    warper = Warping(method="shift", reference=0)
    warper.fit(X)
    # Transform just the reference row
    ref_row = X.iloc[[0]]
    result = warper.transform(ref_row)
    # Should be very close to the original
    np.testing.assert_allclose(
        result.values,
        ref_row.values,
        atol=0.05,
        err_msg="Reference alignment should be near-identity",
    )
