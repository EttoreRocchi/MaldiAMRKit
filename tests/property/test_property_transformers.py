"""Property-based tests for preprocessing transformer invariants."""

from __future__ import annotations

import numpy as np
import pandas as pd
from hypothesis import given, settings
from hypothesis import strategies as st

from maldiamrkit.preprocessing.transformers import (
    ClipNegatives,
    SqrtTransform,
    TICNormalizer,
)


def _make_spectrum_df(intensities: np.ndarray) -> pd.DataFrame:
    """Create a spectrum DataFrame from intensity values."""
    n = len(intensities)
    mass = np.linspace(2000, 20000, n)
    return pd.DataFrame({"mass": mass, "intensity": intensities})


@given(seed=st.integers(min_value=0, max_value=1000))
@settings(max_examples=30)
def test_clip_negatives_no_negative_output(seed):
    """ClipNegatives output must have no negative values."""
    rng = np.random.default_rng(seed)
    intensities = rng.normal(0, 100, 100)  # Some will be negative
    df = _make_spectrum_df(intensities)
    result = ClipNegatives()(df)
    assert (result["intensity"] >= 0).all()


@given(seed=st.integers(min_value=0, max_value=1000))
@settings(max_examples=30)
def test_sqrt_transform_nonneg_output(seed):
    """SqrtTransform output must be >= 0 for non-negative input."""
    rng = np.random.default_rng(seed)
    intensities = np.abs(rng.normal(0, 100, 100))
    df = _make_spectrum_df(intensities)
    result = SqrtTransform()(df)
    assert (result["intensity"] >= 0).all()


@given(seed=st.integers(min_value=0, max_value=1000))
@settings(max_examples=30)
def test_tic_normalizer_sums_to_one(seed):
    """TICNormalizer output must sum to ~1.0 when input sum > 0."""
    rng = np.random.default_rng(seed)
    intensities = np.abs(rng.normal(10, 100, 100))  # all positive
    df = _make_spectrum_df(intensities)
    result = TICNormalizer()(df)
    total = result["intensity"].sum()
    assert abs(total - 1.0) < 1e-10
