"""Property-based tests for spectral distance metric invariants."""

from __future__ import annotations

import numpy as np
import pandas as pd
from hypothesis import given, settings
from hypothesis import strategies as st

from maldiamrkit.similarity.metrics import (
    _cosine_distance,
    _pearson_correlation,
    _spectral_contrast_angle,
)
from maldiamrkit.similarity.pairwise import pairwise_distances


def _make_positive_vector(seed: int, size: int = 50) -> np.ndarray:
    """Generate a strictly positive random vector (simulates intensities)."""
    rng = np.random.default_rng(seed)
    return rng.exponential(1.0, size) + 1e-8


@given(seed=st.integers(min_value=0, max_value=10_000))
@settings(max_examples=50)
def test_cosine_distance_self_is_zero(seed):
    """Cosine distance of a vector with itself must be ~0."""
    x = _make_positive_vector(seed)
    d = _cosine_distance(x, x)
    assert abs(d) < 1e-10, f"cosine(x, x) = {d}, expected ~0"


@given(seed=st.integers(min_value=0, max_value=10_000))
@settings(max_examples=50)
def test_cosine_distance_symmetric(seed):
    """Cosine distance must be symmetric: d(x, y) == d(y, x)."""
    x = _make_positive_vector(seed)
    y = _make_positive_vector(seed + 1)
    assert abs(_cosine_distance(x, y) - _cosine_distance(y, x)) < 1e-12


@given(seed=st.integers(min_value=0, max_value=10_000))
@settings(max_examples=50)
def test_cosine_distance_nonnegative(seed):
    """Cosine distance must be >= 0."""
    x = _make_positive_vector(seed)
    y = _make_positive_vector(seed + 1)
    assert _cosine_distance(x, y) >= -1e-12


@given(seed=st.integers(min_value=0, max_value=10_000))
@settings(max_examples=50)
def test_pearson_distance_self_is_zero(seed):
    """Pearson distance of a vector with itself must be ~0."""
    x = _make_positive_vector(seed)
    d = _pearson_correlation(x, x)
    assert abs(d) < 1e-10, f"pearson(x, x) = {d}, expected ~0"


@given(seed=st.integers(min_value=0, max_value=10_000))
@settings(max_examples=50)
def test_pearson_distance_symmetric(seed):
    """Pearson distance must be symmetric."""
    x = _make_positive_vector(seed)
    y = _make_positive_vector(seed + 1)
    assert abs(_pearson_correlation(x, y) - _pearson_correlation(y, x)) < 1e-12


@given(seed=st.integers(min_value=0, max_value=10_000))
@settings(max_examples=50)
def test_spectral_contrast_angle_symmetric(seed):
    """Spectral contrast angle must be symmetric."""
    x = _make_positive_vector(seed)
    y = _make_positive_vector(seed + 1)
    d_xy = _spectral_contrast_angle(x, y)
    d_yx = _spectral_contrast_angle(y, x)
    assert abs(d_xy - d_yx) < 1e-12


@given(seed=st.integers(min_value=0, max_value=10_000))
@settings(max_examples=50)
def test_spectral_contrast_angle_self_is_zero(seed):
    """Spectral contrast angle of a vector with itself must be ~0."""
    x = _make_positive_vector(seed)
    d = _spectral_contrast_angle(x, x)
    assert abs(d) < 1e-7


@given(seed=st.integers(min_value=0, max_value=10_000))
@settings(max_examples=20)
def test_pairwise_matrix_symmetric(seed):
    """Pairwise distance matrix must be symmetric."""
    rng = np.random.default_rng(seed)
    n_samples, n_features = 5, 30
    X = pd.DataFrame(
        rng.exponential(1.0, (n_samples, n_features)) + 1e-8,
        columns=[str(i) for i in range(n_features)],
        index=[f"s{i}" for i in range(n_samples)],
    )
    D = pairwise_distances(X, metric="cosine")
    np.testing.assert_allclose(D, D.T, atol=1e-12)


@given(seed=st.integers(min_value=0, max_value=10_000))
@settings(max_examples=20)
def test_pairwise_matrix_diagonal_zero(seed):
    """Pairwise distance matrix diagonal must be ~0."""
    rng = np.random.default_rng(seed)
    n_samples, n_features = 5, 30
    X = pd.DataFrame(
        rng.exponential(1.0, (n_samples, n_features)) + 1e-8,
        columns=[str(i) for i in range(n_features)],
        index=[f"s{i}" for i in range(n_samples)],
    )
    D = pairwise_distances(X, metric="cosine")
    np.testing.assert_allclose(np.diag(D), 0.0, atol=1e-10)
