"""Shared fixtures for similarity tests."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from maldiamrkit.spectrum import MaldiSpectrum

# Reuse the synthetic spectrum generator from the top-level conftest.
from tests.conftest import _generate_synthetic_spectrum


@pytest.fixture
def spectrum_pair() -> tuple[MaldiSpectrum, MaldiSpectrum]:
    """Two MaldiSpectrum objects with different random seeds."""
    df_a = _generate_synthetic_spectrum(random_state=42)
    df_b = _generate_synthetic_spectrum(random_state=123)
    spec_a = MaldiSpectrum(df_a)
    spec_b = MaldiSpectrum(df_b)
    return spec_a, spec_b


@pytest.fixture
def binned_pair() -> tuple[np.ndarray, np.ndarray]:
    """Two known 1-D intensity vectors for deterministic assertions."""
    a = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    b = np.array([5.0, 4.0, 3.0, 2.0, 1.0])
    return a, b


@pytest.fixture
def distance_matrix_3clusters() -> tuple[np.ndarray, np.ndarray]:
    """15-point distance matrix with 3 well-separated clusters.

    Returns ``(distance_matrix, true_labels)``.
    """
    rng = np.random.default_rng(42)

    # 3 clusters of 5 points each in 10-D space.
    centers = np.array(
        [
            [0.0] * 10,
            [10.0] * 10,
            [20.0] * 10,
        ]
    )
    points = np.vstack([centers[k] + rng.normal(0, 0.5, (5, 10)) for k in range(3)])
    true_labels = np.array([0] * 5 + [1] * 5 + [2] * 5)

    # Euclidean distance matrix.
    from scipy.spatial.distance import cdist

    D = cdist(points, points, metric="euclidean")
    return D, true_labels


@pytest.fixture
def small_binned_df() -> pd.DataFrame:
    """5-sample, 100-feature binned DataFrame."""
    rng = np.random.default_rng(42)
    data = rng.exponential(1.0, (5, 100))
    data = data / data.sum(axis=1, keepdims=True)
    cols = [str(2000 + i * 3) for i in range(100)]
    return pd.DataFrame(data, columns=cols, index=[f"s{i}" for i in range(5)])
