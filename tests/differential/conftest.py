"""Shared fixtures for differential-analysis tests."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


def _make_synthetic_differential_dataset(
    n_samples: int = 60,
    n_features: int = 40,
    n_true_markers: int = 5,
    effect: float = 3.0,
    random_state: int = 0,
) -> tuple[pd.DataFrame, pd.Series, list[str]]:
    """Synthetic binned matrix with injected resistant-vs-susceptible signal.

    Half the samples are resistant (label ``1``); the remaining half are
    susceptible (label ``0``).  ``n_true_markers`` columns are boosted in
    the resistant group to produce clear differential peaks; all other
    columns carry only class-independent noise.
    """
    rng = np.random.default_rng(random_state)
    X = rng.normal(loc=1.0, scale=0.2, size=(n_samples, n_features))
    X = np.abs(X)

    labels = np.zeros(n_samples, dtype=int)
    labels[: n_samples // 2] = 1
    rng.shuffle(labels)

    marker_cols = list(range(n_true_markers))
    for j in marker_cols:
        X[labels == 1, j] += effect

    columns = [f"{2000 + 3 * j}" for j in range(n_features)]
    index = [f"sample_{i}" for i in range(n_samples)]
    X_df = pd.DataFrame(X, index=index, columns=columns)
    y = pd.Series(labels, index=index, name="resistance")

    marker_names = [columns[j] for j in marker_cols]
    return X_df, y, marker_names


@pytest.fixture
def differential_dataset() -> tuple[pd.DataFrame, pd.Series, list[str]]:
    """Default synthetic differential-analysis dataset."""
    return _make_synthetic_differential_dataset()


@pytest.fixture
def differential_dataset_single_class() -> tuple[pd.DataFrame, pd.Series]:
    """Dataset where all labels are identical (for error-path tests)."""
    X, _, _ = _make_synthetic_differential_dataset(n_samples=20, n_features=10)
    y = pd.Series(np.ones(20, dtype=int), index=X.index)
    return X, y


@pytest.fixture
def differential_dataset_tiny() -> tuple[pd.DataFrame, pd.Series]:
    """Very small dataset for edge cases (near-empty groups)."""
    rng = np.random.default_rng(7)
    X = pd.DataFrame(
        rng.uniform(0.1, 2.0, size=(4, 6)),
        index=[f"s{i}" for i in range(4)],
        columns=[f"{2000 + 3 * j}" for j in range(6)],
    )
    y = pd.Series([0, 0, 1, 1], index=X.index)
    return X, y
