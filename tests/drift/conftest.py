"""Shared fixtures for drift-monitoring tests."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import pytest


@dataclass
class _FakeMaldiSet:
    """Minimal MaldiSet shim exposing ``.X``, ``.meta``, ``.get_y_single``."""

    X: pd.DataFrame
    meta: pd.DataFrame

    def get_y_single(self, antibiotic: str | None = None) -> pd.Series:
        column = antibiotic or "Drug"
        return self.meta.loc[self.X.index, column]


def _make_drift_dataset(
    n_samples: int = 120,
    n_features: int = 50,
    n_days: int = 180,
    inject_drift: bool = False,
    drift_scale: float = 2.0,
    inject_peaks: bool = False,
    random_state: int = 0,
) -> _FakeMaldiSet:
    """Build a synthetic timestamped dataset suitable for drift tests.

    - n_samples rows, n_features columns (m/z-like labels)
    - Rows are spaced roughly evenly across n_days
    - If inject_drift: post-halfway rows are shifted on the first 5
      features by drift_scale * baseline-std (detectable in PCA)
    - If inject_peaks: 5 features boosted in the resistant (y=1)
      class consistently across all windows (stable top-k peaks)
    - A ``Drug`` column with balanced binary labels (0=S, 1=R)
    """
    rng = np.random.default_rng(random_state)
    X = rng.normal(loc=0.5, scale=0.1, size=(n_samples, n_features))
    X = np.abs(X)

    ids = [f"sample_{i:03d}" for i in range(n_samples)]
    start = pd.Timestamp("2025-01-01")
    days = np.sort(rng.integers(0, n_days, size=n_samples))
    dates = [start + pd.Timedelta(days=int(d)) for d in days]

    labels = np.array([i % 2 for i in range(n_samples)])

    if inject_peaks:
        for j in range(5):
            X[labels == 1, j] += 3.0 * X[:, j].mean()

    if inject_drift:
        halfway_day = n_days // 2
        post_mask = days >= halfway_day
        drift_cols = np.arange(5)
        col_std = X[:, drift_cols].std(axis=0) + 1e-12
        X[np.ix_(post_mask, drift_cols)] += drift_scale * col_std

    # TIC-normalize each row (matches real MALDI preprocessing)
    X = X / X.sum(axis=1, keepdims=True)

    columns = [str(2000 + 3 * j) for j in range(n_features)]
    X_df = pd.DataFrame(X, index=ids, columns=columns)
    meta = pd.DataFrame(
        {"Drug": labels, "acquisition_date": dates},
        index=ids,
    )
    return _FakeMaldiSet(X=X_df, meta=meta)


@pytest.fixture
def drift_set_flat():
    """No-drift synthetic dataset (60 samples, 180 days)."""
    return _make_drift_dataset(n_samples=60, n_features=40, random_state=0)


@pytest.fixture
def drift_set_shifted():
    """Dataset with injected post-baseline shift on the first 5 features."""
    return _make_drift_dataset(
        n_samples=120,
        n_features=40,
        inject_drift=True,
        drift_scale=4.0,
        random_state=1,
    )


@pytest.fixture
def drift_set_with_peaks():
    """Dataset with stable injected R-specific discriminative peaks."""
    return _make_drift_dataset(
        n_samples=120,
        n_features=40,
        inject_peaks=True,
        random_state=2,
    )
