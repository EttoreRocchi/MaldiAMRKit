"""Property-based tests for filter composition invariants."""

from __future__ import annotations

import numpy as np
import pandas as pd
from hypothesis import given, settings
from hypothesis import strategies as st

from maldiamrkit.filters import MetadataFilter, SpeciesFilter


def _make_dataset_meta(n: int, rng: np.random.Generator) -> pd.DataFrame:
    """Create a synthetic metadata DataFrame for filter testing."""
    species = rng.choice(["E. coli", "K. pneumoniae", "S. aureus"], n)
    values = rng.uniform(0, 100, n)
    return pd.DataFrame(
        {
            "ID": [f"s{i}" for i in range(n)],
            "Species": species,
            "value": values,
        }
    )


def _apply_filter(f, meta: pd.DataFrame) -> pd.Series:
    """Apply a row-wise filter to all rows, returning a boolean Series."""
    return meta.apply(f, axis=1)


@given(seed=st.integers(min_value=0, max_value=1000))
@settings(max_examples=30)
def test_and_filter_subset_of_both(seed):
    """(f1 & f2) result must be a subset of both f1 and f2 individually."""
    rng = np.random.default_rng(seed)
    meta = _make_dataset_meta(50, rng)
    f1 = SpeciesFilter("E. coli")
    f2 = MetadataFilter("value", lambda x: x > 50)
    combined = f1 & f2
    mask_f1 = _apply_filter(f1, meta)
    mask_f2 = _apply_filter(f2, meta)
    mask_combined = _apply_filter(combined, meta)
    # Combined must be subset of both
    assert (mask_combined & ~mask_f1).sum() == 0
    assert (mask_combined & ~mask_f2).sum() == 0


@given(seed=st.integers(min_value=0, max_value=1000))
@settings(max_examples=30)
def test_or_filter_superset_of_both(seed):
    """(f1 | f2) result must be a superset of both f1 and f2."""
    rng = np.random.default_rng(seed)
    meta = _make_dataset_meta(50, rng)
    f1 = SpeciesFilter("E. coli")
    f2 = MetadataFilter("value", lambda x: x > 50)
    combined = f1 | f2
    mask_f1 = _apply_filter(f1, meta)
    mask_f2 = _apply_filter(f2, meta)
    mask_combined = _apply_filter(combined, meta)
    # f1 and f2 must both be subsets of combined
    assert (mask_f1 & ~mask_combined).sum() == 0
    assert (mask_f2 & ~mask_combined).sum() == 0


@given(seed=st.integers(min_value=0, max_value=1000))
@settings(max_examples=30)
def test_not_filter_complement(seed):
    """~f must be the complement of f."""
    rng = np.random.default_rng(seed)
    meta = _make_dataset_meta(50, rng)
    f = SpeciesFilter("E. coli")
    mask = _apply_filter(f, meta)
    not_mask = _apply_filter(~f, meta)
    assert (mask | not_mask).all(), "Union of f and ~f should cover all rows"
    assert not (mask & not_mask).any(), "Intersection of f and ~f should be empty"
