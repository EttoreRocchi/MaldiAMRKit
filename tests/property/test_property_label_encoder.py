"""Property-based tests for LabelEncoder invariants."""

from __future__ import annotations

import numpy as np
from hypothesis import given, settings
from hypothesis import strategies as st

from maldiamrkit.susceptibility import LabelEncoder


@given(n=st.integers(min_value=1, max_value=100))
@settings(max_examples=30)
def test_all_resistant_encodes_to_ones(n):
    """All 'R' labels must encode to all 1s."""
    y = np.array(["R"] * n)
    enc = LabelEncoder()
    result = enc.fit_transform(y)
    assert (result == 1).all()


@given(n=st.integers(min_value=1, max_value=100))
@settings(max_examples=30)
def test_all_susceptible_encodes_to_zeros(n):
    """All 'S' labels must encode to all 0s."""
    y = np.array(["S"] * n)
    enc = LabelEncoder()
    result = enc.fit_transform(y)
    assert (result == 0).all()


@given(
    n=st.integers(min_value=2, max_value=50),
    intermediate=st.sampled_from(["susceptible", "resistant", "nan"]),
)
@settings(max_examples=30)
def test_output_length_preserved_non_drop(n, intermediate):
    """Output length must equal input length when intermediate != 'drop'."""
    rng = np.random.default_rng(42)
    y = rng.choice(["R", "S", "I"], n)
    enc = LabelEncoder(intermediate=intermediate)
    result = enc.fit_transform(y)
    assert len(result) == n
