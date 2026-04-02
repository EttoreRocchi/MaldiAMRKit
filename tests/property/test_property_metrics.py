"""Property-based tests for AMR evaluation metric invariants."""

from __future__ import annotations

import numpy as np
from hypothesis import given, settings
from hypothesis import strategies as st

from maldiamrkit.evaluation.metrics import (
    categorical_agreement,
    major_error_rate,
    sensitivity_score,
    specificity_score,
    very_major_error_rate,
)


@st.composite
def binary_labels(draw, min_size=5):
    """Strategy for generating binary label arrays with both classes."""
    n = draw(st.integers(min_value=min_size, max_value=100))
    # Ensure at least one of each class
    y = draw(st.lists(st.integers(min_value=0, max_value=1), min_size=n, max_size=n))
    y = np.array(y)
    if not (y == 1).any():
        y[0] = 1
    if not (y == 0).any():
        y[-1] = 0
    return y


@given(y_true=binary_labels(), y_pred=binary_labels())
@settings(max_examples=50)
def test_vme_plus_sensitivity_equals_one(y_true, y_pred):
    """VME + sensitivity = 1 when at least one positive exists."""
    # Ensure same length
    n = min(len(y_true), len(y_pred))
    y_true, y_pred = y_true[:n], y_pred[:n]
    if not (y_true == 1).any():
        return  # Skip if no positives
    vme = very_major_error_rate(y_true, y_pred)
    sens = sensitivity_score(y_true, y_pred)
    assert abs(vme + sens - 1.0) < 1e-10


@given(y_true=binary_labels(), y_pred=binary_labels())
@settings(max_examples=50)
def test_me_plus_specificity_equals_one(y_true, y_pred):
    """ME + specificity = 1 when at least one negative exists."""
    n = min(len(y_true), len(y_pred))
    y_true, y_pred = y_true[:n], y_pred[:n]
    if not (y_true == 0).any():
        return  # Skip if no negatives
    me = major_error_rate(y_true, y_pred)
    spec = specificity_score(y_true, y_pred)
    assert abs(me + spec - 1.0) < 1e-10


@given(y_true=binary_labels(), y_pred=binary_labels())
@settings(max_examples=50)
def test_categorical_agreement_in_unit_range(y_true, y_pred):
    """Categorical agreement must be in [0, 1]."""
    n = min(len(y_true), len(y_pred))
    y_true, y_pred = y_true[:n], y_pred[:n]
    ca = categorical_agreement(y_true, y_pred)
    assert 0.0 <= ca <= 1.0
