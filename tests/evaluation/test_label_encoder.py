"""Unit tests for LabelEncoder."""

import numpy as np
import pytest

from maldiamrkit.evaluation import LabelEncoder


class TestLabelEncoder:
    """Tests for LabelEncoder."""

    def test_basic_encoding(self):
        enc = LabelEncoder()
        result = enc.fit_transform(["R", "S", "R", "S"])
        np.testing.assert_array_equal(result, [1, 0, 1, 0])

    def test_intermediate_as_susceptible(self):
        enc = LabelEncoder(intermediate="susceptible")
        result = enc.fit_transform(["R", "I", "S"])
        np.testing.assert_array_equal(result, [1, 0, 0])

    def test_intermediate_as_resistant(self):
        enc = LabelEncoder(intermediate="resistant")
        result = enc.fit_transform(["R", "I", "S"])
        np.testing.assert_array_equal(result, [1, 1, 0])

    def test_intermediate_drop(self):
        enc = LabelEncoder(intermediate="drop")
        result = enc.fit_transform(["R", "I", "S", "I", "R"])
        np.testing.assert_array_equal(result, [1, 0, 1])

    def test_full_words(self):
        enc = LabelEncoder()
        result = enc.fit_transform(["resistant", "susceptible", "intermediate"])
        np.testing.assert_array_equal(result, [1, 0, 0])

    def test_capitalized_words(self):
        enc = LabelEncoder()
        result = enc.fit_transform(["Resistant", "Susceptible", "Intermediate"])
        np.testing.assert_array_equal(result, [1, 0, 0])

    def test_lowercase(self):
        enc = LabelEncoder()
        result = enc.fit_transform(["r", "s", "i"])
        np.testing.assert_array_equal(result, [1, 0, 0])

    def test_sklearn_compatible(self):
        from sklearn.base import clone

        enc = LabelEncoder(intermediate="resistant")
        enc2 = clone(enc)
        assert enc2.intermediate == "resistant"

    def test_fit_returns_self(self):
        enc = LabelEncoder()
        result = enc.fit(["R", "S"])
        assert result is enc

    def test_classes_attribute(self):
        enc = LabelEncoder()
        enc.fit(["R", "S"])
        np.testing.assert_array_equal(enc.classes_, [0, 1])

    def test_invalid_intermediate_raises(self):
        with pytest.raises(ValueError, match="intermediate"):
            LabelEncoder(intermediate="invalid")

    def test_empty_input(self):
        enc = LabelEncoder()
        result = enc.fit_transform([])
        assert len(result) == 0
