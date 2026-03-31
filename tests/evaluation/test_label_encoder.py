"""Unit tests for LabelEncoder."""

import numpy as np
import pandas as pd
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

    def test_unrecognized_label_raises(self):
        enc = LabelEncoder()
        with pytest.raises(ValueError, match="Unrecognized"):
            enc.transform(["R", "X", "unknown"])

    def test_transform_dataframe(self):
        enc = LabelEncoder()
        df = pd.DataFrame({"Drug1": ["R", "S", "R"], "Drug2": ["S", "R", "S"]})
        result = enc.fit_transform(df)
        assert isinstance(result, pd.DataFrame)
        assert list(result.columns) == ["Drug1", "Drug2"]
        np.testing.assert_array_equal(result["Drug1"].values, [1, 0, 1])
        np.testing.assert_array_equal(result["Drug2"].values, [0, 1, 0])

    def test_transform_2d_ndarray(self):
        enc = LabelEncoder()
        arr = np.array([["R", "S"], ["S", "R"], ["R", "R"]])
        result = enc.fit_transform(arr)
        assert isinstance(result, pd.DataFrame)
        assert result.shape == (3, 2)
        np.testing.assert_array_equal(result[0].values, [1, 0, 1])
        np.testing.assert_array_equal(result[1].values, [0, 1, 1])

    def test_intermediate_nan_1d(self):
        enc = LabelEncoder(intermediate="nan")
        result = enc.fit_transform(["R", "I", "S", "I"])
        assert result.dtype == np.float64
        assert result[0] == 1.0
        assert np.isnan(result[1])
        assert result[2] == 0.0
        assert np.isnan(result[3])

    def test_intermediate_nan_2d(self):
        enc = LabelEncoder(intermediate="nan")
        df = pd.DataFrame({"D1": ["R", "I", "S"], "D2": ["S", "S", "I"]})
        result = enc.fit_transform(df)
        assert isinstance(result, pd.DataFrame)
        assert result.loc[0, "D1"] == 1.0
        assert np.isnan(result.loc[1, "D1"])
        assert result.loc[1, "D2"] == 0.0
        assert np.isnan(result.loc[2, "D2"])

    def test_1d_returns_ndarray(self):
        enc = LabelEncoder()
        result = enc.fit_transform(["R", "S"])
        assert isinstance(result, np.ndarray)
        assert result.ndim == 1
