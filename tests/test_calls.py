"""Tests for the shared susceptibility call helpers."""

import numpy as np
import pandas as pd
import pytest

from maldiamrkit._calls import _is_call, _plain_call, _plain_calls


class TestPlainCall:
    """Tests for _plain_call and _is_call."""

    @pytest.mark.parametrize(
        ("value", "expected"),
        [
            ("R", "R"),
            ("R*", "R"),
            ("S*", "S"),
            ("I*", "I"),
            (" R ", "R"),
            (" R* ", "R"),
            ("r*", "r"),
        ],
    )
    def test_calls_are_stripped(self, value, expected):
        """Spaces and one trailing star are removed, case is kept."""
        assert _plain_call(value) == expected
        assert _is_call(value)

    @pytest.mark.parametrize("value", ["ESBL", "N/R", "X*", "R**", "", 3, 1.5])
    def test_non_calls_unchanged(self, value):
        """Values that are not S/I/R calls come back unchanged."""
        assert _plain_call(value) == value
        assert not _is_call(value)

    def test_nan_unchanged(self):
        """NaN is returned as is."""
        assert np.isnan(_plain_call(float("nan")))
        assert not _is_call(float("nan"))


class TestPlainCalls:
    """Tests for _plain_calls."""

    def test_series_stays_series(self):
        """A Series keeps its index and type."""
        s = pd.Series(["R*", "S", "ESBL"], index=[5, 6, 7])
        out = _plain_calls(s)
        assert isinstance(out, pd.Series)
        assert list(out.index) == [5, 6, 7]
        assert list(out) == ["R", "S", "ESBL"]

    def test_array_input(self):
        """Array-likes give an ndarray of plain calls."""
        out = _plain_calls(["R*", " S ", "I*"])
        assert isinstance(out, np.ndarray)
        assert list(out) == ["R", "S", "I"]

    def test_numeric_unchanged(self):
        """Numeric arrays and Series are returned unchanged."""
        arr = np.array([0, 1, 1])
        assert _plain_calls(arr) is arr
        s = pd.Series([0, 1])
        assert _plain_calls(s) is s

    def test_object_array_with_nan(self):
        """Object arrays with NaN and numbers keep those values."""
        out = _plain_calls(np.array(["R*", np.nan, 1], dtype=object))
        assert out[0] == "R"
        assert np.isnan(out[1])
        assert out[2] == 1
