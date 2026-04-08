"""Tests for BINNING_REGISTRY and edge-generator wrapper functions."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from maldiamrkit.preprocessing.binning import (
    BINNING_REGISTRY,
    _adaptive_edge_fn,
    _custom_edge_fn,
    _proportional_edge_fn,
    _proportional_edges,
    _uniform_edge_fn,
    _validate_custom_edges,
    bin_spectrum,
)


class TestUniformEdgeFn:
    """Tests for _uniform_edge_fn wrapper."""

    def test_valid_input(self):
        edges = _uniform_edge_fn(mz_min=2000, mz_max=2030, bin_width=3)
        assert edges[0] == 2000
        assert edges[-1] >= 2030

    def test_bin_width_below_one_raises(self):
        with pytest.raises(ValueError, match="bin_width must be >= 1"):
            _uniform_edge_fn(mz_min=2000, mz_max=2030, bin_width=0.5)


class TestProportionalEdgeFn:
    """Tests for _proportional_edge_fn wrapper."""

    def test_valid_input(self):
        edges = _proportional_edge_fn(mz_min=2000, mz_max=5000, bin_width=3)
        assert edges[0] == 2000
        assert edges[-1] >= 5000

    def test_bin_width_below_one_raises(self):
        with pytest.raises(ValueError, match="bin_width must be >= 1"):
            _proportional_edge_fn(mz_min=2000, mz_max=5000, bin_width=0.5)


class TestProportionalEdgesLastEdge:
    """Test _proportional_edges when last edge < mz_max."""

    def test_last_edge_appended_when_below_mz_max(self):
        edges = _proportional_edges(mz_min=2000, mz_max=2010, bin_width=3)
        assert edges[-1] >= 2010


class TestAdaptiveEdgeFn:
    """Tests for _adaptive_edge_fn wrapper."""

    def test_min_width_below_one_raises(self):
        with pytest.raises(ValueError, match="adaptive_min_width must be >= 1"):
            _adaptive_edge_fn(
                df=pd.DataFrame({"mass": [2000, 3000], "intensity": [1.0, 1.0]}),
                mz_min=2000,
                mz_max=3000,
                adaptive_min_width=0.5,
            )


class TestCustomEdgeFn:
    """Tests for _custom_edge_fn wrapper."""

    def test_none_edges_raises(self):
        with pytest.raises(ValueError, match="custom_edges is required"):
            _custom_edge_fn(mz_min=2000, mz_max=20000)


class TestValidateCustomEdges:
    """Tests for _validate_custom_edges edge cases."""

    def test_too_few_edges(self):
        with pytest.raises(ValueError, match="at least 2"):
            _validate_custom_edges([2000], 2000, 20000)

    def test_unsorted_edges(self):
        with pytest.raises(ValueError, match="sorted"):
            _validate_custom_edges([20000, 2000], 2000, 20000)

    def test_first_edge_above_mz_min(self):
        with pytest.raises(ValueError, match="First edge"):
            _validate_custom_edges([3000, 20000], 2000, 20000)

    def test_last_edge_below_mz_max(self):
        with pytest.raises(ValueError, match="Last edge"):
            _validate_custom_edges([2000, 10000], 2000, 20000)

    def test_too_narrow_bins(self):
        with pytest.raises(ValueError, match="1 Dalton"):
            _validate_custom_edges([2000, 2000.5, 20000], 2000, 20000)


class TestBinningRegistryExtensibility:
    """Tests for BINNING_REGISTRY as a public API."""

    def test_all_methods_registered(self):
        assert set(BINNING_REGISTRY.keys()) == {
            "uniform",
            "proportional",
            "adaptive",
            "custom",
        }

    def test_custom_method_registration(self):
        """Custom registry keys not in the Enum are rejected by Enum coercion."""

        def my_edges(*, mz_min, mz_max, **kwargs):
            return np.array([mz_min, (mz_min + mz_max) / 2, mz_max])

        BINNING_REGISTRY["test_method"] = my_edges
        try:
            df = pd.DataFrame(
                {
                    "mass": np.linspace(2000, 20000, 100),
                    "intensity": np.ones(100),
                }
            )
            with pytest.raises(ValueError, match="is not a valid"):
                bin_spectrum(df, method="test_method")
        finally:
            del BINNING_REGISTRY["test_method"]

    def test_invalid_method_raises(self):
        df = pd.DataFrame({"mass": [2000, 3000], "intensity": [1.0, 1.0]})
        with pytest.raises(ValueError, match="is not a valid"):
            bin_spectrum(df, method="nonexistent")
