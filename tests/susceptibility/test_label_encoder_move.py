"""Tests for LabelEncoder location move from evaluation to susceptibility."""

from __future__ import annotations

import numpy as np
import pytest


class TestNewLocation:
    def test_import_from_susceptibility(self):
        from maldiamrkit.susceptibility import IntermediateHandling, LabelEncoder

        enc = LabelEncoder()
        out = enc.fit_transform(["R", "S", "R"])
        np.testing.assert_array_equal(out, [1, 0, 1])
        assert IntermediateHandling.susceptible.value == "susceptible"


class TestOldLocationRemoved:
    """The v0.15 deprecation shims were removed in v0.17.1."""

    def test_evaluation_attribute_gone(self):
        import maldiamrkit.evaluation as ev

        with pytest.raises(AttributeError):
            _ = ev.LabelEncoder

    def test_submodule_gone(self):
        with pytest.raises(ModuleNotFoundError):
            import maldiamrkit.evaluation.label_encoder  # noqa: F401
