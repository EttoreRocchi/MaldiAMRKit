"""Tests for LabelEncoder location move from evaluation to susceptibility."""

from __future__ import annotations

import warnings

import numpy as np


class TestNewLocation:
    def test_import_from_susceptibility(self):
        from maldiamrkit.susceptibility import IntermediateHandling, LabelEncoder

        enc = LabelEncoder()
        out = enc.fit_transform(["R", "S", "R"])
        np.testing.assert_array_equal(out, [1, 0, 1])
        assert IntermediateHandling.susceptible.value == "susceptible"


class TestDeprecatedReexport:
    def test_lazy_attribute_emits_deprecation(self):
        import maldiamrkit.evaluation as ev

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            cls = ev.LabelEncoder
        deps = [w for w in caught if issubclass(w.category, DeprecationWarning)]
        assert deps, "expected a DeprecationWarning when touching ev.LabelEncoder"
        assert cls is not None
        assert cls(intermediate="resistant").fit_transform(["R"]).tolist() == [1]

    def test_submodule_import_emits_deprecation(self):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            from maldiamrkit.evaluation.label_encoder import LabelEncoder as Le
        deps = [w for w in caught if issubclass(w.category, DeprecationWarning)]
        assert deps, "expected a DeprecationWarning on submodule import"
        assert Le(intermediate="susceptible").fit_transform(["S"]).tolist() == [0]

    def test_unrelated_evaluation_imports_silent(self):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            from maldiamrkit.evaluation import (  # noqa: F401
                amr_classification_report,
                very_major_error_rate,
            )
        deps = [
            w
            for w in caught
            if issubclass(w.category, DeprecationWarning)
            and "LabelEncoder" in str(w.message)
        ]
        assert not deps, (
            "unrelated evaluation imports should not trigger a "
            "LabelEncoder deprecation warning"
        )
