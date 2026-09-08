"""Tests for the public spectral-metric registry."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from maldiamrkit.similarity import (
    extract_mz_intensity,
    list_spectral_metrics,
    pairwise_distances,
    register_spectral_metric,
    spectral_distance,
    unregister_spectral_metric,
)
from maldiamrkit.similarity.metrics import _DEFAULT_SPECTRAL_METRICS, _METRIC_REGISTRY
from tests.conftest import make_registry_guard

_BUILTINS = sorted(_DEFAULT_SPECTRAL_METRICS)

clean_registry = make_registry_guard(_METRIC_REGISTRY)


def manhattan(spec_a, spec_b) -> float:
    """Module-level custom metric (picklable for process-based joblib)."""
    _, a = extract_mz_intensity(spec_a)
    _, b = extract_mz_intensity(spec_b)
    return float(np.abs(np.asarray(a, dtype=float) - np.asarray(b, dtype=float)).sum())


class TestRegister:
    def test_registered_metric_dispatches(self, clean_registry):
        register_spectral_metric("manhattan", manhattan)
        d = spectral_distance(
            np.array([1.0, 2.0, 3.0]), np.array([1.0, 5.0, 3.0]), metric="manhattan"
        )
        assert d == pytest.approx(3.0)

    def test_appears_in_listing(self, clean_registry):
        assert "manhattan" not in list_spectral_metrics()
        register_spectral_metric("manhattan", manhattan)
        assert "manhattan" in list_spectral_metrics()

    def test_listing_is_sorted_and_includes_builtins(self):
        listed = list_spectral_metrics()
        assert listed == sorted(listed)
        assert set(_BUILTINS) <= set(listed)

    def test_rejects_non_callable(self):
        with pytest.raises(TypeError, match="must be callable"):
            register_spectral_metric("bad", "not-a-function")

    def test_custom_name_can_be_replaced_silently(self, clean_registry):
        register_spectral_metric("mine", manhattan)
        register_spectral_metric("mine", lambda a, b: 42.0)
        assert spectral_distance(np.array([1.0]), np.array([1.0]), "mine") == 42.0

    @pytest.mark.parametrize("name", _BUILTINS)
    def test_builtin_protected_without_override(self, name, clean_registry):
        with pytest.raises(ValueError, match="built-in spectral metric"):
            register_spectral_metric(name, manhattan)
        assert _METRIC_REGISTRY[name] is not manhattan

    @pytest.mark.parametrize("name", _BUILTINS)
    def test_builtin_replaceable_with_override(self, name, clean_registry):
        register_spectral_metric(name, manhattan, override=True)
        assert _METRIC_REGISTRY[name] is manhattan


class TestUnregister:
    def test_removes_custom_metric(self, clean_registry):
        register_spectral_metric("manhattan", manhattan)
        unregister_spectral_metric("manhattan")
        assert "manhattan" not in list_spectral_metrics()

    def test_unknown_name_raises_key_error(self, clean_registry):
        with pytest.raises(KeyError, match="No spectral metric named"):
            unregister_spectral_metric("never_registered")

    @pytest.mark.parametrize("name", _BUILTINS)
    def test_builtin_cannot_be_removed(self, name, clean_registry):
        with pytest.raises(ValueError, match="Cannot unregister built-in"):
            unregister_spectral_metric(name)
        assert name in _METRIC_REGISTRY

    def test_unregister_overridden_builtin_restores_default(self, clean_registry):
        default = _METRIC_REGISTRY["cosine"]
        register_spectral_metric("cosine", manhattan, override=True)
        assert _METRIC_REGISTRY["cosine"] is manhattan
        unregister_spectral_metric("cosine")
        assert _METRIC_REGISTRY["cosine"] is default
        a = np.array([1.0, 2.0, 3.0])
        assert spectral_distance(a, a, metric="cosine") == pytest.approx(0.0, abs=1e-9)


class TestDispatchErrors:
    def test_unknown_metric_lists_valid_names(self):
        a = np.array([1.0])
        with pytest.raises(ValueError, match="register_spectral_metric"):
            spectral_distance(a, a, metric="nope")

    def test_pairwise_rejects_unknown_metric(self):
        X = pd.DataFrame(np.eye(3))
        with pytest.raises(ValueError, match="is not a valid spectral metric"):
            pairwise_distances(X, metric="nope")


class TestPairwiseWithCustomMetric:
    """A registered metric must work through ``pairwise_distances``.

    Worker processes do not inherit the parent's registry, so the metric
    callable is resolved in the parent and shipped to the workers. Without
    that, ``n_jobs != 1`` would raise ``KeyError`` inside a worker.
    """

    @staticmethod
    def _reference(X: pd.DataFrame) -> np.ndarray:
        values = X.to_numpy()
        n = len(values)
        D = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                D[i, j] = np.abs(values[i] - values[j]).sum()
        return D

    @pytest.fixture
    def X(self):
        return pd.DataFrame(np.random.RandomState(0).rand(5, 8))

    @pytest.mark.parametrize("n_jobs", [1, 2, -1])
    def test_matches_reference_at_every_n_jobs(self, X, n_jobs, clean_registry):
        register_spectral_metric("manhattan", manhattan)
        D = pairwise_distances(X, metric="manhattan", n_jobs=n_jobs)
        np.testing.assert_allclose(D, self._reference(X), atol=1e-9)

    def test_matrix_is_symmetric_with_zero_diagonal(self, X, clean_registry):
        register_spectral_metric("manhattan", manhattan)
        D = pairwise_distances(X, metric="manhattan", n_jobs=2)
        np.testing.assert_allclose(D, D.T)
        np.testing.assert_allclose(np.diag(D), 0.0)

    def test_list_of_spectra_input(self, clean_registry):
        register_spectral_metric("manhattan", manhattan)
        rows = [np.array([1.0, 2.0]), np.array([1.0, 5.0]), np.array([0.0, 2.0])]
        D = pairwise_distances(rows, metric="manhattan", n_jobs=1)
        assert D.shape == (3, 3)
        assert D[0, 1] == pytest.approx(3.0)
        assert D[0, 2] == pytest.approx(1.0)


class TestBuiltinsUnaffected:
    """Registering custom metrics must not perturb the built-ins."""

    @pytest.mark.parametrize("metric", ["cosine", "pearson", "spectral_contrast_angle"])
    def test_builtin_still_dispatches(self, metric, clean_registry):
        register_spectral_metric("manhattan", manhattan)
        a = np.array([1.0, 2.0, 3.0])
        assert spectral_distance(a, a, metric=metric) == pytest.approx(0.0, abs=1e-9)

    def test_enum_member_still_accepted(self, clean_registry):
        from maldiamrkit.similarity import SpectralMetric

        register_spectral_metric("manhattan", manhattan)
        a = np.array([1.0, 2.0, 3.0])
        assert spectral_distance(a, a, metric=SpectralMetric.cosine) == pytest.approx(
            0.0, abs=1e-9
        )
