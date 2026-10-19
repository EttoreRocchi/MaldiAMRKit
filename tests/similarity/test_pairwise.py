"""Tests for pairwise distance matrix computation."""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

import maldiamrkit.similarity.pairwise as pairwise_module
from maldiamrkit.similarity import hierarchical_clustering, spectral_distance
from maldiamrkit.similarity.pairwise import pairwise_distances
from maldiamrkit.spectrum import MaldiSpectrum

_VECTORIZED = ["cosine", "pearson", "spectral_contrast_angle"]
_D = 6000


def _per_pair(rows, metric: str) -> np.ndarray:
    """Reference matrix from an explicit ``spectral_distance`` loop."""
    n = len(rows)
    D = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            D[i, j] = D[j, i] = spectral_distance(rows[i], rows[j], metric=metric)
    return D


def _spectra_on_grid(X: np.ndarray, offset: float = 0.0) -> list[MaldiSpectrum]:
    mz = np.linspace(2000.0, 20000.0, X.shape[1]) + offset
    return [MaldiSpectrum(pd.DataFrame({"mass": mz, "intensity": row})) for row in X]


@pytest.fixture(scope="module")
def edge_matrix() -> np.ndarray:
    """Non-negative rows covering the numerical edge cases of the built-ins.

    Random rows, a zero row, a near-zero (norm ~1e-14) row, duplicated flat
    rows at three levels, and a duplicated random row.
    """
    rng = np.random.default_rng(7)
    rows = [rng.exponential(1.0, _D) for _ in range(8)]
    rows += [
        np.zeros(_D),
        1e-15 * rng.random(_D),
        np.full(_D, 0.1),
        np.full(_D, 0.1),
        np.full(_D, 1 / 6000),
        np.full(_D, 1 / 6000),
        np.full(_D, 7.0),
        np.full(_D, 7.0),
        rows[0].copy(),
    ]
    return np.vstack(rows)


@pytest.fixture
def no_per_pair(monkeypatch):
    """Make the per-pair path fail, to prove the vectorized path was taken."""

    def _fail(*args, **kwargs):
        raise AssertionError("per-pair path was used")

    monkeypatch.setattr(pairwise_module, "_pairwise_general", _fail)


@pytest.fixture
def per_pair_calls(monkeypatch) -> list:
    """Record calls to the per-pair path while still running it."""
    calls: list = []
    real = pairwise_module._pairwise_general

    def _spy(*args, **kwargs):
        calls.append(args)
        return real(*args, **kwargs)

    monkeypatch.setattr(pairwise_module, "_pairwise_general", _spy)
    return calls


class TestPairwiseDistances:
    """Shape, symmetry, diagonal, and parallelism checks."""

    def test_shape(self, small_binned_df):
        D = pairwise_distances(small_binned_df, metric="cosine")
        n = len(small_binned_df)
        assert D.shape == (n, n)

    def test_symmetry(self, small_binned_df):
        D = pairwise_distances(small_binned_df, metric="cosine")
        np.testing.assert_allclose(D, D.T, atol=1e-12)

    def test_zero_diagonal(self, small_binned_df):
        D = pairwise_distances(small_binned_df, metric="cosine")
        np.testing.assert_allclose(np.diag(D), 0.0, atol=1e-12)

    def test_njobs_equivalence(self, small_binned_df):
        D1 = pairwise_distances(small_binned_df, metric="cosine", n_jobs=1)
        D2 = pairwise_distances(small_binned_df, metric="cosine", n_jobs=2)
        np.testing.assert_allclose(D1, D2, atol=1e-12)

    def test_fast_path_matches_general(self, small_binned_df):
        """Vectorized path matches an explicit per-pair loop."""
        D_fast = pairwise_distances(small_binned_df, metric="cosine")
        D_ref = _per_pair(list(small_binned_df.to_numpy()), "cosine")
        np.testing.assert_allclose(D_fast, D_ref, atol=1e-12)

    def test_invalid_metric(self, small_binned_df):
        with pytest.raises(ValueError, match="is not a valid"):
            pairwise_distances(small_binned_df, metric="bad")

    def test_pearson_metric(self, small_binned_df):
        D = pairwise_distances(small_binned_df, metric="pearson")
        assert D.shape == (len(small_binned_df), len(small_binned_df))
        np.testing.assert_allclose(np.diag(D), 0.0, atol=1e-12)


class TestVectorizedBuiltins:
    """The whole-matrix built-ins agree with their per-pair definitions."""

    @pytest.mark.parametrize("metric", _VECTORIZED)
    def test_matches_per_pair(self, edge_matrix, metric):
        rows = list(edge_matrix)
        D = pairwise_distances(pd.DataFrame(edge_matrix), metric=metric)
        R = _per_pair(rows, metric)
        if metric == "spectral_contrast_angle":
            # arccos is ill-conditioned next to 1: near-identical pairs only
            # agree to ~1e-8 between the two forms.
            near = _per_pair(rows, "cosine") < 1e-10
            np.testing.assert_allclose(D[near], R[near], atol=1e-6)
            np.testing.assert_allclose(D[~near], R[~near], atol=1e-12)
        else:
            np.testing.assert_allclose(D, R, atol=1e-12)

    @pytest.mark.parametrize("dtype", [np.float64, np.float32])
    @pytest.mark.parametrize("metric", _VECTORIZED)
    def test_matrix_contract(self, edge_matrix, metric, dtype):
        D = pairwise_distances(pd.DataFrame(edge_matrix.astype(dtype)), metric=metric)
        upper = 1.0 if metric == "spectral_contrast_angle" else 2.0
        assert D.dtype == np.float64
        assert np.array_equal(D, D.T)
        assert np.all(np.diag(D) == 0.0)
        assert D.min() >= 0.0
        assert D.max() <= upper
        hierarchical_clustering(D)

    @pytest.mark.parametrize("bad", [np.nan, np.inf])
    @pytest.mark.parametrize("metric", _VECTORIZED)
    def test_non_finite_input_raises(self, metric, bad):
        X = np.random.default_rng(0).random((4, 10))
        X[1, 3] = bad
        with pytest.raises(ValueError, match="finite"):
            pairwise_distances(pd.DataFrame(X), metric=metric)
        with pytest.raises(ValueError, match="finite"):
            pairwise_distances(list(X), metric=metric)


class TestRouting:
    """Which inputs take the vectorized path and which stay per-pair."""

    @pytest.mark.parametrize("metric", _VECTORIZED)
    def test_stackable_inputs_are_vectorized(self, edge_matrix, metric, no_per_pair):
        X = edge_matrix[:6]
        inputs = {
            "DataFrame": pd.DataFrame(X),
            "ndarray": X,
            "list[MaldiSpectrum]": _spectra_on_grid(X),
        }
        reference = None
        for spectra in inputs.values():
            per_n_jobs = [
                pairwise_distances(spectra, metric=metric, n_jobs=n_jobs)
                for n_jobs in (1, 2, -1)
            ]
            for D in per_n_jobs[1:]:
                assert np.array_equal(D, per_n_jobs[0])
            if reference is None:
                reference = per_n_jobs[0]
            np.testing.assert_allclose(per_n_jobs[0], reference, atol=1e-12)

    def test_different_grids_stay_per_pair(self, edge_matrix, per_pair_calls):
        X = edge_matrix[:4]
        spectra = [
            _spectra_on_grid(row[None, :], offset=float(k))[0]
            for k, row in enumerate(X)
        ]
        D = pairwise_distances(spectra, metric="cosine")
        assert per_pair_calls
        np.testing.assert_allclose(D, _per_pair(list(X), "cosine"), atol=1e-12)

    def test_mixed_inputs_stay_per_pair(self, edge_matrix, per_pair_calls):
        X = edge_matrix[:3]
        spectra = [X[0], *_spectra_on_grid(X[1:])]
        D = pairwise_distances(spectra, metric="cosine")
        assert per_pair_calls
        np.testing.assert_allclose(D, _per_pair(list(X), "cosine"), atol=1e-12)

    def test_empty_vectors_stay_per_pair(self, per_pair_calls):
        D = pairwise_distances([np.array([]), np.array([]), np.array([])], "cosine")
        assert per_pair_calls
        assert D.shape == (3, 3)

    def test_unequal_lengths_still_raise(self):
        with pytest.raises(ValueError):
            pairwise_distances([np.ones(5), np.ones(6), np.ones(5)], metric="cosine")

    @pytest.mark.parametrize("spectra", [[], pd.DataFrame()])
    def test_empty_input(self, spectra):
        assert pairwise_distances(spectra, metric="cosine").shape == (0, 0)

    @pytest.mark.parametrize("spectra", [[np.ones(3)], pd.DataFrame([[1.0, 2.0, 3.0]])])
    def test_single_spectrum(self, spectra):
        np.testing.assert_array_equal(
            pairwise_distances(spectra, metric="cosine"), [[0.0]]
        )

    def test_unknown_metric_checked_before_size(self):
        with pytest.raises(ValueError, match="is not a valid"):
            pairwise_distances([], metric="bad")


class TestFlatPearson:
    """A flat spectrum has no defined correlation."""

    @pytest.mark.parametrize("value", [0.1, 1 / 6000, 7.0])
    def test_identical_flat_spectra(self, value):
        a = np.full(_D, value)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            assert spectral_distance(a, a.copy(), metric="pearson") == 0.0
            D = pairwise_distances(pd.DataFrame([a, a.copy()]), metric="pearson")
        assert D[0, 1] == 0.0

    def test_flat_against_flat_and_non_flat(self):
        rng = np.random.default_rng(3)
        rows = [
            np.full(_D, 0.1),
            np.full(_D, 1 / 6000),
            np.full(_D, 7.0),
            rng.exponential(1.0, _D),
        ]
        expected = np.array(
            [
                [0.0, 0.0, 0.0, 1.0],
                [0.0, 0.0, 0.0, 1.0],
                [0.0, 0.0, 0.0, 1.0],
                [1.0, 1.0, 1.0, 0.0],
            ]
        )
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            D = pairwise_distances(pd.DataFrame(rows), metric="pearson")
            R = _per_pair(rows, "pearson")
        np.testing.assert_array_equal(D, expected)
        np.testing.assert_array_equal(R, expected)
