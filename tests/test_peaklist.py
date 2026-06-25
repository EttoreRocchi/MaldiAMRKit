"""Tests for peak sets: PeakSet / PeakList, transform_peaklist,
create_peakset_input, and the stateless peak cache."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from maldiamrkit.detection import (
    MaldiPeakDetector,
    PeakList,
    PeakSet,
    create_peakset_input,
)


class TestPeakSet:
    def test_sorts_by_mz(self):
        ps = PeakSet([7500, 3000, 5000], [0.3, 0.1, 0.2])
        np.testing.assert_array_equal(ps.mz, [3000, 5000, 7500])
        np.testing.assert_array_equal(ps.intensity, [0.1, 0.2, 0.3])
        assert ps.n_peaks == 3
        assert len(ps) == 3

    def test_mismatched_lengths_raise(self):
        with pytest.raises(ValueError, match="same shape"):
            PeakSet([1, 2, 3], [1, 2])

    def test_top_k_keeps_most_intense(self):
        ps = PeakSet([3000, 5000, 7500, 10000], [0.1, 0.4, 0.2, 0.3])
        top = ps.top_k(2)
        assert top.n_peaks == 2
        # 0.4 @ 5000 and 0.3 @ 10000, returned m/z-sorted
        np.testing.assert_array_equal(top.mz, [5000, 10000])
        np.testing.assert_array_equal(top.intensity, [0.4, 0.3])

    def test_top_k_more_than_available(self):
        ps = PeakSet([3000, 5000], [0.1, 0.2])
        assert ps.top_k(10).n_peaks == 2

    def test_top_k_uses_score_when_present(self):
        # 5000 is the most intense, but the score ranks 3000 highest
        ps = PeakSet([3000, 5000], [0.1, 0.2], score=[9.0, 1.0])
        top = ps.top_k(1)
        np.testing.assert_array_equal(top.mz, [3000])
        np.testing.assert_array_equal(top.score, [9.0])

    def test_top_k_intensity_fallback_without_score(self):
        ps = PeakSet([3000, 5000], [0.1, 0.2])
        np.testing.assert_array_equal(ps.top_k(1).mz, [5000])

    def test_score_follows_mz_sort(self):
        ps = PeakSet([7500, 3000, 5000], [0.3, 0.1, 0.2], score=[0.9, 0.7, 0.8])
        np.testing.assert_array_equal(ps.mz, [3000, 5000, 7500])
        np.testing.assert_array_equal(ps.score, [0.7, 0.8, 0.9])

    def test_score_length_mismatch_raises(self):
        with pytest.raises(ValueError, match="score must have the same shape"):
            PeakSet([3000, 5000], [0.1, 0.2], score=[1.0])

    def test_repr(self):
        assert "n_peaks=2" in repr(PeakSet([3000, 5000], [0.1, 0.2]))
        assert repr(PeakSet([], [])) == "PeakSet(n_peaks=0)"

    def test_as_array(self):
        ps = PeakSet([3000, 5000], [0.1, 0.2])
        arr = ps.as_array()
        assert arr.shape == (2, 2)
        np.testing.assert_array_equal(arr[:, 0], [3000, 5000])

    def test_empty(self):
        ps = PeakSet([], [])
        assert ps.n_peaks == 0


class TestPeakList:
    def _make(self):
        return PeakList(
            [
                PeakSet([3000, 5000], [0.1, 0.2]),
                PeakSet([4000, 6000, 8000], [0.3, 0.4, 0.5]),
            ],
            index=["a", "b"],
        )

    def test_basic(self):
        pl = self._make()
        assert len(pl) == 2
        np.testing.assert_array_equal(pl.n_peaks, [2, 3])
        assert pl[1].n_peaks == 3
        assert pl.meta["warped"] is False

    def test_index_length_mismatch_raises(self):
        with pytest.raises(ValueError, match="index has"):
            PeakList([PeakSet([1], [1])], index=["a", "b"])

    def test_to_padded(self):
        pl = self._make()
        values, mask = pl.to_padded()
        assert values.shape == (2, 3, 2)
        assert mask.shape == (2, 3)
        # first spectrum has 2 peaks -> last column padded
        assert mask[0].tolist() == [True, True, False]
        assert mask[1].tolist() == [True, True, True]
        np.testing.assert_array_equal(values[0, :2, 0], [3000, 5000])
        np.testing.assert_array_equal(values[0, 2], [0.0, 0.0])

    def test_to_padded_truncates_to_max_peaks(self):
        pl = self._make()
        values, mask = pl.to_padded(max_peaks=2)
        assert values.shape == (2, 2, 2)
        # second spectrum truncated to its 2 most intense (0.4@6000, 0.5@8000)
        assert mask[1].tolist() == [True, True]
        np.testing.assert_array_equal(values[1, :, 0], [6000, 8000])

    def test_save_load_round_trip(self, tmp_path):
        pl = self._make()
        pl.meta["method"] = "local"
        pl.save(tmp_path / "pl")
        loaded = PeakList.load(tmp_path / "pl")
        assert len(loaded) == 2
        for orig, new in zip(pl.peaks, loaded.peaks, strict=True):
            np.testing.assert_array_equal(orig.mz, new.mz)
            np.testing.assert_array_equal(orig.intensity, new.intensity)
        assert loaded.meta["method"] == "local"
        assert loaded.meta["warped"] is False
        assert loaded.index.tolist() == ["a", "b"]

    def test_save_load_all_empty(self, tmp_path):
        pl = PeakList([PeakSet([], []), PeakSet([], [])])
        pl.save(tmp_path / "empty")
        loaded = PeakList.load(tmp_path / "empty")
        assert len(loaded) == 2
        assert loaded[0].n_peaks == 0

    def test_getitem_slice_returns_peaklist(self):
        pl = PeakList(
            [PeakSet([1], [1]), PeakSet([2], [2]), PeakSet([3], [3])],
            index=["a", "b", "c"],
        )
        sub = pl[1:]
        assert isinstance(sub, PeakList)
        assert len(sub) == 2
        assert sub.index.tolist() == ["b", "c"]
        assert sub.meta["warped"] is False
        # int access still returns a PeakSet
        assert isinstance(pl[0], PeakSet)

    def test_repr(self):
        pl = PeakList([PeakSet([1], [1])], meta={"method": "local", "top_k": 5})
        r = repr(pl)
        assert "n=1" in r and "method=local" in r and "top_k=5" in r

    def test_to_padded_truncates_by_score(self):
        # intensity order differs from score order; truncation must follow score
        pl = PeakList(
            [PeakSet([3000, 5000, 7500], [0.5, 0.4, 0.3], score=[0.1, 0.2, 0.9])]
        )
        values, mask = pl.to_padded(max_peaks=2)
        # top-2 by score are 7500 (0.9) and 5000 (0.2), returned m/z-sorted
        np.testing.assert_array_equal(values[0, :, 0], [5000, 7500])
        assert mask[0].tolist() == [True, True]

    def test_save_load_round_trips_score(self, tmp_path):
        pl = PeakList([PeakSet([3000, 5000], [0.1, 0.2], score=[0.7, 0.9])])
        pl.save(tmp_path / "scored")
        loaded = PeakList.load(tmp_path / "scored")
        np.testing.assert_array_equal(loaded[0].score, [0.7, 0.9])

    def test_save_load_preserves_integer_index(self, tmp_path):
        # regression: index used to round-trip to strings, so a cache hit
        # returned a different index dtype than a cache miss.
        pl = PeakList([PeakSet([1], [1]), PeakSet([2], [2])], index=[10, 20])
        pl.save(tmp_path / "ints")
        loaded = PeakList.load(tmp_path / "ints")
        assert loaded.index.tolist() == [10, 20]
        assert loaded.index.dtype == pl.index.dtype


class TestTransformPeaklist:
    def test_basic_shapes(self, binned_dataset: pd.DataFrame):
        det = MaldiPeakDetector(method="local", prominence=1e-4)
        pl = det.transform_peaklist(binned_dataset, top_k=10)
        assert len(pl) == len(binned_dataset)
        assert list(pl.index) == list(binned_dataset.index)
        assert pl.meta["warped"] is False
        assert pl.meta["top_k"] == 10
        for ps in pl:
            assert ps.n_peaks <= 10
            if ps.n_peaks:
                assert ps.mz.min() >= 2000
                assert ps.mz.max() <= 20000

    def test_mz_recovered_from_columns(self, binned_dataset: pd.DataFrame):
        det = MaldiPeakDetector(method="local", prominence=1e-4)
        pl = det.transform_peaklist(binned_dataset, top_k=5)
        # m/z must be drawn from the column labels (2000..20000)
        all_mz = np.concatenate([ps.mz for ps in pl if ps.n_peaks])
        cols = binned_dataset.columns.to_numpy(dtype=float)
        assert np.isin(all_mz, cols).all()

    def test_explicit_mz_for_ndarray(self):
        rng = np.random.default_rng(0)
        X = rng.random((3, 50))
        mz = np.linspace(2000, 20000, 50)
        det = MaldiPeakDetector(method="local")
        pl = det.transform_peaklist(X, top_k=5, mz=mz)
        assert len(pl) == 3

    def test_ndarray_without_mz_raises(self):
        det = MaldiPeakDetector(method="local")
        with pytest.raises(ValueError, match="mz="):
            det.transform_peaklist(np.zeros((2, 10)), top_k=5)

    def test_non_numeric_columns_raise(self):
        df = pd.DataFrame(np.zeros((2, 3)), columns=["a", "b", "c"])
        det = MaldiPeakDetector(method="local")
        with pytest.raises(ValueError, match="m/z axis"):
            det.transform_peaklist(df, top_k=2)

    def test_rank_by_prominence(self, binned_dataset: pd.DataFrame):
        det = MaldiPeakDetector(method="local", prominence=1e-4)
        pl = det.transform_peaklist(binned_dataset, top_k=10, rank_by="prominence")
        assert pl.meta["rank_by"] == "prominence"

    def test_persistence_requires_ph(self, binned_dataset: pd.DataFrame):
        det = MaldiPeakDetector(method="local", prominence=1e-4)
        with pytest.raises(ValueError, match="persistence"):
            det.transform_peaklist(binned_dataset.iloc[:1], rank_by="persistence")

    def test_unknown_rank_by_raises(self, binned_dataset: pd.DataFrame):
        det = MaldiPeakDetector(method="local")
        with pytest.raises(ValueError, match="Unknown rank_by"):
            det.transform_peaklist(binned_dataset.iloc[:1], rank_by="bogus")

    def test_series_input(self, binned_dataset: pd.DataFrame):
        det = MaldiPeakDetector(method="local", prominence=1e-4)
        pl = det.transform_peaklist(binned_dataset.iloc[0], top_k=5)
        assert len(pl) == 1

    def test_meta_method_is_clean_value(self, binned_dataset: pd.DataFrame):
        # regression: meta["method"] used to be "PeakMethod.local"
        det = MaldiPeakDetector(method="local", prominence=1e-4)
        pl = det.transform_peaklist(binned_dataset.iloc[:1], top_k=5)
        assert pl.meta["method"] == "local"

    def test_no_leakage_global_equals_subset(self, binned_dataset: pd.DataFrame):
        """Per-spectrum extraction: a global precompute is byte-identical to a
        per-subset (per-fold) one. This is the leak-safety guarantee."""
        det = MaldiPeakDetector(method="local", prominence=1e-4)
        full = det.transform_peaklist(binned_dataset, top_k=15)
        subset = det.transform_peaklist(binned_dataset.iloc[:5], top_k=15)
        for i in range(5):
            np.testing.assert_array_equal(full[i].mz, subset[i].mz)
            np.testing.assert_array_equal(full[i].intensity, subset[i].intensity)


class TestPeaklistCache:
    def test_cache_hit_and_invalidation(self, binned_dataset, tmp_path):
        det = MaldiPeakDetector(method="local", prominence=1e-4)
        X = binned_dataset.iloc[:5]

        pl1 = det.transform_peaklist(X, top_k=10, cache_dir=tmp_path)
        cached_files = list(Path(tmp_path).glob("peaklist_*.npz"))
        assert len(cached_files) == 1

        pl2 = det.transform_peaklist(X, top_k=10, cache_dir=tmp_path)
        for i in range(len(pl1)):
            np.testing.assert_array_equal(pl1[i].mz, pl2[i].mz)
            np.testing.assert_array_equal(pl1[i].intensity, pl2[i].intensity)

        # Different config (top_k) -> new cache entry, old one preserved.
        det.transform_peaklist(X, top_k=5, cache_dir=tmp_path)
        assert len(list(Path(tmp_path).glob("peaklist_*.npz"))) == 2

    def test_extra_find_peaks_kwargs_invalidate_cache(self, binned_dataset, tmp_path):
        """Regression: a find_peaks kwarg that is not one of the explicit params
        (here ``threshold``) lives only in ``self.kwargs`` and used to be invisible
        to the cache key, so a stale entry silently returned the wrong peaks."""
        X = binned_dataset.iloc[:3]
        det_a = MaldiPeakDetector(method="local", threshold=0.0)
        det_b = MaldiPeakDetector(method="local", threshold=0.5)

        pl_a = det_a.transform_peaklist(X, top_k=50, cache_dir=tmp_path)
        pl_b = det_b.transform_peaklist(X, top_k=50, cache_dir=tmp_path)
        pl_b_true = det_b.transform_peaklist(X, top_k=50)

        assert pl_a.meta["config_hash"] != pl_b.meta["config_hash"]
        for i in range(len(pl_b)):
            np.testing.assert_array_equal(pl_b[i].mz, pl_b_true[i].mz)


def _write_spectrum(path: Path, seed: int) -> None:
    rng = np.random.default_rng(seed)
    mz = np.linspace(2000, 20000, 2000)
    intensity = np.zeros_like(mz)
    for pos in (3000, 5000, 7500, 10000, 12500):
        intensity += 1000.0 * np.exp(-0.5 * ((mz - pos) / 10.0) ** 2)
    intensity += rng.normal(0, 5.0, mz.size)
    intensity = np.clip(intensity, 0, None)
    np.savetxt(path, np.column_stack([mz, intensity]))


class TestCreatePeaksetInput:
    def test_from_directory(self, tmp_path):
        for i in range(3):
            _write_spectrum(tmp_path / f"s{i}.txt", seed=i)
        pl = create_peakset_input(tmp_path, top_k=20)
        assert len(pl) == 3
        assert pl.meta["warped"] is False
        assert pl.meta["source"] == "create_peakset_input"
        for ps in pl:
            assert ps.n_peaks <= 20
            if ps.n_peaks:
                assert ps.mz.min() >= 1999
                assert ps.mz.max() <= 20001

    def test_cache_round_trip(self, tmp_path):
        spectra = tmp_path / "spectra"
        spectra.mkdir()
        for i in range(2):
            _write_spectrum(spectra / f"s{i}.txt", seed=i)
        cache = tmp_path / "cache"
        pl1 = create_peakset_input(spectra, top_k=15, cache_dir=cache)
        pl2 = create_peakset_input(spectra, top_k=15, cache_dir=cache)
        assert len(list(cache.glob("peaklist_*.npz"))) == 1
        for i in range(len(pl1)):
            np.testing.assert_array_equal(pl1[i].mz, pl2[i].mz)

    def test_unknown_rank_by_raises(self, tmp_path):
        _write_spectrum(tmp_path / "s0.txt", seed=0)
        with pytest.raises(ValueError, match="Unknown rank_by"):
            create_peakset_input(tmp_path, rank_by="bogus")
