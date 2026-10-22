"""Unit tests for Warping class."""

import os
import re
import warnings
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import pytest
from scipy.signal import find_peaks

matplotlib.use("Agg")  # Use non-interactive backend for tests
import matplotlib.pyplot as plt

import maldiamrkit.alignment.strategies as strategies_module
import maldiamrkit.alignment.warping as warping_module
from maldiamrkit import MaldiSpectrum
from maldiamrkit.alignment import Warping
from maldiamrkit.alignment.strategies import (
    ShiftStrategy,
    _conservative_resample,
    _shift_row,
)
from tests.conftest import SPECTRA_DIR

NON_DTW_METHODS = ["shift", "linear", "piecewise", "quadratic", "cubic", "lowess"]
NON_DTW_METHODS_LOWESS_SLOW = [
    pytest.param(m, marks=pytest.mark.slow) if m == "lowess" else m
    for m in NON_DTW_METHODS
]


def _rowwise_transform(warper: Warping, X: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Row-by-row transform (reference oracle for the chunked implementation).

    Applies the fitted shift step of ``prealign`` (if any), detects peaks
    on the whole of ``X`` (keeping the ``n_peaks`` most prominent), aligns
    one row at a time and stacks the rows in the input dtype. Returns the
    aligned frame and the number of non-monotonic rows.
    """
    lags = np.zeros(len(X), dtype=int)
    if warper.prealign_ref_ is not None:
        prealign_peaks = warper._prealign_reference_peaks(X.columns)
        shifted, lags = warper._prealign_rows(X, prealign_peaks)
        X = pd.DataFrame(shifted, index=X.index, columns=X.columns)
    mz_axis = np.arange(X.shape[1])
    ref_peaks_df = warper.peak_detector.transform(
        pd.DataFrame(warper.ref_spec_[np.newaxis, :], columns=X.columns)
    )
    ref_peaks = warper._select_peaks(
        warper.ref_spec_, ref_peaks_df.iloc[0].to_numpy().nonzero()[0]
    )
    peaks_df = None if warper.method == "dtw" else warper.peak_detector.transform(X)
    rows, n_non_monotonic = [], 0
    for i in range(len(X)):
        row = X.iloc[i].to_numpy()
        peaks = None
        if peaks_df is not None:
            peaks = warper._select_peaks(row, peaks_df.iloc[i].to_numpy().nonzero()[0])
        row, non_monotonic = warper._align_single_row(
            row, peaks, ref_peaks, mz_axis, int(lags[i])
        )
        rows.append(row)
        n_non_monotonic += non_monotonic
    stacked = np.vstack(rows).astype(X.to_numpy().dtype)
    return pd.DataFrame(stacked, index=X.index, columns=X.columns), n_non_monotonic


def _transform_with_count(warper: Warping, X: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Run ``warper.transform`` and parse the aggregated non-monotonic count."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = warper.transform(X)
    matches = [
        re.search(r"mappings for (\d+)/(\d+) sample", str(w.message)) for w in caught
    ]
    counts = [m for m in matches if m is not None]
    assert len(counts) <= 1
    if not counts:
        return result, 0
    assert int(counts[0].group(2)) == len(X)
    return result, int(counts[0].group(1))


@pytest.fixture(scope="module")
def real_binned_dataset() -> pd.DataFrame:
    """Binned (3 Da) matrix of the real spectra shipped in ``data/``."""
    paths = sorted(SPECTRA_DIR.glob("*.txt"))
    if not paths:
        pytest.skip("No spectrum files found in data directory")
    rows = [MaldiSpectrum(p).bin(3).binned["intensity"].to_numpy() for p in paths]
    return pd.DataFrame(
        np.vstack(rows),
        index=[p.stem for p in paths],
        columns=[str(2000 + 3 * i) for i in range(len(rows[0]))],
    )


class TestWarpingInit:
    """Tests for Warping initialization."""

    def test_default_init(self):
        """Test default initialization."""
        warper = Warping()

        assert warper.method == "shift"
        assert warper.reference == "median"
        assert warper.n_jobs == 1

    def test_custom_init(self):
        """Test custom initialization."""
        warper = Warping(
            method="dtw",
            reference=0,
            n_jobs=4,
            dtw_radius=20,
        )

        assert warper.method == "dtw"
        assert warper.reference == 0
        assert warper.n_jobs == 4
        assert warper.dtw_radius == 20


class TestWarpingFit:
    """Tests for Warping fit."""

    def test_fit_with_median(self, binned_dataset: pd.DataFrame):
        """Test fitting with median reference."""
        warper = Warping(reference="median")
        warper.fit(binned_dataset)

        assert hasattr(warper, "ref_spec_")
        assert len(warper.ref_spec_) == binned_dataset.shape[1]

    def test_fit_with_index(self, binned_dataset: pd.DataFrame):
        """Test fitting with index reference."""
        warper = Warping(reference=5)
        warper.fit(binned_dataset)

        assert hasattr(warper, "ref_spec_")
        np.testing.assert_array_equal(
            warper.ref_spec_, binned_dataset.iloc[5].to_numpy()
        )

    def test_fit_invalid_index_raises(self, binned_dataset: pd.DataFrame):
        """Test that invalid reference index raises ValueError."""
        warper = Warping(reference=1000)

        with pytest.raises(ValueError, match="out of bounds"):
            warper.fit(binned_dataset)

    def test_fit_invalid_method_raises(self, binned_dataset: pd.DataFrame):
        """Test that invalid method raises ValueError."""
        with pytest.raises(ValueError, match="is not a valid"):
            Warping(method="invalid")


class TestWarpingTransform:
    """Tests for Warping transform."""

    @pytest.mark.parametrize(
        "method", ["shift", "linear", "piecewise", "quadratic", "cubic", "lowess"]
    )
    @pytest.mark.filterwarnings("ignore:Warping produced non-monotonic")
    def test_all_methods_run(self, binned_dataset: pd.DataFrame, method: str):
        """Test that all methods run without error."""
        warper = Warping(method=method)
        warper.fit(binned_dataset)
        result = warper.transform(binned_dataset)

        assert result.shape == binned_dataset.shape
        assert result.index.equals(binned_dataset.index)

    @pytest.mark.slow
    def test_dtw_method_run(self, binned_dataset: pd.DataFrame):
        """Test that DTW method runs without error."""
        small_dataset = binned_dataset.iloc[:5]
        warper = Warping(method="dtw")
        warper.fit(small_dataset)
        result = warper.transform(small_dataset)

        assert result.shape == small_dataset.shape
        assert result.index.equals(small_dataset.index)

    def test_transform_before_fit_raises(self, binned_dataset: pd.DataFrame):
        """Test that transform before fit raises RuntimeError."""
        warper = Warping()

        with pytest.raises(RuntimeError, match="must be fitted"):
            warper.transform(binned_dataset)

    def test_transform_preserves_shape(self, binned_dataset: pd.DataFrame):
        """Test that transform preserves shape."""
        warper = Warping(method="shift")
        warper.fit(binned_dataset)
        result = warper.transform(binned_dataset)

        assert result.shape == binned_dataset.shape
        assert result.columns.equals(binned_dataset.columns)


class TestWarpingParallelization:
    """Tests for parallelization."""

    @pytest.mark.parametrize(
        "method", ["shift", "linear", "piecewise", "quadratic", "cubic", "lowess"]
    )
    @pytest.mark.filterwarnings("ignore:Warping produced non-monotonic")
    def test_parallel_produces_same_results(
        self, binned_dataset: pd.DataFrame, method: str
    ):
        """Test that parallel processing produces same results as sequential."""
        warper_seq = Warping(method=method, n_jobs=1)
        warper_par = Warping(method=method, n_jobs=2)

        warper_seq.fit(binned_dataset)
        warper_par.fit(binned_dataset)

        result_seq = warper_seq.transform(binned_dataset)
        result_par = warper_par.transform(binned_dataset)

        pd.testing.assert_frame_equal(result_seq, result_par)

    @pytest.mark.slow
    def test_parallel_dtw(self, binned_dataset: pd.DataFrame):
        """Test parallel processing with DTW method."""
        # Use smaller dataset for faster test
        small_dataset = binned_dataset.iloc[:5]

        warper_seq = Warping(method="dtw", n_jobs=1)
        warper_par = Warping(method="dtw", n_jobs=2)

        warper_seq.fit(small_dataset)
        warper_par.fit(small_dataset)

        result_seq = warper_seq.transform(small_dataset)
        result_par = warper_par.transform(small_dataset)

        pd.testing.assert_frame_equal(result_seq, result_par)


class TestWarpingChunkedTransform:
    """Chunked transform matches the former row-by-row implementation."""

    @staticmethod
    def _assert_matches_rowwise(warper: Warping, X: pd.DataFrame) -> int:
        expected, expected_count = _rowwise_transform(warper, X)
        result, count = _transform_with_count(warper, X)
        pd.testing.assert_frame_equal(result, expected, check_exact=True)
        assert count == expected_count
        return count

    @pytest.mark.parametrize("method", NON_DTW_METHODS_LOWESS_SLOW)
    @pytest.mark.parametrize("dtype", [np.float64, np.float32])
    @pytest.mark.parametrize("n_jobs", [1, 2])
    @pytest.mark.parametrize("max_chunk_rows", [256, 7])
    def test_matches_rowwise_synthetic(
        self, binned_dataset, method, dtype, n_jobs, max_chunk_rows, monkeypatch
    ):
        monkeypatch.setattr(warping_module, "_MAX_CHUNK_ROWS", max_chunk_rows)
        X = binned_dataset.astype(dtype)
        warper = Warping(method=method, n_jobs=n_jobs).fit(X)
        self._assert_matches_rowwise(warper, X)

    @pytest.mark.parametrize("method", NON_DTW_METHODS_LOWESS_SLOW)
    @pytest.mark.parametrize("dtype", [np.float64, np.float32])
    def test_matches_rowwise_real_spectra(
        self, real_binned_dataset, method, dtype, monkeypatch
    ):
        monkeypatch.setattr(warping_module, "_MAX_CHUNK_ROWS", 5)
        X = real_binned_dataset.astype(dtype)
        warper = Warping(method=method, n_jobs=2).fit(X)
        self._assert_matches_rowwise(warper, X)

    @pytest.mark.slow
    @pytest.mark.parametrize("n_jobs", [1, 2])
    def test_matches_rowwise_dtw(self, binned_dataset, n_jobs, monkeypatch):
        monkeypatch.setattr(warping_module, "_MAX_CHUNK_ROWS", 2)
        X = binned_dataset.iloc[:5]
        warper = Warping(method="dtw", n_jobs=n_jobs).fit(X)
        self._assert_matches_rowwise(warper, X)

    @pytest.mark.parametrize("method", ["shift", "linear"])
    @pytest.mark.parametrize("n_jobs", [1, 2])
    def test_single_row(self, binned_dataset, method, n_jobs):
        warper = Warping(method=method, n_jobs=n_jobs).fit(binned_dataset)
        self._assert_matches_rowwise(warper, binned_dataset.iloc[[3]])

    def test_preserves_index_and_columns(self, binned_dataset, monkeypatch):
        monkeypatch.setattr(warping_module, "_MAX_CHUNK_ROWS", 7)
        rng = np.random.default_rng(0)
        X = binned_dataset.iloc[rng.permutation(len(binned_dataset))]
        X.index = pd.Index([f"id_{i}" for i in rng.permutation(len(X))], name="sid")
        warper = Warping(method="piecewise", n_jobs=2).fit(X)
        result = warper.transform(X)
        assert result.index.equals(X.index)
        assert result.index.name == "sid"
        assert result.columns.equals(X.columns)
        self._assert_matches_rowwise(warper, X)

    @pytest.mark.parametrize("n_jobs", [1, 2])
    def test_memmap_backed_input(self, binned_dataset, n_jobs, tmp_path, monkeypatch):
        monkeypatch.setattr(warping_module, "_MAX_CHUNK_ROWS", 7)
        path = tmp_path / "spectra.npy"
        np.save(path, binned_dataset.to_numpy(dtype=np.float32))
        X = pd.DataFrame(
            np.load(path, mmap_mode="r"),
            index=binned_dataset.index,
            columns=binned_dataset.columns,
        )
        warper = Warping(method="shift", n_jobs=n_jobs).fit(X)
        self._assert_matches_rowwise(warper, X)

    def test_non_monotonic_count_aggregated_across_chunks(
        self, binned_dataset, monkeypatch
    ):
        monkeypatch.setattr(warping_module, "_MAX_CHUNK_ROWS", 4)
        monkeypatch.setattr(
            strategies_module.LinearStrategy,
            "_positions",
            lambda self, sample, ref, axis, raw: axis[::-1].copy(),
        )
        X = binned_dataset.iloc[:15]
        warper = Warping(method="linear", n_jobs=1, prealign=False).fit(X)
        assert self._assert_matches_rowwise(warper, X) > 0
        shifted = Warping(method="shift", n_jobs=1, prealign=False).fit(X).transform(X)
        pd.testing.assert_frame_equal(_transform_with_count(warper, X)[0], shifted)

    @pytest.mark.parametrize("dtype", [np.float32, np.float64])
    @pytest.mark.parametrize("method", ["shift", "linear", "piecewise", "lowess"])
    def test_output_keeps_input_dtype(self, binned_dataset, method, dtype, monkeypatch):
        monkeypatch.setattr(warping_module, "_MAX_CHUNK_ROWS", 7)
        X = binned_dataset.astype(dtype)
        X.iloc[:10] = 0.0
        warper = Warping(method=method, n_jobs=1).fit(X)
        self._assert_matches_rowwise(warper, X)
        assert (warper.transform(X).dtypes == dtype).all()


class TestWarpingQuality:
    """Tests for alignment quality."""

    def test_get_alignment_quality(self, binned_dataset_with_shift: pd.DataFrame):
        """Test get_alignment_quality method."""
        warper = Warping(method="shift")
        warper.fit(binned_dataset_with_shift)

        X_aligned = warper.transform(binned_dataset_with_shift)
        quality = warper.get_alignment_quality(binned_dataset_with_shift, X_aligned)

        assert "correlation_before" in quality.columns
        assert "correlation_after" in quality.columns
        assert "improvement" in quality.columns
        assert "rmse_before" in quality.columns
        assert "rmse_after" in quality.columns
        assert len(quality) == len(binned_dataset_with_shift)

    def test_get_alignment_quality_linear(self, binned_dataset: pd.DataFrame):
        """Test quality with linear method."""
        warper = Warping(method="linear")
        warper.fit(binned_dataset)
        X_aligned = warper.transform(binned_dataset)
        quality = warper.get_alignment_quality(binned_dataset, X_aligned)
        assert "improvement" in quality.columns

    def test_get_alignment_quality_piecewise(self, binned_dataset: pd.DataFrame):
        """Test quality with piecewise method."""
        warper = Warping(method="piecewise", n_segments=3)
        warper.fit(binned_dataset)
        X_aligned = warper.transform(binned_dataset)
        quality = warper.get_alignment_quality(binned_dataset, X_aligned)
        assert len(quality) == len(binned_dataset)

    def test_get_alignment_quality_without_aligned(self, binned_dataset: pd.DataFrame):
        """Test quality when X_aligned is None (auto-transform)."""
        warper = Warping(method="shift")
        warper.fit(binned_dataset)
        quality = warper.get_alignment_quality(binned_dataset)
        assert quality is not None
        assert len(quality) == len(binned_dataset)


class TestWarpingPlot:
    """Tests for the plot_alignment function."""

    def test_plot_alignment_basic(self, binned_dataset: pd.DataFrame):
        """Test basic alignment plot."""
        from maldiamrkit.visualization import plot_alignment

        warper = Warping(method="shift")
        warper.fit(binned_dataset)
        X_aligned = warper.transform(binned_dataset)
        fig, axes = plot_alignment(
            warper, binned_dataset, X_aligned, indices=[0, 1], show=False
        )
        assert fig is not None
        assert axes.shape == (2, 2)
        plt.close(fig)

    def test_plot_alignment_single_index(self, binned_dataset: pd.DataFrame):
        """Test plot with single index."""
        from maldiamrkit.visualization import plot_alignment

        warper = Warping(method="shift")
        warper.fit(binned_dataset)
        X_aligned = warper.transform(binned_dataset)
        fig, axes = plot_alignment(
            warper, binned_dataset, X_aligned, indices=[0], show=False
        )
        assert fig is not None
        plt.close(fig)

    def test_plot_alignment_without_aligned(self, binned_dataset: pd.DataFrame):
        """Test plot when X_aligned is None (auto-transform)."""
        from maldiamrkit.visualization import plot_alignment

        warper = Warping(method="shift")
        warper.fit(binned_dataset)
        fig, axes = plot_alignment(warper, binned_dataset, indices=[0], show=False)
        assert fig is not None
        plt.close(fig)

    def test_plot_alignment_invalid_index(self, binned_dataset: pd.DataFrame):
        """Test plot with invalid index raises ValueError."""
        from maldiamrkit.visualization import plot_alignment

        warper = Warping(method="shift")
        warper.fit(binned_dataset)
        with pytest.raises(ValueError, match="out of bounds"):
            plot_alignment(warper, binned_dataset, indices=[999])


class TestWarpingSklearn:
    """Tests for sklearn compatibility."""

    def test_sklearn_clone(self, binned_dataset: pd.DataFrame):
        """Test that warper can be cloned."""
        from sklearn.base import clone

        warper = Warping(method="piecewise", n_segments=3, n_jobs=2)
        cloned = clone(warper)

        assert cloned.method == "piecewise"
        assert cloned.n_segments == 3
        assert cloned.n_jobs == 2

    def test_sklearn_pipeline(self, binned_dataset: pd.DataFrame):
        """Test warper in sklearn pipeline."""
        from sklearn.pipeline import Pipeline

        pipe = Pipeline(
            [
                ("warp", Warping(method="shift")),
            ]
        )

        result = pipe.fit_transform(binned_dataset)

        assert result.shape == binned_dataset.shape

    def test_params_round_trip(self):
        from sklearn.base import clone

        warper = Warping(
            method="lowess", shift_estimator="peaks", min_gain=0.1, min_support=0.7
        )
        warper.set_params(n_peaks=None, max_shift=4)
        cloned = clone(warper)
        for name, value in warper.get_params(deep=False).items():
            if name != "peak_detector":
                assert cloned.get_params(deep=False)[name] == value
        assert cloned.min_support == 0.7
        assert cloned.n_peaks is None
        assert cloned.max_shift == 4

    @pytest.mark.parametrize("method", NON_DTW_METHODS_LOWESS_SLOW)
    def test_transform_is_independent_of_the_batch(self, binned_dataset, method):
        train, test = binned_dataset.iloc[:30], binned_dataset.iloc[30:]
        warper = Warping(method=method, n_jobs=2).fit(train)
        reference = warper.ref_spec_.copy()
        together = warper.transform(test)
        for i in range(len(test)):
            alone = warper.transform(test.iloc[[i]])
            pd.testing.assert_frame_equal(alone, together.iloc[[i]], check_exact=True)
        np.testing.assert_array_equal(warper.ref_spec_, reference)

    def test_fit_uses_only_the_training_fold(self, binned_dataset):
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import KFold, cross_validate
        from sklearn.pipeline import Pipeline

        y = np.arange(len(binned_dataset)) % 2
        pipe = Pipeline(
            [
                ("warp", Warping(method="linear")),
                ("clf", LogisticRegression(max_iter=200)),
            ]
        )
        folds = list(KFold(n_splits=3, shuffle=True, random_state=0).split(y))
        result = cross_validate(
            pipe, binned_dataset, y, cv=folds, return_estimator=True
        )
        for estimator, (train, _) in zip(result["estimator"], folds, strict=True):
            warp = estimator.named_steps["warp"]
            X_train = binned_dataset.iloc[train].to_numpy()
            np.testing.assert_array_equal(
                warp.prealign_ref_, np.median(X_train, axis=0)
            )
            shift = ShiftStrategy(max_shift=3, estimator="xcorr", min_gain=0.05)
            shifted = [
                _shift_row(r, shift.estimate_binned(r, None, None, warp.prealign_ref_))
                for r in X_train
            ]
            np.testing.assert_array_equal(warp.ref_spec_, np.median(shifted, axis=0))

    def test_grid_search_over_alignment_parameters(self, binned_dataset):
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import GridSearchCV
        from sklearn.pipeline import Pipeline

        y = np.arange(len(binned_dataset)) % 2
        pipe = Pipeline(
            [
                ("warp", Warping()),
                ("clf", LogisticRegression(max_iter=200)),
            ]
        )
        grid = {
            "warp__method": ["shift", "linear"],
            "warp__min_support": [0.5, 0.6],
            "warp__n_peaks": [100, None],
        }
        search = GridSearchCV(pipe, grid, cv=3, n_jobs=2).fit(binned_dataset, y)
        assert len(search.cv_results_["params"]) == 8
        assert set(search.best_params_) == set(grid)


class TestWarpingPrealign:
    """The global-shift step that runs before a non-rigid warp."""

    @pytest.mark.parametrize("method", ["shift", "linear", "dtw"])
    def test_reference_steps(self, real_binned_dataset, method):
        X = real_binned_dataset.to_numpy(dtype=np.float64)
        warper = Warping(method=method).fit(real_binned_dataset)
        np.testing.assert_array_equal(warper.prealign_ref_, np.median(X, axis=0))
        shift = ShiftStrategy(max_shift=3, estimator="xcorr", min_gain=0.05)
        shifted = [
            _shift_row(r, shift.estimate_binned(r, None, None, warper.prealign_ref_))
            for r in X
        ]
        np.testing.assert_allclose(warper.ref_spec_, np.median(shifted, axis=0))

    def test_off_fits_one_pass(self, real_binned_dataset):
        warper = Warping(method="linear", prealign=False).fit(real_binned_dataset)
        assert warper.prealign_ref_ is None
        np.testing.assert_array_equal(
            warper.ref_spec_, real_binned_dataset.median(axis=0).to_numpy()
        )

    def test_total_shift_stays_within_max_shift(self):
        # The second spectrum sits 4 bins after the first (the reference):
        # the first step moves it by max_shift=2 and the second may not add
        # to that, so it ends 2 bins after the reference.
        x = np.arange(300, dtype=float)
        centres = np.array([40.0, 90.0, 150.0, 200.0, 260.0])
        reference = np.exp(-0.5 * ((x[:, None] - centres) / 1.5) ** 2).sum(axis=1)
        X = pd.DataFrame([reference, _shift_row(reference, 4)])
        warper = Warping(max_shift=2, reference=0).fit(X)
        np.testing.assert_allclose(
            warper.transform(X).to_numpy()[1], _shift_row(reference, 2), atol=1e-12
        )

    @pytest.mark.filterwarnings("ignore:Warping produced non-monotonic")
    @pytest.mark.parametrize("method", ["linear", "quadratic", "lowess"])
    def test_aligns_two_calibrations(self, method):
        # Two equal groups of spectra, the second moved by 3 bins and
        # stretched: their median holds two copies of every peak. The shift
        # step realigns one group onto the other before the warp is fitted.
        rng = np.random.default_rng(0)
        x = np.arange(1200, dtype=float)
        centres = np.sort(rng.choice(np.arange(40, 1160), 40, replace=False))
        heights = rng.uniform(0.3, 1.0, len(centres))

        def spectrum(at):
            scale = heights * rng.uniform(0.8, 1.2, len(heights))
            peaks = scale[:, None] * np.exp(-0.5 * ((x - at[:, None]) / 1.5) ** 2)
            return peaks.sum(axis=0) + 1e-4

        X = pd.DataFrame(
            [spectrum(centres) for _ in range(8)]
            + [spectrum(1.002 * centres + 3) for _ in range(8)]
        )

        def spread(aligned):
            found = []
            for row in aligned.to_numpy():
                peaks = find_peaks(row, prominence=0.05)[0]
                found.append(peaks[np.abs(peaks[:, None] - centres).argmin(axis=0)])
            return np.std(np.vstack(found), axis=0).mean()

        one_pass = Warping(method=method, prealign=False).fit(X).transform(X)
        two_steps = Warping(method=method).fit(X).transform(X)
        assert spread(X) > 1.5
        assert spread(one_pass) > 1.0
        assert spread(two_steps) < 0.3


class TestWarpingRenormalize:
    """Optional division of each aligned spectrum by its total intensity."""

    @pytest.mark.parametrize(
        "method", ["shift", "lowess", pytest.param("dtw", marks=pytest.mark.slow)]
    )
    def test_sums_to_one_and_keeps_the_shape(self, real_binned_dataset, method):
        X = real_binned_dataset
        plain = Warping(method=method).fit(X).transform(X).to_numpy()
        scaled = Warping(method=method, renormalize=True).fit(X).transform(X)
        np.testing.assert_allclose(scaled.sum(axis=1), 1.0)
        np.testing.assert_allclose(
            scaled.to_numpy(), plain / plain.sum(axis=1, keepdims=True)
        )

    def test_empty_spectrum_stays_empty(self, binned_dataset):
        X = binned_dataset.iloc[:6].copy()
        X.iloc[0] = 0.0
        out = Warping(renormalize=True).fit(X).transform(X)
        assert (out.iloc[0] == 0.0).all()
        assert not out.isna().to_numpy().any()


class TestWarpingValidation:
    """Tests for input validation and error handling."""

    def test_fit_empty_dataframe(self):
        w = Warping(method="shift")
        with pytest.raises(ValueError, match="empty"):
            w.fit(pd.DataFrame())

    def test_fit_invalid_reference_type(self, binned_dataset: pd.DataFrame):
        w = Warping(method="shift", reference="invalid")
        with pytest.raises(ValueError, match="Unsupported reference"):
            w.fit(binned_dataset)

    def test_fit_invalid_n_segments(self, binned_dataset: pd.DataFrame):
        w = Warping(method="piecewise", n_segments=0)
        with pytest.raises(ValueError, match="n_segments"):
            w.fit(binned_dataset)

    def test_fit_invalid_max_shift(self, binned_dataset: pd.DataFrame):
        w = Warping(method="shift", max_shift=-1)
        with pytest.raises(ValueError, match="max_shift"):
            w.fit(binned_dataset)

    def test_fit_invalid_lowess_frac(self, binned_dataset: pd.DataFrame):
        w = Warping(method="lowess", lowess_frac=0.0)
        with pytest.raises(ValueError, match="lowess_frac"):
            w.fit(binned_dataset)

    def test_fit_invalid_lowess_it(self, binned_dataset: pd.DataFrame):
        w = Warping(method="lowess", lowess_it=-1)
        with pytest.raises(ValueError, match="lowess_it"):
            w.fit(binned_dataset)

    @pytest.mark.parametrize("value", [-0.1, 1.5])
    def test_fit_invalid_min_support(self, binned_dataset, value):
        with pytest.raises(ValueError, match="min_support"):
            Warping(method="linear", min_support=value).fit(binned_dataset)

    def test_fit_invalid_n_peaks(self, binned_dataset):
        with pytest.raises(ValueError, match="n_peaks"):
            Warping(method="linear", n_peaks=0).fit(binned_dataset)

    def test_transform_unknown_method(self, binned_dataset: pd.DataFrame):
        w = Warping(method="shift")
        w.fit(binned_dataset)
        w.method = "nonexistent"
        with pytest.raises(KeyError):
            w.transform(binned_dataset)

    def test_reference_quality_warning(self, binned_dataset: pd.DataFrame):
        w = Warping(method="shift", min_reference_peaks=9999)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            with pytest.raises(UserWarning, match="peaks detected"):
                w.fit(binned_dataset)

    def test_fit_with_int_reference(self, binned_dataset: pd.DataFrame):
        w = Warping(method="shift", reference=0)
        w.fit(binned_dataset)
        np.testing.assert_array_equal(w.ref_spec_, binned_dataset.iloc[0].to_numpy())

    def test_get_alignment_quality_unfitted_raises(self, binned_dataset: pd.DataFrame):
        w = Warping(method="shift")
        with pytest.raises(RuntimeError, match="fitted"):
            w.get_alignment_quality(binned_dataset)

    def test_transform_feature_mismatch(self, binned_dataset: pd.DataFrame):
        w = Warping(method="shift")
        w.fit(binned_dataset)
        wrong_cols = binned_dataset.iloc[:, :10]
        with pytest.raises(ValueError, match="does not match"):
            w.transform(wrong_cols)

    def test_alignment_quality_nan_correlation(self, binned_dataset: pd.DataFrame):
        w = Warping(method="shift")
        w.fit(binned_dataset)
        X_aligned = w.transform(binned_dataset)
        X_const = binned_dataset.copy()
        X_const.iloc[0] = 0.0
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=RuntimeWarning)
            with pytest.warns(UserWarning, match="constant signal"):
                quality = w.get_alignment_quality(X_const, X_aligned)
        assert quality.loc[X_const.index[0], "correlation_before"] == 0.0


REALISTIC_CONFIGS = [
    pytest.param({"method": "shift"}, id="shift-xcorr"),
    pytest.param({"method": "shift", "shift_estimator": "peaks"}, id="shift-peaks"),
    pytest.param({"method": "linear"}, id="linear"),
    pytest.param({"method": "piecewise"}, id="piecewise"),
    pytest.param({"method": "quadratic"}, id="quadratic"),
    pytest.param({"method": "cubic"}, id="cubic"),
    pytest.param({"method": "lowess"}, id="lowess", marks=pytest.mark.slow),
    pytest.param({"method": "dtw"}, id="dtw", marks=pytest.mark.slow),
]


def _rowwise_corr(Y: np.ndarray, ref: np.ndarray) -> np.ndarray:
    """Pearson correlation of every row of ``Y`` with ``ref`` (0 if undefined)."""
    Yc = Y - Y.mean(axis=1, keepdims=True)
    rc = ref - ref.mean()
    denom = np.linalg.norm(Yc, axis=1) * np.linalg.norm(rc)
    return np.divide(Yc @ rc, denom, out=np.zeros(len(Y)), where=denom > 0)


def _assert_alignment_contract(warper: Warping, X: pd.DataFrame) -> None:
    """Check what Warping promises on ``X``.

    No warp is rejected as non-monotonic; with "shift" and "dtw" no
    spectrum ends up less correlated with the reference than before; and
    (for every method but "dtw") no intensity is created and only content
    near the edges can be lost.
    """
    aligned, n_non_monotonic = _transform_with_count(warper, X)
    before = X.to_numpy(dtype=np.float64)
    after = aligned.to_numpy(dtype=np.float64)
    ref = warper.ref_spec_.astype(np.float64)
    assert n_non_monotonic == 0
    if warper.method in ("shift", "dtw"):
        assert np.all(_rowwise_corr(after, ref) >= _rowwise_corr(before, ref) - 1e-9)
    if warper.method != "dtw":
        rtol = max(1e-9, 10 * np.finfo(aligned.dtypes.iloc[0]).eps)
        margin = 2 * warper.max_shift
        near_edges = before[:, :margin].sum(axis=1) + before[:, -margin:].sum(axis=1)
        lost = before.sum(axis=1) - after.sum(axis=1)
        assert np.all(lost >= -rtol * before.sum(axis=1))
        assert np.all(lost <= near_edges + rtol * before.sum(axis=1))


class TestWarpingRealSpectra:
    """Every method on the real spectra shipped in ``data/``."""

    @pytest.mark.parametrize("config", REALISTIC_CONFIGS)
    def test_contract(self, real_binned_dataset, config):
        warper = Warping(n_jobs=2, **config).fit(real_binned_dataset)
        _assert_alignment_contract(warper, real_binned_dataset)

    def test_xcorr_recovers_a_known_shift(self, real_binned_dataset):
        # Each real spectrum, moved by 2 bins, is realigned onto itself.
        estimator = ShiftStrategy(max_shift=3, estimator="xcorr", min_gain=0.05)
        for row in real_binned_dataset.to_numpy(dtype=np.float64):
            moved = _shift_row(row, 2)
            assert estimator.estimate_binned(moved, None, None, row) == -2

    @pytest.mark.parametrize(
        "method", ["linear", "piecewise", "quadratic", "cubic", "lowess"]
    )
    def test_non_rigid_methods_undo_a_known_drift(self, real_binned_dataset, method):
        X = real_binned_dataset.to_numpy(dtype=np.float64)[:8]
        axis = np.arange(X.shape[1], dtype=float)
        mz = real_binned_dataset.columns.astype(float).to_numpy()
        drift = 2.5 * mz / mz[-1]
        for row in X:
            drifted = _conservative_resample(axis, axis + drift, row)
            frame = pd.DataFrame([row, drifted], columns=real_binned_dataset.columns)
            warper = Warping(method=method, reference=0).fit(frame)
            aligned = warper.transform(frame).to_numpy()[1]
            before = np.abs(drifted - row).sum()
            assert np.abs(aligned - row).sum() < 0.45 * before

    def test_shift_without_peak_detection(self, real_binned_dataset, monkeypatch):
        def fail(*args, **kwargs):
            raise AssertionError("peak detection should not run")

        warper = Warping(method="shift").fit(real_binned_dataset)
        monkeypatch.setattr(warper.peak_detector, "transform", fail)
        warper.transform(real_binned_dataset)


DRIAMS_POOL = os.environ.get("MALDIAMRKIT_DRIAMS_POOL")


@pytest.fixture(scope="module")
def driams_sample() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fit and evaluation spectra drawn from a binned DRIAMS pool (``.npy``).

    Set ``MALDIAMRKIT_DRIAMS_POOL`` to the pool path to enable these tests.
    """
    if not DRIAMS_POOL or not Path(DRIAMS_POOL).exists():
        pytest.skip("MALDIAMRKIT_DRIAMS_POOL does not point to a binned DRIAMS pool")
    pool = np.load(DRIAMS_POOL, mmap_mode="r")
    rows = np.sort(np.random.default_rng(0).choice(len(pool), 1200, replace=False))
    sample = np.asarray(pool[rows], dtype=np.float32)
    return pd.DataFrame(sample[:1000]), pd.DataFrame(sample[1000:])


@pytest.mark.slow
class TestWarpingDRIAMS:
    """Every method on a DRIAMS sample, fitted and evaluated on disjoint spectra."""

    @pytest.mark.parametrize("config", REALISTIC_CONFIGS)
    def test_contract(self, driams_sample, config):
        fit, evaluate = driams_sample
        warper = Warping(n_jobs=4, **config).fit(fit)
        _assert_alignment_contract(warper, evaluate)
