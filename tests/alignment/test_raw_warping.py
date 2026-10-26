"""Unit tests for RawWarping class and create_raw_input utility."""

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from maldiamrkit.alignment import RawWarping, create_raw_input
from maldiamrkit.alignment.strategies import _pearson_by_lag
from maldiamrkit.io.readers import read_spectrum
from tests.conftest import SPECTRA_DIR


class TestCreateRawInput:
    """Tests for create_raw_input utility function."""

    def test_discover_files(self, tmp_path: Path):
        """Test automatic file discovery."""
        # Create some spectrum files
        for i in range(3):
            (tmp_path / f"sample_{i}.txt").write_text("2000\t1.0\n3000\t2.0\n")

        X = create_raw_input(tmp_path)

        assert isinstance(X, pd.DataFrame)
        assert len(X) == 3
        assert "path" in X.columns
        assert list(X.index) == ["sample_0", "sample_1", "sample_2"]

    def test_explicit_sample_ids(self, tmp_path: Path):
        """Test with explicit sample IDs."""
        # Create spectrum files
        for sid in ["a", "b", "c"]:
            (tmp_path / f"{sid}.txt").write_text("2000\t1.0\n3000\t2.0\n")

        X = create_raw_input(tmp_path, sample_ids=["a", "b"])

        assert len(X) == 2
        assert list(X.index) == ["a", "b"]

    def test_custom_extension(self, tmp_path: Path):
        """Test with custom file extension."""
        (tmp_path / "sample.csv").write_text("2000\t1.0\n3000\t2.0\n")

        X = create_raw_input(tmp_path, file_extension=".csv")

        assert len(X) == 1
        assert X.index[0] == "sample"

    def test_no_files_raises(self, tmp_path: Path):
        """Test that empty directory raises error."""
        with pytest.raises(ValueError, match="No files"):
            create_raw_input(tmp_path)


class TestRawWarpingInit:
    """Tests for RawWarping initialization."""

    def test_default_init(self):
        """Test default initialization."""
        warper = RawWarping()
        assert warper.method == "shift"
        assert warper.reference == "median"
        assert warper.n_jobs == 1
        assert warper.bin_width == 3

    def test_all_methods(self):
        """Test all method options are accepted."""
        for method in ["shift", "linear", "piecewise", "dtw"]:
            warper = RawWarping(method=method)
            assert warper.method == method

    def test_custom_init(self):
        """Test custom initialization parameters."""
        warper = RawWarping(
            method="piecewise",
            bin_width=5.0,
            n_segments=10,
            smooth_sigma=3.0,
            n_jobs=4,
        )
        assert warper.bin_width == 5.0
        assert warper.n_segments == 10
        assert warper.smooth_sigma == 3.0
        assert warper.n_jobs == 4


class TestRawWarpingFitTransform:
    """Test RawWarping fit and transform with synthetic data."""

    @pytest.fixture
    def synthetic_raw_spectra(self, tmp_path: Path):
        """Create synthetic raw spectrum files."""
        rng = np.random.default_rng(42)
        mz = np.linspace(2000, 20000, 1000)
        sample_ids = []

        for i in range(5):
            # Add slight shift to each spectrum
            shift = i * 2
            intensity = np.exp(-((mz - 5000 - shift) ** 2) / 50000)
            intensity += np.exp(-((mz - 10000 - shift) ** 2) / 50000)
            intensity += rng.normal(0, 0.01, len(mz))
            intensity = np.maximum(intensity, 0)

            sample_id = f"sample_{i}"
            path = tmp_path / f"{sample_id}.txt"
            with open(path, "w") as f:
                for m, inten in zip(mz, intensity, strict=True):
                    f.write(f"{m}\t{inten}\n")
            sample_ids.append(sample_id)

        # Create input DataFrame using utility function
        X = create_raw_input(tmp_path, sample_ids=sample_ids)

        return tmp_path, X

    def test_fit_with_paths(self, synthetic_raw_spectra):
        """Test fit with path-based DataFrame."""
        spectra_dir, X = synthetic_raw_spectra

        warper = RawWarping(method="shift")
        warper.fit(X)

        assert hasattr(warper, "ref_mz_")
        assert hasattr(warper, "ref_intensity_")
        assert hasattr(warper, "output_columns_")
        assert len(warper.ref_mz_) > 0

    def test_transform_produces_binned_output(self, synthetic_raw_spectra):
        """Test transform produces proper binned output."""
        spectra_dir, X = synthetic_raw_spectra

        warper = RawWarping(method="shift", bin_width=3.0)
        result = warper.fit_transform(X)

        assert isinstance(result, pd.DataFrame)
        assert len(result) == len(X)
        # Output should not have "path" column anymore
        assert "path" not in result.columns
        # Output should have numeric columns (m/z values)
        assert result.shape[1] > 0

    def test_transform_shift(self, synthetic_raw_spectra):
        """Test transform with shift method."""
        spectra_dir, X = synthetic_raw_spectra

        warper = RawWarping(method="shift", bin_width=3.0)
        result = warper.fit_transform(X)

        assert isinstance(result, pd.DataFrame)
        # Check that values are non-zero
        assert result.values.sum() > 0

    def test_transform_linear(self, synthetic_raw_spectra):
        """Test transform with linear method."""
        spectra_dir, X = synthetic_raw_spectra

        warper = RawWarping(method="linear", bin_width=3.0)
        result = warper.fit_transform(X)

        assert isinstance(result, pd.DataFrame)

    def test_transform_piecewise(self, synthetic_raw_spectra):
        """Test transform with piecewise method."""
        spectra_dir, X = synthetic_raw_spectra

        warper = RawWarping(
            method="piecewise",
            bin_width=3.0,
            n_segments=3,
        )
        result = warper.fit_transform(X)

        assert isinstance(result, pd.DataFrame)

    @pytest.mark.slow
    def test_transform_dtw(self, synthetic_raw_spectra):
        """Test transform with DTW method."""
        spectra_dir, X = synthetic_raw_spectra
        # Use fewer samples for DTW (it's slow)
        X_small = X.iloc[:3]

        warper = RawWarping(method="dtw", bin_width=3.0)
        result = warper.fit_transform(X_small)

        assert isinstance(result, pd.DataFrame)
        assert len(result) == 3

    def test_get_alignment_quality(self, synthetic_raw_spectra):
        """Test alignment quality method."""
        spectra_dir, X = synthetic_raw_spectra

        warper = RawWarping(method="shift", bin_width=3.0)
        result = warper.fit_transform(X)

        quality = warper.get_alignment_quality(X, result)

        assert isinstance(quality, pd.DataFrame)
        assert "correlation_before" in quality.columns
        assert "correlation_after" in quality.columns
        assert "improvement" in quality.columns
        assert len(quality) == len(X)


class TestRawWarpingAveraging:
    """Tests for the _original_id replicate averaging path in transform."""

    @pytest.fixture
    def spectra_with_original_id(self, tmp_path: Path):
        """Create spectra with _original_id for averaging."""
        rng = np.random.default_rng(42)
        mz = np.linspace(2000, 20000, 500)
        for i in range(4):
            intensity = np.exp(-((mz - 5000 - i) ** 2) / 50000)
            intensity += rng.normal(0, 0.01, len(mz))
            intensity = np.maximum(intensity, 0)
            with open(tmp_path / f"s{i}.txt", "w") as f:
                for m, inten in zip(mz, intensity, strict=True):
                    f.write(f"{m}\t{inten}\n")

        X = create_raw_input(tmp_path, sample_ids=[f"s{i}" for i in range(4)])
        # s0 and s1 are replicates of "A"; s2 and s3 are replicates of "B"
        X["_original_id"] = ["A", "A", "B", "B"]
        return X

    def test_transform_averages_replicates(self, spectra_with_original_id):
        """Verify _original_id groups are averaged in transform output."""
        X = spectra_with_original_id
        warper = RawWarping(method="shift", bin_width=3.0)
        result = warper.fit_transform(X)
        assert len(result) == 2
        assert set(result.index) == {"A", "B"}
        assert "_original_id" not in result.columns

    def test_averaged_rows_keep_the_input_order(self, spectra_with_original_id):
        X = spectra_with_original_id.copy()
        X["_original_id"] = ["iso_z", "iso_a", "iso_z", "iso_m"]
        result = RawWarping(method="shift", bin_width=3.0).fit_transform(X)
        assert list(result.index) == ["iso_z", "iso_a", "iso_m"]

    def test_transform_without_original_id_passthrough(self, tmp_path: Path):
        """Verify no averaging when _original_id is absent."""
        rng = np.random.default_rng(42)
        mz = np.linspace(2000, 20000, 500)
        for i in range(3):
            intensity = np.exp(-((mz - 5000) ** 2) / 50000)
            intensity += rng.normal(0, 0.01, len(mz))
            intensity = np.maximum(intensity, 0)
            with open(tmp_path / f"s{i}.txt", "w") as f:
                for m, inten in zip(mz, intensity, strict=True):
                    f.write(f"{m}\t{inten}\n")
        X = create_raw_input(tmp_path)
        warper = RawWarping(method="shift", bin_width=3.0)
        result = warper.fit_transform(X)
        assert len(result) == 3


class TestRawWarpingReferenceEdgeCases:
    """Edge-case tests for reference parameter."""

    def test_reference_out_of_bounds_raises(self, tmp_path: Path):
        """Verify out-of-bounds integer reference raises ValueError."""
        for i in range(3):
            (tmp_path / f"s{i}.txt").write_text("2000\t1.0\n3000\t2.0\n")
        X = create_raw_input(tmp_path)
        warper = RawWarping(method="shift", reference=999)
        with pytest.raises(ValueError, match="out of bounds"):
            warper.fit(X)


class TestRawWarpingPipelineCompatibility:
    """Tests for sklearn Pipeline compatibility."""

    @pytest.fixture
    def synthetic_raw_spectra_pipeline(self, tmp_path: Path):
        """Create synthetic data for pipeline tests."""
        rng = np.random.default_rng(42)
        mz = np.linspace(2000, 20000, 500)

        for i in range(10):
            shift = i * 2
            intensity = np.exp(-((mz - 5000 - shift) ** 2) / 50000)
            intensity += np.exp(-((mz - 10000 - shift) ** 2) / 50000)
            intensity += rng.normal(0, 0.01, len(mz))
            intensity = np.maximum(intensity, 0)

            path = tmp_path / f"sample_{i}.txt"
            with open(path, "w") as f:
                for m, inten in zip(mz, intensity, strict=True):
                    f.write(f"{m}\t{inten}\n")

        X = create_raw_input(tmp_path)

        return tmp_path, X

    def test_sklearn_pipeline_integration(self, synthetic_raw_spectra_pipeline):
        """Test RawWarping works in sklearn Pipeline."""
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler

        spectra_dir, X = synthetic_raw_spectra_pipeline

        pipe = Pipeline(
            [
                ("warp", RawWarping(method="shift", bin_width=3.0)),
                ("scaler", StandardScaler()),
            ]
        )

        result = pipe.fit_transform(X)

        # StandardScaler outputs numpy array
        assert isinstance(result, np.ndarray)
        assert len(result) == len(X)


class TestRawWarpingParallelization:
    """Tests for parallelization."""

    @pytest.fixture
    def synthetic_raw_spectra_parallel(self, tmp_path: Path):
        """Create synthetic raw spectrum files for parallel tests."""
        rng = np.random.default_rng(42)
        mz = np.linspace(2000, 20000, 500)

        for i in range(10):
            shift = i * 2
            intensity = np.exp(-((mz - 5000 - shift) ** 2) / 50000)
            intensity += np.exp(-((mz - 10000 - shift) ** 2) / 50000)
            intensity += rng.normal(0, 0.01, len(mz))
            intensity = np.maximum(intensity, 0)

            path = tmp_path / f"sample_{i}.txt"
            with open(path, "w") as f:
                for m, inten in zip(mz, intensity, strict=True):
                    f.write(f"{m}\t{inten}\n")

        X = create_raw_input(tmp_path)

        return tmp_path, X

    def test_parallel_produces_same_results(self, synthetic_raw_spectra_parallel):
        """Test that parallel processing produces same results as sequential."""
        spectra_dir, X = synthetic_raw_spectra_parallel

        warper_seq = RawWarping(method="shift", bin_width=3.0, n_jobs=1)
        warper_par = RawWarping(method="shift", bin_width=3.0, n_jobs=2)

        result_seq = warper_seq.fit_transform(X)
        result_par = warper_par.fit_transform(X)

        pd.testing.assert_frame_equal(result_seq, result_par)


class TestRawWarpingErrors:
    """Tests for error handling."""

    def test_missing_path_column_raises(self, tmp_path: Path):
        """Test that missing 'path' column raises ValueError."""
        X = pd.DataFrame({"dummy": [0]}, index=["sample_0"])
        warper = RawWarping(method="shift")

        with pytest.raises(ValueError, match="'path' column"):
            warper.fit(X)

    def test_missing_file_raises(self, tmp_path: Path):
        """Test that missing spectrum file raises FileNotFoundError."""
        X = pd.DataFrame(
            {"path": [str(tmp_path / "nonexistent.txt")]}, index=["sample_0"]
        )
        warper = RawWarping(method="shift")

        with pytest.raises(FileNotFoundError):
            warper.fit(X)

    def test_fit_empty_dataframe(self):
        warper = RawWarping(method="shift")
        with pytest.raises(ValueError, match="empty"):
            warper.fit(pd.DataFrame())

    def test_fit_invalid_method(self, tmp_path: Path):
        for i in range(3):
            (tmp_path / f"s{i}.txt").write_text("2000\t1.0\n3000\t2.0\n")
        with pytest.raises(ValueError, match="is not a valid"):
            RawWarping(method="nonexistent")

    def test_fit_invalid_reference(self, tmp_path: Path):
        for i in range(3):
            (tmp_path / f"s{i}.txt").write_text("2000\t1.0\n3000\t2.0\n")
        X = create_raw_input(tmp_path)
        warper = RawWarping(reference="invalid")
        with pytest.raises(ValueError, match="Unsupported reference"):
            warper.fit(X)

    def test_fit_integer_reference(self, tmp_path: Path):
        rng = np.random.default_rng(42)
        mz = np.linspace(2000, 20000, 500)
        for i in range(3):
            intensity = np.exp(-((mz - 5000 - i * 2) ** 2) / 50000)
            intensity += rng.normal(0, 0.01, len(mz))
            intensity = np.maximum(intensity, 0)
            path = tmp_path / f"s{i}.txt"
            with open(path, "w") as f:
                for m, inten in zip(mz, intensity, strict=True):
                    f.write(f"{m}\t{inten}\n")
        X = create_raw_input(tmp_path)
        warper = RawWarping(method="shift", reference=0)
        warper.fit(X)
        assert hasattr(warper, "ref_mz_")

    def test_fit_reference_quality_warning(self, tmp_path: Path):
        rng = np.random.default_rng(42)
        mz = np.linspace(2000, 20000, 500)
        for i in range(3):
            intensity = np.exp(-((mz - 5000) ** 2) / 50000)
            intensity += rng.normal(0, 0.01, len(mz))
            intensity = np.maximum(intensity, 0)
            with open(tmp_path / f"s{i}.txt", "w") as f:
                for m, inten in zip(mz, intensity, strict=True):
                    f.write(f"{m}\t{inten}\n")
        X = create_raw_input(tmp_path)
        warper = RawWarping(method="shift", min_reference_peaks=9999)
        with pytest.warns(UserWarning, match="peaks"):
            warper.fit(X)

    def test_transform_before_fit(self, tmp_path: Path):
        (tmp_path / "s0.txt").write_text("2000\t1.0\n3000\t2.0\n")
        X = create_raw_input(tmp_path)
        warper = RawWarping(method="shift")
        with pytest.raises(RuntimeError, match="fitted"):
            warper.transform(X)

    def test_transform_missing_path_column(self, tmp_path: Path):
        rng = np.random.default_rng(42)
        mz = np.linspace(2000, 20000, 500)
        for i in range(3):
            intensity = np.exp(-((mz - 5000) ** 2) / 50000)
            intensity += rng.normal(0, 0.01, len(mz))
            intensity = np.maximum(intensity, 0)
            with open(tmp_path / f"s{i}.txt", "w") as f:
                for m, inten in zip(mz, intensity, strict=True):
                    f.write(f"{m}\t{inten}\n")
        X = create_raw_input(tmp_path)
        warper = RawWarping(method="shift")
        warper.fit(X)
        with pytest.raises(ValueError, match="path"):
            warper.transform(pd.DataFrame({"dummy": [0]}))

    @pytest.mark.parametrize(
        ("params", "match"),
        [
            ({"min_support": -0.1}, "min_support"),
            ({"min_support": 1.5}, "min_support"),
            ({"n_peaks": 0}, "n_peaks"),
        ],
    )
    def test_fit_invalid_support_parameters(self, tmp_path: Path, params, match):
        (tmp_path / "s0.txt").write_text("2000\t1.0\n3000\t2.0\n")
        X = create_raw_input(tmp_path)
        with pytest.raises(ValueError, match=match):
            RawWarping(method="linear", **params).fit(X)

    def test_get_alignment_quality_before_fit(self, tmp_path: Path):
        (tmp_path / "s0.txt").write_text("2000\t1.0\n3000\t2.0\n")
        X = create_raw_input(tmp_path)
        warper = RawWarping(method="shift")
        with pytest.raises(RuntimeError, match="fitted"):
            warper.get_alignment_quality(X)

    def test_get_alignment_quality_auto_transform(self, tmp_path: Path):
        rng = np.random.default_rng(42)
        mz = np.linspace(2000, 20000, 500)
        for i in range(3):
            intensity = np.exp(-((mz - 5000 - i * 2) ** 2) / 50000)
            intensity += rng.normal(0, 0.01, len(mz))
            intensity = np.maximum(intensity, 0)
            with open(tmp_path / f"s{i}.txt", "w") as f:
                for m, inten in zip(mz, intensity, strict=True):
                    f.write(f"{m}\t{inten}\n")
        X = create_raw_input(tmp_path)
        warper = RawWarping(method="shift")
        warper.fit(X)
        quality = warper.get_alignment_quality(X)
        assert isinstance(quality, pd.DataFrame)
        assert len(quality) == 3


RAW_CONFIGS = [
    pytest.param({"method": "shift"}, id="shift-xcorr"),
    pytest.param({"method": "shift", "shift_estimator": "peaks"}, id="shift-peaks"),
    pytest.param({"method": "linear"}, id="linear"),
    pytest.param({"method": "piecewise"}, id="piecewise"),
    pytest.param({"method": "quadratic"}, id="quadratic"),
    pytest.param({"method": "lowess"}, id="lowess", marks=pytest.mark.slow),
    pytest.param({"method": "dtw"}, id="dtw", marks=pytest.mark.slow),
]


def _rowwise_corr(Y: np.ndarray, ref: np.ndarray) -> np.ndarray:
    Yc = Y - Y.mean(axis=1, keepdims=True)
    rc = ref - ref.mean()
    denom = np.linalg.norm(Yc, axis=1) * np.linalg.norm(rc)
    return np.divide(Yc @ rc, denom, out=np.zeros(len(Y)), where=denom > 0)


@pytest.fixture(scope="module")
def raw_input() -> pd.DataFrame:
    """Path input for the real raw spectra shipped in ``data/``."""
    if not any(SPECTRA_DIR.glob("*.txt")):
        pytest.skip("No spectrum files found in data directory")
    return create_raw_input(SPECTRA_DIR)


@pytest.fixture(scope="module")
def unaligned(raw_input) -> np.ndarray:
    """The same spectra preprocessed and binned without alignment."""
    warper = RawWarping(method="shift", max_shift_da=0.0, n_jobs=2).fit(raw_input)
    return warper.transform(raw_input).to_numpy(dtype=np.float64)


class TestRawWarpingRealSpectra:
    """Every method on the real raw spectra shipped in ``data/``."""

    @pytest.mark.parametrize("config", RAW_CONFIGS)
    def test_contract(self, raw_input, unaligned, config):
        warper = RawWarping(n_jobs=2, **config).fit(raw_input)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            aligned = warper.transform(raw_input).to_numpy(dtype=np.float64)
        assert not any("non-monotonic" in str(w.message) for w in caught)
        ref = warper._bin_warped(warper.ref_mz_, warper.ref_intensity_)
        ref = ref["intensity"].to_numpy(dtype=np.float64)
        if warper.method in ("shift", "dtw"):
            corr_before = _rowwise_corr(unaligned, ref)
            assert np.all(_rowwise_corr(aligned, ref) >= corr_before - 1e-3)
        if warper.method != "dtw":
            margin = int(np.ceil(2 * warper.max_shift_da / warper.bin_width)) + 1
            near_edges = unaligned[:, :margin].sum(axis=1) + unaligned[:, -margin:].sum(
                axis=1
            )
            lost = unaligned.sum(axis=1) - aligned.sum(axis=1)
            assert np.all(lost >= -1e-9 * unaligned.sum(axis=1))
            assert np.all(lost <= near_edges + 1e-12)

    @pytest.mark.parametrize(
        "method", ["linear", "piecewise", "quadratic", "cubic", "lowess"]
    )
    def test_non_rigid_methods_undo_a_known_drift(self, tmp_path, method):
        for k, source in enumerate(sorted(SPECTRA_DIR.glob("*.txt"))[:3]):
            folder = tmp_path / str(k)
            folder.mkdir()
            raw = read_spectrum(source)
            mz, intensity = raw["mass"].to_numpy(), raw["intensity"].to_numpy()
            np.savetxt(folder / "a.txt", np.c_[mz, intensity], delimiter="\t")
            np.savetxt(folder / "b.txt", np.c_[mz * 1.0004, intensity], delimiter="\t")
            X = create_raw_input(folder)
            unaligned = RawWarping(max_shift_da=0.0, reference=0).fit(X).transform(X)
            original, drifted = unaligned.to_numpy()
            aligned = RawWarping(method=method, reference=0).fit(X).transform(X)
            before = np.abs(drifted - original).sum()
            assert np.abs(aligned.to_numpy()[1] - original).sum() < 0.25 * before


class TestRawWarpingPrealign:
    """The global m/z shift that runs before a non-rigid warp."""

    def test_reference_steps(self, raw_input):
        one_pass = RawWarping(method="shift", prealign=False).fit(raw_input)
        warper = RawWarping(method="linear").fit(raw_input)
        np.testing.assert_array_equal(warper.prealign_mz_, one_pass.ref_mz_)
        np.testing.assert_array_equal(
            warper.prealign_intensity_, one_pass.ref_intensity_
        )
        assert not np.array_equal(warper.ref_intensity_, one_pass.ref_intensity_)
        off = RawWarping(method="linear", prealign=False).fit(raw_input)
        assert off.prealign_mz_ is None
        np.testing.assert_array_equal(off.ref_intensity_, one_pass.ref_intensity_)

    @pytest.mark.parametrize("method", ["shift", "dtw"])
    def test_applies_to_every_method(self, raw_input, method):
        assert RawWarping(method=method).fit(raw_input).prealign_mz_ is not None
        off = RawWarping(method=method, prealign=False).fit(raw_input)
        assert off.prealign_mz_ is None

    def test_total_shift_stays_within_max_shift(self, tmp_path):
        # The second spectrum sits 16 Da after the first (the reference):
        # the first step moves it by max_shift_da=8 and the second may not
        # add to that, so it ends about 8 Da (2 to 3 bins) after the first.
        raw = read_spectrum(sorted(SPECTRA_DIR.glob("*.txt"))[0])
        mz, intensity = raw["mass"].to_numpy(), raw["intensity"].to_numpy()
        for name, offset in (("a", 0.0), ("b", 16.0)):
            np.savetxt(tmp_path / f"{name}.txt", np.c_[mz + offset, intensity])
        X = create_raw_input(tmp_path)
        aligned = RawWarping(max_shift_da=8.0, reference=0).fit(X).transform(X)
        lags, corr = _pearson_by_lag(
            aligned.loc["b"].to_numpy(), aligned.loc["a"].to_numpy(), 6
        )
        assert lags[np.argmax(corr)] in (-3, -2)

    def test_sharpens_a_reference_of_two_calibrations(self, tmp_path):
        # A third of the spectra moved by 6 Da blurs their median; the shift
        # step realigns them before the warp is fitted.
        sources = sorted(SPECTRA_DIR.glob("*.txt"))[:9]
        for k, source in enumerate(sources):
            raw = read_spectrum(source)
            mz = raw["mass"].to_numpy() + (6.0 if k % 3 == 0 else 0.0)
            np.savetxt(
                tmp_path / source.name,
                np.c_[mz, raw["intensity"].to_numpy()],
                delimiter="\t",
            )
        X = create_raw_input(tmp_path)
        moved = X.index[::3]
        target = (
            RawWarping(max_shift_da=0.0)
            .fit(create_raw_input(SPECTRA_DIR).iloc[:9])
            .transform(create_raw_input(SPECTRA_DIR).iloc[:9])
            .median(axis=0)
            .to_numpy()
        )
        one = RawWarping(method="lowess", prealign=False).fit(X).transform(X)
        two = RawWarping(method="lowess").fit(X).transform(X)
        corr_one = _rowwise_corr(one.loc[moved].to_numpy(), target)
        corr_two = _rowwise_corr(two.loc[moved].to_numpy(), target)
        assert corr_two.mean() > corr_one.mean()


class TestRawWarpingRenormalize:
    """Optional division of each output spectrum by its total intensity."""

    @pytest.mark.parametrize(
        "method", ["lowess", pytest.param("dtw", marks=pytest.mark.slow)]
    )
    def test_sums_to_one_and_keeps_the_shape(self, raw_input, method):
        plain = RawWarping(method=method).fit(raw_input).transform(raw_input)
        scaled = RawWarping(method=method, renormalize=True).fit(raw_input)
        scaled = scaled.transform(raw_input)
        np.testing.assert_allclose(scaled.sum(axis=1), 1.0)
        expected = plain.to_numpy() / plain.to_numpy().sum(axis=1, keepdims=True)
        np.testing.assert_allclose(scaled.to_numpy(), expected)


class TestRawWarpingSklearn:
    """Fitted state comes from the training spectra only."""

    def test_params_round_trip(self):
        from sklearn.base import clone

        warper = RawWarping(method="lowess", min_support=0.7)
        warper.set_params(n_peaks=None, max_shift_da=6.0)
        cloned = clone(warper)
        for name, value in warper.get_params(deep=False).items():
            if name != "peak_detector":
                assert cloned.get_params(deep=False)[name] == value
        assert cloned.min_support == 0.7
        assert cloned.n_peaks is None

    @pytest.mark.parametrize("method", ["shift", "linear", "piecewise"])
    def test_transform_is_independent_of_the_batch(self, raw_input, method):
        train, test = raw_input.iloc[:20], raw_input.iloc[20:]
        warper = RawWarping(method=method).fit(train)
        reference = warper.ref_intensity_.copy()
        together = warper.transform(test)
        for i in range(len(test)):
            alone = warper.transform(test.iloc[[i]])
            pd.testing.assert_frame_equal(alone, together.iloc[[i]], check_exact=True)
        np.testing.assert_array_equal(warper.ref_intensity_, reference)

    def test_fit_uses_only_the_training_fold(self, raw_input):
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import KFold, cross_validate
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler

        y = np.arange(len(raw_input)) % 2
        pipe = Pipeline(
            [
                ("warp", RawWarping(method="linear")),
                ("scale", StandardScaler()),
                ("clf", LogisticRegression(max_iter=200)),
            ]
        )
        folds = list(KFold(n_splits=3, shuffle=True, random_state=0).split(y))
        result = cross_validate(pipe, raw_input, y, cv=folds, return_estimator=True)
        for estimator, (train, _) in zip(result["estimator"], folds, strict=True):
            alone = RawWarping(method="linear").fit(raw_input.iloc[train])
            np.testing.assert_array_equal(
                estimator.named_steps["warp"].ref_intensity_, alone.ref_intensity_
            )


class TestRawWarpingAdaptiveBins:
    """Adaptive bin edges are learned from the reference in fit."""

    def test_every_spectrum_gets_the_fitted_columns(self, raw_input):
        warper = RawWarping(bin_method="adaptive").fit(raw_input.iloc[:3])
        same = warper.transform(raw_input.iloc[:3])
        other = warper.transform(raw_input.iloc[3:6])
        assert same.columns.equals(warper.output_columns_)
        assert other.columns.equals(warper.output_columns_)

    def test_refit_with_uniform_bins_drops_the_adaptive_edges(self, raw_input):
        warper = RawWarping(bin_method="adaptive").fit(raw_input.iloc[:3])
        warper.set_params(bin_method="uniform").fit(raw_input.iloc[:3])
        out = warper.transform(raw_input.iloc[:3])
        assert out.shape[1] == 6000


class TestBinOnEdges:
    """The binning core gives the same result as bin_spectrum."""

    def test_matches_bin_spectrum(self):
        from maldiamrkit.preprocessing import PreprocessingPipeline
        from maldiamrkit.preprocessing.binning import (
            _bin_on_edges,
            _uniform_edges,
            bin_spectrum,
        )
        from maldiamrkit.preprocessing.pipeline import preprocess

        for f in sorted(SPECTRA_DIR.glob("*.txt"))[:5]:
            pre = preprocess(read_spectrum(f), PreprocessingPipeline.default())
            edges = _uniform_edges(2000, 20000, 3)
            expected, meta = bin_spectrum(pre, method="custom", custom_edges=edges)
            got, got_meta = _bin_on_edges(pre, edges)
            pd.testing.assert_frame_equal(got, expected)
            pd.testing.assert_frame_equal(got_meta, meta)
