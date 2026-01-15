"""Unit tests for RawWarping class and create_raw_input utility."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from maldiamrkit.alignment.raw_warping import RawWarping, create_raw_input


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
                for m, inten in zip(mz, intensity):
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
                for m, inten in zip(mz, intensity):
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
                for m, inten in zip(mz, intensity):
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
