"""End-to-end integration tests for the full MALDI-TOF preprocessing pipeline."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from maldiamrkit import MaldiSpectrum
from maldiamrkit.alignment import Warping
from maldiamrkit.data import DatasetBuilder, FlatLayout, ProcessingHandler
from maldiamrkit.detection import MaldiPeakDetector
from maldiamrkit.preprocessing import (
    SpectrumQuality,
    bin_spectrum,
    detect_outlier_replicates,
    merge_replicates,
    preprocess,
)


def _make_spectrum(seed: int = 42) -> pd.DataFrame:
    """Generate a synthetic spectrum with known peaks."""
    rng = np.random.default_rng(seed)
    mz = np.linspace(2000, 20000, 18000)
    intensity = np.zeros_like(mz)
    for pos in [3000, 5000, 7500, 10000, 12500, 15000]:
        intensity += 1000 * np.exp(-0.5 * ((mz - pos) / 10.0) ** 2)
    intensity += rng.normal(0, 10.0, len(intensity))
    intensity = np.maximum(intensity, 0)
    return pd.DataFrame({"mass": mz, "intensity": intensity})


@pytest.mark.slow
class TestEndToEnd:
    """Integration tests combining multiple pipeline steps."""

    def test_full_pipeline_synthetic(self):
        """Test: generate -> preprocess -> bin -> valid output."""
        raw = _make_spectrum(seed=42)
        preprocessed = preprocess(raw)
        binned, metadata = bin_spectrum(preprocessed, bin_width=3, method="uniform")

        assert isinstance(binned, pd.DataFrame)
        assert "mass" in binned.columns
        assert "intensity" in binned.columns
        assert not binned["intensity"].isna().any()
        assert (binned["intensity"] >= 0).all()
        expected_bins = (20000 - 2000) // 3
        assert len(binned) == expected_bins

    def test_full_pipeline_with_quality(self):
        """Test: generate -> quality assess + preprocess -> bin -> valid output."""
        raw = _make_spectrum(seed=42)
        spec = MaldiSpectrum(raw)

        # Quality assessment on raw spectrum
        qc = SpectrumQuality()
        report = qc.assess(spec)
        assert report.snr > 0
        assert report.peak_count > 0
        assert report.total_ion_count > 0
        assert 0 <= report.baseline_fraction <= 1
        assert report.noise_level >= 0
        assert report.dynamic_range >= 0

        # Then preprocess and bin
        preprocessed = preprocess(raw)
        binned, _ = bin_spectrum(preprocessed, bin_width=3)
        assert len(binned) > 0
        assert not binned["intensity"].isna().any()

    def test_full_pipeline_with_merging(self):
        """Test: generate replicates -> outlier detect -> merge -> preprocess -> bin."""
        spectra = [MaldiSpectrum(_make_spectrum(seed=i)) for i in range(3)]

        # Outlier detection - all similar, so all should be kept
        keep_mask = detect_outlier_replicates(spectra)
        assert keep_mask.all()

        # Merge replicates
        merged = merge_replicates(spectra, method="mean")
        assert isinstance(merged, pd.DataFrame)
        assert "mass" in merged.columns
        assert "intensity" in merged.columns

        # Preprocess and bin the merged spectrum
        preprocessed = preprocess(merged)
        binned, _ = bin_spectrum(preprocessed, bin_width=3)
        assert len(binned) > 0
        assert not binned["intensity"].isna().any()
        assert (binned["intensity"] >= 0).all()

    def test_preprocess_bin_align(self):
        """Test: build binned matrix -> align with Warping -> valid output."""
        # Generate a small binned dataset (5 samples)
        rng = np.random.default_rng(42)
        n_samples, n_bins = 5, 100
        X = rng.exponential(0.01, (n_samples, n_bins))
        columns = [str(2000 + i * 3) for i in range(n_bins)]
        index = [f"sample_{i}" for i in range(n_samples)]
        df = pd.DataFrame(X, columns=columns, index=index)

        warper = Warping(method="shift", n_segments=3, max_shift=5)
        warper.fit(df)
        aligned = warper.transform(df)

        assert aligned.shape == df.shape
        assert not aligned.isna().any().any()

    def test_preprocess_bin_detect_peaks(self):
        """Test: generate -> preprocess -> bin -> detect peaks -> valid output."""
        raw = _make_spectrum(seed=42)
        preprocessed = preprocess(raw)
        binned, _ = bin_spectrum(preprocessed, bin_width=3)

        # Stack single spectrum into a 1-row DataFrame for peak detection
        row = binned.set_index("mass")["intensity"]
        matrix = pd.DataFrame([row.values], columns=row.index.astype(str))

        detector = MaldiPeakDetector(method="local", binary=True)
        detector.fit(matrix)
        peaks = detector.transform(matrix)

        assert peaks.shape == matrix.shape
        # Binary mode: values should be 0 or 1
        unique_vals = np.unique(peaks.values)
        assert all(v in (0.0, 1.0) for v in unique_vals)

    def test_build_dataset_end_to_end(self, tmp_path: Path):
        """Test: generate spectra -> build DRIAMS dataset -> validate structure."""
        # Create synthetic spectra files
        spectra_dir = tmp_path / "spectra"
        spectra_dir.mkdir()
        ids = []
        for i in range(3):
            raw = _make_spectrum(seed=42 + i)
            name = f"spec_{i}"
            ids.append(name)
            np.savetxt(
                spectra_dir / f"{name}.txt",
                raw[["mass", "intensity"]].values,
                header="mass intensity",
                comments="# ",
                fmt="%.6f",
            )

        # Create metadata with year column
        meta = pd.DataFrame(
            {
                "ID": ids,
                "Species": ["E. coli"] * 3,
                "Drug": ["S", "R", "S"],
                "acquisition_date": ["2015-01-10", "2016-03-20", "2015-06-15"],
            }
        )
        meta_path = tmp_path / "meta.csv"
        meta.to_csv(meta_path, index=False)

        # Build with year split and an extra handler
        out = tmp_path / "driams"
        report = DatasetBuilder(
            FlatLayout(
                spectra_dir,
                meta_path,
                year_column="acquisition_date",
            ),
            out,
            extra_handlers=[
                ProcessingHandler("binned_3000", "binned", bin_width=6),
            ],
            n_jobs=1,
        ).build()

        # Validate report
        assert report.total == 3
        assert report.succeeded == 3
        assert report.failed == 0

        # Validate directory structure
        assert (out / "raw" / "2015").is_dir()
        assert (out / "raw" / "2016").is_dir()
        assert (out / "preprocessed" / "2015").is_dir()
        assert (out / "binned_6000" / "2015").is_dir()
        assert (out / "binned_3000" / "2015").is_dir()

        # Validate file counts
        assert len(list((out / "raw" / "2015").glob("*.txt"))) == 2
        assert len(list((out / "raw" / "2016").glob("*.txt"))) == 1

        # Validate metadata split
        meta_2015 = pd.read_csv(out / "id" / "2015" / "2015_clean.csv")
        assert len(meta_2015) == 2
        assert "code" in meta_2015.columns

        # Validate binned output is readable and has correct format
        binned_file = list((out / "binned_6000" / "2015").glob("*.txt"))[0]
        content = binned_file.read_text()
        lines = content.strip().split("\n")
        assert lines[0] == "bin_index binned_intensity"
        assert 5990 <= len(lines) - 1 <= 6010
