"""End-to-end integration tests for the full MALDI-TOF preprocessing pipeline."""

import numpy as np
import pandas as pd
import pytest

from maldiamrkit import MaldiSpectrum
from maldiamrkit.alignment import Warping
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
        """Test: generate → preprocess → bin → valid output."""
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
        """Test: generate → quality assess + preprocess → bin → valid output."""
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
        """Test: generate replicates → outlier detect → merge → preprocess → bin."""
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
        """Test: build binned matrix → align with Warping → valid output."""
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
        """Test: generate → preprocess → bin → detect peaks → valid output."""
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
