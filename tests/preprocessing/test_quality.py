"""Unit tests for SpectrumQuality class."""

import numpy as np
import pandas as pd
import pytest

from maldiamrkit import MaldiSpectrum
from maldiamrkit.preprocessing import (
    SpectrumQuality,
    SpectrumQualityReport,
    estimate_snr,
)


@pytest.fixture
def synthetic_maldi(synthetic_spectrum: pd.DataFrame) -> MaldiSpectrum:
    """Wrap the synthetic spectrum DataFrame in a MaldiSpectrum."""
    return MaldiSpectrum(synthetic_spectrum)


class TestEstimateSNR:
    """Tests for estimate_snr function."""

    def test_estimate_snr_basic(self, synthetic_maldi):
        """Test basic SNR estimation."""
        snr = estimate_snr(synthetic_maldi)

        assert isinstance(snr, float)
        assert snr > 0

    def test_estimate_snr_custom_noise_region(self, synthetic_maldi):
        """Test SNR with custom noise region."""
        snr = estimate_snr(synthetic_maldi, noise_region=(3000, 4000))

        assert isinstance(snr, float)
        assert snr > 0

    def test_estimate_snr_empty_noise_region(self):
        """Test SNR returns inf for empty noise region."""
        df = pd.DataFrame({"mass": [10000, 11000, 12000], "intensity": [100, 200, 150]})
        snr = estimate_snr(MaldiSpectrum(df), noise_region=(2000, 3000))

        assert snr == np.inf

    def test_estimate_snr_median_peaks(self, synthetic_maldi):
        """Test SNR with median_peaks signal method."""
        snr = estimate_snr(synthetic_maldi, signal_method="median_peaks")

        assert isinstance(snr, float)
        assert snr > 0

    def test_estimate_snr_median_peaks_no_peaks(self):
        """Test median_peaks falls back to max when no peaks found."""
        # Constant spectrum has no peaks
        df = pd.DataFrame(
            {"mass": np.linspace(2000, 20000, 1000), "intensity": np.ones(1000)}
        )
        snr = estimate_snr(MaldiSpectrum(df), signal_method="median_peaks")
        assert isinstance(snr, float)

    def test_signal_methods_differ(self, synthetic_maldi):
        """Test that max and median_peaks give different results."""
        snr_max = estimate_snr(synthetic_maldi, signal_method="max")
        snr_median = estimate_snr(synthetic_maldi, signal_method="median_peaks")

        # max uses the single highest point; median_peaks uses top-N median
        # They should differ unless all peaks are the same height
        assert snr_max != snr_median

    def test_estimate_snr_invalid_method(self, synthetic_maldi):
        """Test that invalid signal_method raises ValueError."""
        with pytest.raises(ValueError, match="signal_method"):
            estimate_snr(synthetic_maldi, signal_method="invalid")


class TestSpectrumQuality:
    """Tests for SpectrumQuality class."""

    def test_default_init(self):
        """Test default initialization."""
        qc = SpectrumQuality()

        assert qc.noise_region == (19500, 20000)
        assert qc.peak_prominence == 1e-4
        assert qc.signal_method == "max"
        assert qc.n_top_peaks == 10

    def test_custom_init(self):
        """Test custom initialization."""
        qc = SpectrumQuality(noise_region=(3000, 4000), peak_prominence=1e-3)

        assert qc.noise_region == (3000, 4000)
        assert qc.peak_prominence == 1e-3

    def test_estimate_noise_level(self, synthetic_maldi):
        """Test noise level estimation."""
        qc = SpectrumQuality()
        noise = qc.estimate_noise_level(synthetic_maldi)

        assert isinstance(noise, float)
        assert noise >= 0

    def test_estimate_noise_level_empty_region(self):
        """Test noise level returns 0 for empty region."""
        spec = MaldiSpectrum(
            pd.DataFrame({"mass": [10000, 11000, 12000], "intensity": [100, 200, 150]})
        )
        qc = SpectrumQuality(noise_region=(2000, 3000))
        noise = qc.estimate_noise_level(spec)

        assert noise == 0.0

    def test_estimate_baseline_fraction(self, synthetic_maldi):
        """Test baseline fraction estimation."""
        qc = SpectrumQuality()
        baseline_frac = qc.estimate_baseline_fraction(synthetic_maldi)

        assert isinstance(baseline_frac, float)
        assert 0 <= baseline_frac <= 1

    def test_estimate_baseline_fraction_empty_region(self):
        """Test baseline fraction returns 0 for empty noise region."""
        spec = MaldiSpectrum(
            pd.DataFrame({"mass": [10000, 11000, 12000], "intensity": [100, 200, 150]})
        )
        qc = SpectrumQuality(noise_region=(2000, 3000))
        baseline_frac = qc.estimate_baseline_fraction(spec)

        assert baseline_frac == 0.0

    def test_estimate_dynamic_range(self, synthetic_maldi):
        """Test dynamic range estimation."""
        qc = SpectrumQuality()
        dr = qc.estimate_dynamic_range(synthetic_maldi)

        assert isinstance(dr, float)
        assert dr >= 0

    def test_estimate_dynamic_range_constant_signal(self):
        """Test dynamic range with constant signal."""
        spec = MaldiSpectrum(
            pd.DataFrame(
                {
                    "mass": np.linspace(2000, 20000, 1000),
                    "intensity": np.ones(1000) * 100,
                }
            )
        )
        qc = SpectrumQuality()
        dr = qc.estimate_dynamic_range(spec)

        # Constant signal should have 0 dynamic range
        assert dr == 0.0

    def test_count_peaks(self, synthetic_maldi):
        """Test peak counting."""
        qc = SpectrumQuality()
        count = qc.count_peaks(synthetic_maldi)

        assert isinstance(count, int)
        assert count >= 0

    def test_assess_returns_report(self, synthetic_maldi):
        """Test that assess returns a SpectrumQualityReport."""
        qc = SpectrumQuality()
        report = qc.assess(synthetic_maldi)

        assert isinstance(report, SpectrumQualityReport)

    def test_assess_report_fields(self, synthetic_maldi):
        """Test that report contains all expected fields."""
        qc = SpectrumQuality()
        report = qc.assess(synthetic_maldi)

        assert hasattr(report, "snr")
        assert hasattr(report, "total_ion_count")
        assert hasattr(report, "peak_count")
        assert hasattr(report, "baseline_fraction")
        assert hasattr(report, "noise_level")
        assert hasattr(report, "dynamic_range")

    def test_assess_report_values(self, synthetic_maldi):
        """Test that report values are valid."""
        qc = SpectrumQuality()
        report = qc.assess(synthetic_maldi)

        assert report.snr > 0
        assert report.total_ion_count > 0
        assert report.peak_count >= 0
        assert 0 <= report.baseline_fraction <= 1
        assert report.noise_level >= 0
        assert report.dynamic_range >= 0


class TestSpectrumQualityReport:
    """Tests for SpectrumQualityReport dataclass."""

    def test_dataclass_creation(self):
        """Test that dataclass can be created with all fields."""
        report = SpectrumQualityReport(
            snr=100.0,
            total_ion_count=1e6,
            peak_count=50,
            baseline_fraction=0.3,
            noise_level=10.0,
            dynamic_range=2.5,
        )

        assert report.snr == 100.0
        assert report.total_ion_count == 1e6
        assert report.peak_count == 50
        assert report.baseline_fraction == 0.3
        assert report.noise_level == 10.0
        assert report.dynamic_range == 2.5


class TestSpectrumQualityReproducibility:
    """Tests for reproducibility of quality metrics."""

    def test_same_input_same_output(self, synthetic_maldi):
        """Test that same input produces same output."""
        qc = SpectrumQuality()

        report1 = qc.assess(synthetic_maldi)
        report2 = qc.assess(synthetic_maldi)

        assert report1.snr == report2.snr
        assert report1.total_ion_count == report2.total_ion_count
        assert report1.peak_count == report2.peak_count
        assert report1.baseline_fraction == report2.baseline_fraction
        assert report1.noise_level == report2.noise_level
        assert report1.dynamic_range == report2.dynamic_range
