"""Quality metrics for MALDI-TOF spectra."""
from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.signal import find_peaks


@dataclass
class SpectrumQualityReport:
    """
    Quality metrics report for a single MALDI-TOF spectrum.

    Attributes
    ----------
    snr : float
        Signal-to-noise ratio.
    total_ion_count : float
        Sum of all intensities (total ion count).
    peak_count : int
        Number of detected peaks.
    baseline_fraction : float
        Fraction of data points below noise floor (baseline contamination).
    noise_level : float
        Estimated noise level (standard deviation).
    dynamic_range : float
        Log10 ratio of max to median signal intensity.
    """
    snr: float
    total_ion_count: float
    peak_count: int
    baseline_fraction: float
    noise_level: float
    dynamic_range: float


class SpectrumQuality:
    """
    Comprehensive quality assessment for MALDI-TOF spectra.

    Provides methods to compute various quality metrics for individual
    spectra, useful for quality control and filtering poor-quality
    acquisitions.

    Parameters
    ----------
    noise_region : tuple of (float, float), default=(19500, 20000)
        m/z range to use for noise estimation. Should be a region
        with minimal peaks (typically high m/z range).
    peak_prominence : float, default=1e-4
        Minimum prominence for peak detection.

    Examples
    --------
    >>> from maldiamrkit.preprocessing.quality import SpectrumQuality
    >>> qc = SpectrumQuality(noise_region=(19500, 20000))
    >>> report = qc.assess(spectrum_df)
    >>> print(f"SNR: {report.snr:.1f}")
    >>> print(f"TIC: {report.total_ion_count:.2e}")
    >>> print(f"Peaks: {report.peak_count}")
    """

    def __init__(
        self,
        noise_region: tuple[float, float] = (19500, 20000),
        peak_prominence: float = 1e-4,
    ):
        self.noise_region = noise_region
        self.peak_prominence = peak_prominence

    def estimate_noise_level(self, df: pd.DataFrame) -> float:
        """
        Estimate noise level using MAD in noise region.

        Parameters
        ----------
        df : pd.DataFrame
            Spectrum with columns 'mass' and 'intensity'.

        Returns
        -------
        float
            Estimated noise standard deviation. Returns 0 if noise region
            is empty.
        """
        noise_mask = df['mass'].between(*self.noise_region)
        noise = df.loc[noise_mask, 'intensity']

        if len(noise) == 0:
            return 0.0

        mad = np.median(np.abs(noise - np.median(noise)))
        return 1.4826 * mad  # MAD to std conversion

    def estimate_baseline_fraction(self, df: pd.DataFrame) -> float:
        """
        Estimate fraction of intensity below noise floor.

        This indicates how much of the spectrum is dominated by baseline
        rather than signal. High values suggest poor acquisition quality
        or excessive baseline.

        Parameters
        ----------
        df : pd.DataFrame
            Spectrum with columns 'mass' and 'intensity'.

        Returns
        -------
        float
            Fraction of data points below 2x noise level (0 to 1).
        """
        noise_level = self.estimate_noise_level(df)
        if noise_level == 0:
            return 0.0

        baseline_threshold = 2 * noise_level
        baseline_points = (df['intensity'] < baseline_threshold).sum()
        return baseline_points / len(df)

    def estimate_dynamic_range(self, df: pd.DataFrame) -> float:
        """
        Estimate dynamic range as log10 ratio of max to median signal.

        Higher values indicate better separation between signal and
        background.

        Parameters
        ----------
        df : pd.DataFrame
            Spectrum with columns 'mass' and 'intensity'.

        Returns
        -------
        float
            Log10 ratio of max to median intensity. Returns 0 if
            median is zero.
        """
        # Exclude very low values (likely noise/baseline)
        signal_mask = df['intensity'] > df['intensity'].quantile(0.1)

        if signal_mask.sum() == 0:
            return 0.0

        max_signal = df.loc[signal_mask, 'intensity'].max()
        median_signal = df.loc[signal_mask, 'intensity'].median()

        if median_signal <= 0:
            return 0.0

        return np.log10(max_signal / median_signal)

    def count_peaks(self, df: pd.DataFrame) -> int:
        """
        Count the number of peaks in the spectrum.

        Parameters
        ----------
        df : pd.DataFrame
            Spectrum with columns 'mass' and 'intensity'.

        Returns
        -------
        int
            Number of detected peaks.
        """
        noise_level = self.estimate_noise_level(df)
        min_prominence = max(self.peak_prominence, noise_level * 3)

        peaks, _ = find_peaks(df['intensity'].values, prominence=min_prominence)
        return len(peaks)

    def assess(self, df: pd.DataFrame) -> SpectrumQualityReport:
        """
        Perform full quality assessment of a spectrum.

        Parameters
        ----------
        df : pd.DataFrame
            Spectrum with columns 'mass' and 'intensity'.

        Returns
        -------
        SpectrumQualityReport
            Dataclass containing all quality metrics.
        """
        noise_level = self.estimate_noise_level(df)
        snr = estimate_snr(df, self.noise_region)

        return SpectrumQualityReport(
            snr=snr,
            total_ion_count=df['intensity'].sum(),
            peak_count=self.count_peaks(df),
            baseline_fraction=self.estimate_baseline_fraction(df),
            noise_level=noise_level,
            dynamic_range=self.estimate_dynamic_range(df),
        )


def estimate_snr(
    df: pd.DataFrame,
    noise_region: tuple[float, float] = (19500, 20000)
) -> float:
    """
    Estimate signal-to-noise ratio of a spectrum.

    Uses median absolute deviation (MAD) in a noise region to estimate
    noise level, and the maximum intensity as the signal level.

    Parameters
    ----------
    df : pd.DataFrame
        Spectrum with columns 'mass' and 'intensity'.
    noise_region : tuple of (float, float), default=(19500, 20000)
        m/z range to use for noise estimation. Should be a region
        with minimal peaks (typically high m/z range).

    Returns
    -------
    float
        Estimated signal-to-noise ratio. Returns inf if noise is zero.

    Notes
    -----
    The MAD-to-standard-deviation conversion factor (1.4826) assumes
    normally distributed noise.

    Examples
    --------
    >>> from maldiamrkit.preprocessing import estimate_snr
    >>> snr = estimate_snr(spectrum_df)
    >>> print(f"SNR: {snr:.1f}")
    """
    noise_mask = df['mass'].between(*noise_region)
    noise = df.loc[noise_mask, 'intensity']

    if len(noise) == 0:
        return np.inf

    mad = np.median(np.abs(noise - np.median(noise)))
    noise_std = 1.4826 * mad  # MAD to std conversion for normal distribution

    signal = df['intensity'].max()

    return signal / noise_std if noise_std > 0 else np.inf
