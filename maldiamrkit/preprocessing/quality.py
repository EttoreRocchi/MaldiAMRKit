"""Quality metrics for MALDI-TOF spectra."""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from scipy.signal import find_peaks

if TYPE_CHECKING:
    from maldiamrkit.spectrum import MaldiSpectrum


def _get_spectrum_df(spectrum: MaldiSpectrum) -> pd.DataFrame:
    """Return preprocessed data if available, otherwise raw."""
    if spectrum._preprocessed is not None:
        return spectrum.preprocessed
    return spectrum.raw


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
    signal_method : str, default="max"
        How to estimate the signal level for SNR calculation:

        - ``"max"``: use the maximum intensity (standard, but
          sensitive to single outlier peaks).
        - ``"median_peaks"``: use the median intensity of the top
          *n_top_peaks* detected peaks (more robust).
    n_top_peaks : int, default=10
        Number of top peaks to consider when
        ``signal_method="median_peaks"``.

    Examples
    --------
    >>> from maldiamrkit import MaldiSpectrum
    >>> from maldiamrkit.preprocessing.quality import SpectrumQuality
    >>> spec = MaldiSpectrum("spectrum.txt").preprocess()
    >>> qc = SpectrumQuality(noise_region=(19500, 20000))
    >>> report = qc.assess(spec)
    >>> print(f"SNR: {report.snr:.1f}")
    >>> print(f"TIC: {report.total_ion_count:.2e}")
    >>> print(f"Peaks: {report.peak_count}")
    """

    def __init__(
        self,
        noise_region: tuple[float, float] = (19500, 20000),
        peak_prominence: float = 1e-4,
        signal_method: str = "max",
        n_top_peaks: int = 10,
    ):
        self.noise_region = noise_region
        self.peak_prominence = peak_prominence
        self.signal_method = signal_method
        self.n_top_peaks = n_top_peaks

    def estimate_noise_level(self, spectrum: MaldiSpectrum) -> float:
        """
        Estimate noise level using MAD in noise region.

        Parameters
        ----------
        spectrum : MaldiSpectrum
            Spectrum to assess. Uses preprocessed data if available,
            otherwise raw.

        Returns
        -------
        float
            Estimated noise standard deviation. Returns 0 if noise region
            is empty.
        """
        df = _get_spectrum_df(spectrum)
        noise_mask = df["mass"].between(*self.noise_region)
        noise = df.loc[noise_mask, "intensity"]

        if len(noise) == 0:
            return 0.0

        mad = np.median(np.abs(noise - np.median(noise)))
        return 1.4826 * mad  # MAD to std conversion

    def estimate_baseline_fraction(self, spectrum: MaldiSpectrum) -> float:
        """
        Estimate fraction of intensity below noise floor.

        This indicates how much of the spectrum is dominated by baseline
        rather than signal. High values suggest poor acquisition quality
        or excessive baseline.

        Parameters
        ----------
        spectrum : MaldiSpectrum
            Spectrum to assess. Uses preprocessed data if available,
            otherwise raw.

        Returns
        -------
        float
            Fraction of data points below 2x noise level (0 to 1).
        """
        df = _get_spectrum_df(spectrum)
        noise_level = self.estimate_noise_level(spectrum)
        if noise_level == 0:
            return 0.0

        baseline_threshold = 2 * noise_level
        baseline_points = (df["intensity"] < baseline_threshold).sum()
        return baseline_points / len(df)

    def estimate_dynamic_range(self, spectrum: MaldiSpectrum) -> float:
        """
        Estimate dynamic range as log10 ratio of max to median signal.

        Higher values indicate better separation between signal and
        background.

        Parameters
        ----------
        spectrum : MaldiSpectrum
            Spectrum to assess. Uses preprocessed data if available,
            otherwise raw.

        Returns
        -------
        float
            Log10 ratio of max to median intensity. Returns 0 if
            median is zero.
        """
        df = _get_spectrum_df(spectrum)
        # Exclude very low values (likely noise/baseline)
        signal_mask = df["intensity"] > df["intensity"].quantile(0.1)

        if signal_mask.sum() == 0:
            return 0.0

        max_signal = df.loc[signal_mask, "intensity"].max()
        median_signal = df.loc[signal_mask, "intensity"].median()

        if median_signal <= 0:
            return 0.0

        return np.log10(max_signal / median_signal)

    def count_peaks(self, spectrum: MaldiSpectrum) -> int:
        """
        Count the number of peaks in the spectrum.

        Parameters
        ----------
        spectrum : MaldiSpectrum
            Spectrum to assess. Uses preprocessed data if available,
            otherwise raw.

        Returns
        -------
        int
            Number of detected peaks.
        """
        df = _get_spectrum_df(spectrum)
        noise_level = self.estimate_noise_level(spectrum)
        min_prominence = max(self.peak_prominence, noise_level * 3)

        peaks, _ = find_peaks(df["intensity"].values, prominence=min_prominence)
        return len(peaks)

    def assess(self, spectrum: MaldiSpectrum) -> SpectrumQualityReport:
        """
        Perform full quality assessment of a spectrum.

        Parameters
        ----------
        spectrum : MaldiSpectrum
            Spectrum to assess. Uses preprocessed data if available,
            otherwise raw.

        Returns
        -------
        SpectrumQualityReport
            Dataclass containing all quality metrics.
        """
        df = _get_spectrum_df(spectrum)
        mz_max = df["mass"].max()
        if self.noise_region[0] > mz_max:
            warnings.warn(
                f"noise_region {self.noise_region} is outside spectrum range "
                f"(max m/z={mz_max:.0f}). Quality metrics will be unreliable. "
                f"Adjust noise_region to fall within the trimmed spectrum range.",
                UserWarning,
                stacklevel=2,
            )

        noise_level = self.estimate_noise_level(spectrum)
        snr = estimate_snr(
            spectrum,
            self.noise_region,
            signal_method=self.signal_method,
            n_top_peaks=self.n_top_peaks,
        )

        return SpectrumQualityReport(
            snr=snr,
            total_ion_count=df["intensity"].sum(),
            peak_count=self.count_peaks(spectrum),
            baseline_fraction=self.estimate_baseline_fraction(spectrum),
            noise_level=noise_level,
            dynamic_range=self.estimate_dynamic_range(spectrum),
        )


def estimate_snr(
    spectrum: MaldiSpectrum,
    noise_region: tuple[float, float] = (19500, 20000),
    signal_method: str = "max",
    n_top_peaks: int = 10,
) -> float:
    """
    Estimate signal-to-noise ratio of a spectrum.

    Uses median absolute deviation (MAD) in a noise region to estimate
    noise level.  The signal level is determined by *signal_method*.

    Parameters
    ----------
    spectrum : MaldiSpectrum
        Spectrum to assess. Uses preprocessed data if available,
        otherwise raw.
    noise_region : tuple of (float, float), default=(19500, 20000)
        m/z range to use for noise estimation. Should be a region
        with minimal peaks (typically high m/z range).
    signal_method : str, default="max"
        How to estimate the signal level:

        - ``"max"``: maximum intensity (standard approach).
        - ``"median_peaks"``: median intensity of the top
          *n_top_peaks* detected peaks.  More robust to single
          outlier peaks.
    n_top_peaks : int, default=10
        Number of top peaks to consider when
        ``signal_method="median_peaks"``.

    Returns
    -------
    float
        Estimated signal-to-noise ratio. Returns inf if noise is zero.

    Raises
    ------
    ValueError
        If ``signal_method`` is not one of 'max' or 'median_peaks'.

    Notes
    -----
    The MAD-to-standard-deviation conversion factor (1.4826) assumes
    normally distributed noise.

    Examples
    --------
    >>> from maldiamrkit import MaldiSpectrum
    >>> from maldiamrkit.preprocessing import estimate_snr
    >>> spec = MaldiSpectrum("spectrum.txt").preprocess()
    >>> snr = estimate_snr(spec)
    >>> print(f"SNR: {snr:.1f}")
    >>> snr_robust = estimate_snr(spec, signal_method="median_peaks")
    """
    valid_methods = ("max", "median_peaks")
    if signal_method not in valid_methods:
        raise ValueError(
            f"signal_method must be one of {valid_methods}, got {signal_method!r}."
        )

    df = _get_spectrum_df(spectrum)

    noise_mask = df["mass"].between(*noise_region)
    noise = df.loc[noise_mask, "intensity"]

    if len(noise) == 0:
        return np.inf

    mad = np.median(np.abs(noise - np.median(noise)))
    noise_std = 1.4826 * mad  # MAD to std conversion for normal distribution

    if signal_method == "max":
        signal = df["intensity"].max()
    else:  # median_peaks
        peaks, _ = find_peaks(df["intensity"].values)
        if len(peaks) == 0:
            signal = df["intensity"].max()
        else:
            peak_intensities = df["intensity"].values[peaks]
            top_n = np.sort(peak_intensities)[-n_top_peaks:]
            signal = np.median(top_n)

    return signal / noise_std if noise_std > 0 else np.inf
