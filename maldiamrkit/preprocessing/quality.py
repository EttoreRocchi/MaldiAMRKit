"""Quality metrics for MALDI-TOF spectra."""
import numpy as np
import pandas as pd


def estimate_snr(
    df: pd.DataFrame,
    noise_region: tuple[float, float] = (2000, 2500)
) -> float:
    """
    Estimate signal-to-noise ratio of a spectrum.

    Uses median absolute deviation (MAD) in a noise region to estimate
    noise level, and the maximum intensity as the signal level.

    Parameters
    ----------
    df : pd.DataFrame
        Spectrum with columns 'mass' and 'intensity'.
    noise_region : tuple of (float, float), default=(2000, 2500)
        m/z range to use for noise estimation. Should be a region
        with minimal peaks (typically low m/z range).

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
