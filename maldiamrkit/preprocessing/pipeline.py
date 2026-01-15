"""Main preprocessing pipeline for MALDI-TOF spectra."""

import numpy as np
import pandas as pd
from pybaselines import Baseline
from scipy.signal import savgol_filter

from ..core.config import PreprocessingSettings


def preprocess(
    df: pd.DataFrame, cfg: PreprocessingSettings = PreprocessingSettings()
) -> pd.DataFrame:
    """
    Apply preprocessing pipeline to a raw MALDI-TOF spectrum.

    Performs the following steps:
    1. Clip negative intensities to zero
    2. Square root transformation (variance stabilization)
    3. Savitzky-Golay smoothing
    4. SNIP baseline correction
    5. Trim to specified m/z range
    6. Total intensity normalization

    Parameters
    ----------
    df : pd.DataFrame
        Raw spectrum with columns 'mass' and 'intensity'.
    cfg : PreprocessingSettings, default=PreprocessingSettings()
        Configuration object with preprocessing parameters.

    Returns
    -------
    pd.DataFrame
        Preprocessed spectrum with columns 'mass' and 'intensity'.
        Intensities are normalized to sum to 1.

    See Also
    --------
    bin_spectrum : Bin preprocessed spectrum into m/z bins.
    estimate_snr : Estimate signal-to-noise ratio.

    Examples
    --------
    >>> from maldiamrkit.preprocessing import preprocess
    >>> from maldiamrkit.core.config import PreprocessingSettings
    >>> preprocessed = preprocess(raw_df, PreprocessingSettings())
    """
    df = df.copy()
    df["intensity"] = df["intensity"].clip(lower=0)

    # Sqrt transform + smoothing
    intensity = np.sqrt(df["intensity"])
    intensity = savgol_filter(
        intensity, window_length=cfg.savgol_window, polyorder=cfg.savgol_poly
    )

    # Baseline correction using SNIP algorithm
    bkg = Baseline(x_data=df["mass"]).snip(
        intensity,
        max_half_window=cfg.baseline_half_window,
        decreasing=True,
        smooth_half_window=0,
    )[0]
    intensity -= bkg
    intensity[intensity < 0] = 0  # Remove any small negative values post-baseline

    out = pd.DataFrame({"mass": df["mass"], "intensity": intensity})

    # Trim to m/z range
    mmin, mmax = cfg.trim_from, cfg.trim_to
    out = out[(out.mass.between(mmin, mmax))].reset_index(drop=True)

    # Normalize to total intensity
    total = out["intensity"].sum()
    if total > 0:
        out["intensity"] /= total

    return out
