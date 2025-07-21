import numpy as np
import pandas as pd
from pybaselines import Baseline
from scipy.signal import savgol_filter

from .config import PreprocessingSettings


def preprocess(
        df: pd.DataFrame,
        cfg: PreprocessingSettings = PreprocessingSettings()
    ) -> pd.DataFrame:
    """Return intensity-normalised, baseline-corrected and trimmed spectrum."""
    df = df.copy()
    df["intensity"] = df["intensity"].clip(lower=0)

    # smooth+sqrt
    intensity = np.sqrt(df["intensity"])
    intensity = savgol_filter(intensity,
                              window_length=cfg.savgol_window,
                              polyorder=cfg.savgol_poly)

    # baseline correction
    bkg = Baseline(x_data=df["mass"]).snip(
        intensity,
        max_half_window=cfg.baseline_half_window,
        decreasing=True,
        smooth_half_window=0
    )[0]
    intensity -= bkg
    intensity[intensity < 0] = 0  # remove any small negative values post-baseline

    out = pd.DataFrame({"mass": df["mass"], "intensity": intensity})

    mmin, mmax = cfg.trim_from, cfg.trim_to
    out = out[(out.mass.between(mmin, mmax))].reset_index(drop=True)

    total = out["intensity"].sum()
    if total > 0:
        out["intensity"] /= total
    return out


def bin_spectrum(
        df: pd.DataFrame,
        cfg: PreprocessingSettings,
        bin_width: int | float | None = None,
    ) -> pd.DataFrame:
    """
    Bin intensities using *inclusive left* intervals
    [start, start+bin_width). Returns DataFrame («mass», «intensity»).
    """
    if bin_width is None:
        raise ValueError(" 'bin_width=None': no binning requested.")

    edges = np.arange(cfg.trim_from, cfg.trim_to + bin_width, bin_width)
    labels = edges[:-1].astype(str)
    binned = (
        df
        .assign(bins=pd.cut(df.mass, edges, labels=labels, include_lowest=True))
        .groupby("bins", observed=True)["intensity"]
        .sum()
        .reindex(labels, fill_value=0.0)
        .reset_index()
        .rename(columns={"bins": "mass"})
    )
    return binned
