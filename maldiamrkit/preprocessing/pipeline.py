"""Main preprocessing pipeline for MALDI-TOF spectra."""

from __future__ import annotations

import pandas as pd

from .preprocessing_pipeline import PreprocessingPipeline


def preprocess(
    df: pd.DataFrame,
    pipeline: PreprocessingPipeline | None = None,
) -> pd.DataFrame:
    """
    Apply preprocessing pipeline to a raw MALDI-TOF spectrum.

    By default applies: clip negatives → sqrt transform → Savitzky-Golay
    smoothing → SNIP baseline → m/z trim (2000–20000 Da) → TIC normalization.

    Parameters
    ----------
    df : pd.DataFrame
        Raw spectrum with columns 'mass' and 'intensity'.
    pipeline : PreprocessingPipeline, optional
        Custom pipeline. If None, uses ``PreprocessingPipeline.default()``.

    Returns
    -------
    pd.DataFrame
        Preprocessed spectrum with columns 'mass' and 'intensity'.

    See Also
    --------
    PreprocessingPipeline : Composable preprocessing pipeline class.
    bin_spectrum : Bin preprocessed spectrum into m/z bins.

    Examples
    --------
    >>> from maldiamrkit.preprocessing import preprocess, PreprocessingPipeline
    >>> preprocessed = preprocess(raw_df)
    >>> preprocessed = preprocess(raw_df, PreprocessingPipeline.default())
    """
    if pipeline is None:
        pipeline = PreprocessingPipeline.default()
    return pipeline(df)
