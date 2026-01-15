"""Configuration dataclasses for preprocessing settings."""

from dataclasses import dataclass


@dataclass()
class PreprocessingSettings:
    """
    Configuration settings for MALDI-TOF spectrum preprocessing.

    Parameters
    ----------
    trim_from : int, default=2000
        Lower m/z bound for spectrum trimming (in Daltons).
    trim_to : int, default=20000
        Upper m/z bound for spectrum trimming (in Daltons).
    savgol_window : int, default=20
        Window length for Savitzky-Golay smoothing filter.
    savgol_poly : int, default=2
        Polynomial order for Savitzky-Golay smoothing filter.
    baseline_half_window : int, default=40
        Half-window size for SNIP baseline correction algorithm.
    """

    trim_from: int = 2_000
    trim_to: int = 20_000

    savgol_window: int = 20
    savgol_poly: int = 2
    baseline_half_window: int = 40

    def as_dict(self) -> dict:
        """Return settings as a dictionary."""
        return self.__dict__.copy()
