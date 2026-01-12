"""Shared plotting utilities."""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def plot_spectrum(
    df: pd.DataFrame,
    ax: plt.Axes | None = None,
    title: str | None = None,
    color: str = "black",
    alpha: float = 0.8,
    **kwargs
) -> plt.Axes:
    """
    Plot a single spectrum.

    Parameters
    ----------
    df : pd.DataFrame
        Spectrum with columns 'mass' and 'intensity'.
    ax : matplotlib.axes.Axes, optional
        Axes to plot on. Creates new figure if None.
    title : str, optional
        Plot title.
    color : str, default="black"
        Line color.
    alpha : float, default=0.8
        Line transparency.
    **kwargs : dict
        Additional arguments passed to ax.plot().

    Returns
    -------
    matplotlib.axes.Axes
        The axes containing the plot.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(10, 4))

    ax.plot(df['mass'], df['intensity'], color=color, alpha=alpha, **kwargs)
    ax.set_xlabel('m/z')
    ax.set_ylabel('Intensity')
    if title:
        ax.set_title(title)
    ax.set_ylim(bottom=0)

    return ax


def plot_spectra_comparison(
    spectra: list[pd.DataFrame],
    labels: list[str] | None = None,
    colors: list[str] | None = None,
    ax: plt.Axes | None = None,
    title: str | None = None,
    alpha: float = 0.7,
    **kwargs
) -> plt.Axes:
    """
    Plot multiple spectra for comparison.

    Parameters
    ----------
    spectra : list of pd.DataFrame
        List of spectra with columns 'mass' and 'intensity'.
    labels : list of str, optional
        Labels for each spectrum.
    colors : list of str, optional
        Colors for each spectrum.
    ax : matplotlib.axes.Axes, optional
        Axes to plot on.
    title : str, optional
        Plot title.
    alpha : float, default=0.7
        Line transparency.
    **kwargs : dict
        Additional arguments passed to ax.plot().

    Returns
    -------
    matplotlib.axes.Axes
        The axes containing the plot.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(12, 5))

    if colors is None:
        colors = plt.cm.tab10(np.linspace(0, 1, len(spectra)))

    if labels is None:
        labels = [f'Spectrum {i}' for i in range(len(spectra))]

    for df, label, color in zip(spectra, labels, colors):
        ax.plot(
            df['mass'], df['intensity'],
            label=label, color=color, alpha=alpha, **kwargs
        )

    ax.set_xlabel('m/z')
    ax.set_ylabel('Intensity')
    ax.legend()
    if title:
        ax.set_title(title)
    ax.set_ylim(bottom=0)

    return ax
