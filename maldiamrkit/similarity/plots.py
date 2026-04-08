"""Visualizations for spectral similarity analysis."""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import matplotlib.pyplot as plt


def plot_distance_heatmap(
    distance_matrix: np.ndarray,
    labels: list[str] | np.ndarray | None = None,
    *,
    cmap: str = "viridis",
    ax: plt.Axes | None = None,
    title: str | None = None,
    figsize: tuple[float, float] = (8, 8),
    show: bool = True,
) -> tuple[plt.Figure, plt.Axes]:
    """Plot a pairwise distance matrix as a heatmap.

    Parameters
    ----------
    distance_matrix : ndarray of shape (n, n)
        Symmetric distance matrix.
    labels : list of str, ndarray, or None, default=None
        Tick labels for rows and columns.
    cmap : str, default="viridis"
        Matplotlib / seaborn colormap name.
    ax : Axes or None, default=None
        Pre-existing axes for the plot.  If ``None``, a new figure and
        axes are created.
    title : str or None, default=None
        Plot title.
    figsize : tuple of float, default=(8, 8)
        Figure size in inches (used only when *ax* is ``None``).
    show : bool, default=True
        Whether to call :func:`matplotlib.pyplot.show`.

    Returns
    -------
    tuple[Figure, Axes]
    """
    import matplotlib
    import matplotlib.pyplot as plt
    import seaborn as sns

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()

    tick_labels = labels if labels is not None else False
    sns.heatmap(
        distance_matrix,
        ax=ax,
        cmap=cmap,
        xticklabels=tick_labels,
        yticklabels=tick_labels,
        square=True,
    )

    if title is not None:
        ax.set_title(title)

    if show:
        if not matplotlib.is_interactive():
            warnings.warn(
                "matplotlib is using a non-interactive backend; "
                "plt.show() may not display the figure",
                UserWarning,
                stacklevel=2,
            )
        plt.show()

    return fig, ax


def plot_dendrogram(
    linkage_matrix: np.ndarray,
    labels: list[str] | None = None,
    *,
    ax: plt.Axes | None = None,
    title: str | None = None,
    figsize: tuple[float, float] = (10, 6),
    show: bool = True,
) -> tuple[plt.Figure, plt.Axes]:
    """Plot a dendrogram from a hierarchical clustering linkage matrix.

    Parameters
    ----------
    linkage_matrix : ndarray of shape (n - 1, 4)
        Linkage matrix from
        :func:`~maldiamrkit.similarity.clustering.hierarchical_clustering`.
    labels : list of str or None, default=None
        Leaf labels.
    ax : Axes or None, default=None
        Pre-existing axes for the plot.
    title : str or None, default=None
        Plot title.
    figsize : tuple of float, default=(10, 6)
        Figure size in inches (used only when *ax* is ``None``).
    show : bool, default=True
        Whether to call :func:`matplotlib.pyplot.show`.

    Returns
    -------
    tuple[Figure, Axes]
    """
    import matplotlib
    import matplotlib.pyplot as plt
    from scipy.cluster.hierarchy import dendrogram

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()

    dendrogram(linkage_matrix, labels=labels, ax=ax, leaf_rotation=90)

    if title is not None:
        ax.set_title(title)

    if show:
        if not matplotlib.is_interactive():
            warnings.warn(
                "matplotlib is using a non-interactive backend; "
                "plt.show() may not display the figure",
                UserWarning,
                stacklevel=2,
            )
        plt.show()

    return fig, ax
