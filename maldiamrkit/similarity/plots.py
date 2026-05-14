"""Visualizations for spectral similarity analysis."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from ..visualization._common import show_with_warning

if TYPE_CHECKING:
    import matplotlib.pyplot as plt


# Known theoretical upper bounds for spectral-distance metrics.  When
# the caller passes `metric=...`, the colorbar is clamped to these
# bounds (unless explicit ``vmin``/``vmax`` are given) so the colour
# scale becomes comparable across heatmaps computed with the same
# metric.  Metrics without a finite upper bound (wasserstein, dtw) are
# omitted.
_METRIC_BOUNDS: dict[str, tuple[float, float]] = {
    "cosine": (0.0, 1.0),
    "pearson": (0.0, 2.0),
    "spectral_contrast_angle": (0.0, float(np.pi / 2)),
}


def _dynamic_heatmap_figsize(n: int) -> tuple[float, float]:
    r"""Scale the figure side with matrix size, capped at 16\"."""
    side = min(16.0, 4.0 + 0.1 * n)
    return (side, side)


def plot_distance_heatmap(
    distance_matrix: np.ndarray,
    labels: list[str] | np.ndarray | None = None,
    *,
    metric: str | None = None,
    cmap: str = "viridis",
    ax: plt.Axes | None = None,
    title: str | None = None,
    figsize: tuple[float, float] | None = None,
    annot: bool | None = None,
    cluster: bool = False,
    vmin: float | None = None,
    vmax: float | None = None,
    cbar_label: str = "distance",
    show: bool = True,
) -> tuple[plt.Figure, plt.Axes]:
    """Plot a pairwise distance matrix as a heatmap.

    Parameters
    ----------
    distance_matrix : ndarray of shape (n, n)
        Symmetric distance matrix.
    labels : list of str, ndarray, or None, default=None
        Tick labels for rows and columns.
    metric : str, optional
        Name of the distance metric used (e.g. ``"cosine"``, ``"pearson"``,
        ``"spectral_contrast_angle"``).  When given and recognised, the
        colourbar limits are clamped to the metric's theoretical bounds
        so heatmaps computed with the same metric share a comparable
        colour scale.  Explicit ``vmin``/``vmax`` always win.
    cmap : str, default="viridis"
        Matplotlib / seaborn colormap name.
    ax : Axes or None, default=None
        Pre-existing axes.  If ``None``, a new figure and axes are created.
    title : str or None, default=None
        Plot title.  Defaults to ``"Pairwise distance"`` (including the
        metric name, if provided).
    figsize : tuple of float, optional
        Figure size in inches.  When ``None``, scales with ``n``
        (``side = min(16, 4 + 0.1 * n)``).  Only used when ``ax`` is ``None``.
    annot : bool, optional
        When ``True``, annotate each cell with its distance value.
        When ``None`` (default), annotate iff the matrix is small
        (``n ≤ 15``).
    cluster : bool, default=False
        When ``True``, reorder rows and columns via hierarchical
        clustering so similar samples group visually.  Labels reorder
        accordingly.
    vmin, vmax : float, optional
        Explicit colourbar limits.  Override any metric-derived bounds.
    cbar_label : str, default="distance"
        Label drawn on the colourbar.
    show : bool, default=True
        Call ``plt.show()`` at the end.

    Returns
    -------
    fig : matplotlib.figure.Figure
    ax : matplotlib.axes.Axes
    """
    import matplotlib.pyplot as plt
    import seaborn as sns

    D = np.asarray(distance_matrix, dtype=float)
    n = D.shape[0]

    # Clustering reorder (before figsize / annot decisions).
    if cluster:
        from scipy.cluster.hierarchy import leaves_list

        from .clustering import hierarchical_clustering

        linkage = hierarchical_clustering(D, method="average")
        order = leaves_list(linkage)
        D = D[np.ix_(order, order)]
        if labels is not None:
            labels = np.asarray(labels)[order]

    # Metric-derived colourbar bounds; user-supplied vmin/vmax win.
    if metric is not None and metric in _METRIC_BOUNDS:
        lo, hi = _METRIC_BOUNDS[metric]
        if vmin is None:
            vmin = lo
        if vmax is None:
            vmax = hi

    if annot is None:
        annot = n <= 15

    if ax is None:
        if figsize is None:
            figsize = _dynamic_heatmap_figsize(n)
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()

    tick_labels = labels if labels is not None else False

    sns.heatmap(
        D,
        ax=ax,
        cmap=cmap,
        xticklabels=tick_labels,
        yticklabels=tick_labels,
        square=True,
        vmin=vmin,
        vmax=vmax,
        annot=annot,
        fmt=".2f" if annot else "",
        cbar_kws={"label": cbar_label, "shrink": 0.8},
    )

    if title is None:
        title = "Pairwise distance"
        if metric:
            title = f"{title} ({metric})"
    ax.set_title(title)

    show_with_warning(show)

    return fig, ax


def plot_dendrogram(
    linkage_matrix: np.ndarray,
    labels: list[str] | None = None,
    *,
    ax: plt.Axes | None = None,
    title: str | None = None,
    figsize: tuple[float, float] = (10, 6),
    leaf_rotation: float = 90.0,
    color_threshold: float | None = None,
    truncate_mode: str | None = None,
    p: int = 30,
    show: bool = True,
) -> tuple[plt.Figure, plt.Axes]:
    """Plot a dendrogram from a hierarchical clustering linkage matrix.

    Parameters
    ----------
    linkage_matrix : ndarray of shape (n - 1, 4)
        Linkage matrix from
        :func:`~maldiamrkit.similarity.hierarchical_clustering`.
    labels : list of str or None, default=None
        Leaf labels.
    ax : Axes or None, default=None
        Pre-existing axes.
    title : str or None, default=None
        Plot title.  Defaults to
        ``"Hierarchical clustering dendrogram"``.
    figsize : tuple of float, default=(10, 6)
        Figure size in inches (used only when ``ax`` is ``None``).
    leaf_rotation : float, default=90.0
        Rotation (in degrees) of leaf labels along the bottom axis.
    color_threshold : float, optional
        Colour threshold forwarded to scipy's ``dendrogram``.  Clusters
        below this threshold share a colour.  When ``None`` (default),
        scipy chooses ``0.7 * max(linkage[:, 2])``.
    truncate_mode : {"lastp", "level", None}, optional
        Forwarded to scipy's ``dendrogram`` to collapse deep branches.
        Essential for large trees; pair with ``p``.
    p : int, default=30
        Number of leaves / merges to keep when ``truncate_mode`` is set.
    show : bool, default=True
        Call ``plt.show()`` at the end.

    Returns
    -------
    fig : matplotlib.figure.Figure
    ax : matplotlib.axes.Axes
    """
    import matplotlib.pyplot as plt
    from scipy.cluster.hierarchy import dendrogram

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()

    kwargs: dict = {
        "labels": labels,
        "ax": ax,
        "leaf_rotation": leaf_rotation,
    }
    if color_threshold is not None:
        kwargs["color_threshold"] = color_threshold
    if truncate_mode is not None:
        kwargs["truncate_mode"] = truncate_mode
        kwargs["p"] = p

    dendrogram(linkage_matrix, **kwargs)

    ax.set_title(title or "Hierarchical clustering dendrogram")

    show_with_warning(show)

    return fig, ax
