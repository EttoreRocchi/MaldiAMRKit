"""Dimensionality reduction and exploratory scatter plots."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    import matplotlib.pyplot as plt


def _reduce_dimensions(
    X: pd.DataFrame | np.ndarray,
    method: str,
    n_components: int = 2,
    **kwargs,
) -> tuple[np.ndarray, object]:
    """Compute a low-dimensional embedding.

    Parameters
    ----------
    X : pd.DataFrame or np.ndarray
        Feature matrix of shape ``(n_samples, n_features)``.
    method : {"pca", "tsne", "umap"}
        Dimensionality reduction algorithm.
    n_components : int, default=2
        Number of output dimensions.
    **kwargs : dict
        Extra keyword arguments forwarded to the reducer constructor.

    Returns
    -------
    embedding : np.ndarray
        Array of shape ``(n_samples, n_components)``.
    reducer : object
        The fitted reducer (useful for extracting explained variance
        from PCA).

    Raises
    ------
    ImportError
        If ``method="umap"`` and *umap-learn* is not installed.
    ValueError
        If *method* is not one of the supported algorithms.
    """
    arr = np.asarray(X)

    if method == "pca":
        from sklearn.decomposition import PCA

        reducer = PCA(n_components=n_components, **kwargs)
        embedding = reducer.fit_transform(arr)

    elif method == "tsne":
        from sklearn.manifold import TSNE

        reducer = TSNE(n_components=n_components, **kwargs)
        embedding = reducer.fit_transform(arr)

    elif method == "umap":
        try:
            import umap
        except ImportError:
            raise ImportError(
                "umap-learn is required for UMAP plots. "
                "Install it with: pip install maldiamrkit[batch]"
            ) from None
        reducer = umap.UMAP(n_components=n_components, **kwargs)
        embedding = reducer.fit_transform(arr)

    else:
        raise ValueError(
            f"Unknown method {method!r}. Choose from 'pca', 'tsne', 'umap'."
        )

    return embedding, reducer


def _scatter_embedding(
    embedding: np.ndarray,
    color_by: pd.Series | np.ndarray | None = None,
    *,
    ax: plt.Axes | None = None,
    palette: str | dict | None = None,
    title: str | None = None,
    xlabel: str | None = None,
    ylabel: str | None = None,
    figsize: tuple[float, float] = (8, 6),
    alpha: float = 0.7,
    s: float = 20,
    legend: bool = True,
    show: bool = True,
) -> tuple[plt.Figure, plt.Axes]:
    """Render a 2-D scatter plot of an embedding.

    Parameters
    ----------
    embedding : np.ndarray
        Array of shape ``(n_samples, 2)``.
    color_by : pd.Series, np.ndarray, or None
        Categorical labels used to color points.
    ax : matplotlib.axes.Axes, optional
        Axes to draw on. If ``None``, a new figure is created.
    palette : str or dict, optional
        Matplotlib colormap name or ``{label: color}`` mapping.
    title : str, optional
        Plot title.
    xlabel, ylabel : str, optional
        Axis labels.
    figsize : tuple of float, default=(8, 6)
        Figure size (only used when *ax* is ``None``).
    alpha : float, default=0.7
        Point transparency.
    s : float, default=20
        Marker size.
    legend : bool, default=True
        Whether to show a legend when *color_by* is provided.
    show : bool, default=True
        Call ``plt.show()`` at the end.

    Returns
    -------
    fig : matplotlib.figure.Figure
    ax : matplotlib.axes.Axes
    """
    import matplotlib.pyplot as plt

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()

    if color_by is None:
        ax.scatter(embedding[:, 0], embedding[:, 1], alpha=alpha, s=s)
    else:
        labels = np.asarray(color_by)
        unique_labels = list(dict.fromkeys(labels))

        if isinstance(palette, dict):
            colors = palette
        else:
            cmap = plt.get_cmap(palette or "tab10")
            colors = {lab: cmap(i % cmap.N) for i, lab in enumerate(unique_labels)}

        for lab in unique_labels:
            mask = labels == lab
            ax.scatter(
                embedding[mask, 0],
                embedding[mask, 1],
                label=lab,
                color=colors[lab],
                alpha=alpha,
                s=s,
            )
        if legend:
            ax.legend(title=color_by.name if isinstance(color_by, pd.Series) else None)

    if title:
        ax.set_title(title)
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)

    if show:
        plt.show()

    return fig, ax


def plot_pca(
    X: pd.DataFrame,
    color_by: pd.Series | np.ndarray | None = None,
    n_components: int = 2,
    *,
    ax: plt.Axes | None = None,
    palette: str | dict | None = None,
    title: str | None = None,
    figsize: tuple[float, float] = (8, 6),
    alpha: float = 0.7,
    s: float = 20,
    legend: bool = True,
    show: bool = True,
    **pca_kwargs,
) -> tuple[plt.Figure, plt.Axes]:
    """Scatter plot of a PCA embedding colored by metadata.

    Axis labels include the percentage of explained variance for each
    principal component.

    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix of shape ``(n_samples, n_features)``.
    color_by : pd.Series, np.ndarray, or None
        Categorical labels used to color points (e.g. species, resistance
        phenotype, batch).
    n_components : int, default=2
        Number of principal components.
    ax : matplotlib.axes.Axes, optional
        Axes to draw on. If ``None``, a new figure is created.
    palette : str or dict, optional
        Colormap name or ``{label: color}`` mapping.
    title : str, optional
        Plot title. Defaults to ``"PCA"``.
    figsize : tuple of float, default=(8, 6)
        Figure size (only used when *ax* is ``None``).
    alpha : float, default=0.7
        Point transparency.
    s : float, default=20
        Marker size.
    legend : bool, default=True
        Whether to show a legend.
    show : bool, default=True
        Call ``plt.show()`` at the end.
    **pca_kwargs : dict
        Extra keyword arguments forwarded to
        :class:`sklearn.decomposition.PCA`.

    Returns
    -------
    fig : matplotlib.figure.Figure
    ax : matplotlib.axes.Axes

    Examples
    --------
    >>> from maldiamrkit.visualization import plot_pca
    >>> fig, ax = plot_pca(dataset.X, color_by=dataset.meta["Species"])
    """
    embedding, reducer = _reduce_dimensions(X, "pca", n_components, **pca_kwargs)
    var = reducer.explained_variance_ratio_ * 100

    xlabel = f"PC1 ({var[0]:.1f}%)"
    ylabel = f"PC2 ({var[1]:.1f}%)" if n_components >= 2 else None

    return _scatter_embedding(
        embedding,
        color_by=color_by,
        ax=ax,
        palette=palette,
        title=title or "PCA",
        xlabel=xlabel,
        ylabel=ylabel,
        figsize=figsize,
        alpha=alpha,
        s=s,
        legend=legend,
        show=show,
    )


def plot_tsne(
    X: pd.DataFrame,
    color_by: pd.Series | np.ndarray | None = None,
    n_components: int = 2,
    *,
    perplexity: float = 30.0,
    ax: plt.Axes | None = None,
    palette: str | dict | None = None,
    title: str | None = None,
    figsize: tuple[float, float] = (8, 6),
    alpha: float = 0.7,
    s: float = 20,
    legend: bool = True,
    show: bool = True,
    **tsne_kwargs,
) -> tuple[plt.Figure, plt.Axes]:
    """Scatter plot of a t-SNE embedding colored by metadata.

    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix of shape ``(n_samples, n_features)``.
    color_by : pd.Series, np.ndarray, or None
        Categorical labels used to color points.
    n_components : int, default=2
        Number of t-SNE dimensions.
    perplexity : float, default=30.0
        t-SNE perplexity parameter.
    ax : matplotlib.axes.Axes, optional
        Axes to draw on.
    palette : str or dict, optional
        Colormap name or ``{label: color}`` mapping.
    title : str, optional
        Plot title. Defaults to ``"t-SNE"``.
    figsize : tuple of float, default=(8, 6)
        Figure size.
    alpha : float, default=0.7
        Point transparency.
    s : float, default=20
        Marker size.
    legend : bool, default=True
        Whether to show a legend.
    show : bool, default=True
        Call ``plt.show()`` at the end.
    **tsne_kwargs : dict
        Extra keyword arguments forwarded to
        :class:`sklearn.manifold.TSNE`.

    Returns
    -------
    fig : matplotlib.figure.Figure
    ax : matplotlib.axes.Axes

    Examples
    --------
    >>> from maldiamrkit.visualization import plot_tsne
    >>> fig, ax = plot_tsne(dataset.X, color_by=labels, perplexity=15)
    """
    embedding, _ = _reduce_dimensions(
        X, "tsne", n_components, perplexity=perplexity, **tsne_kwargs
    )

    return _scatter_embedding(
        embedding,
        color_by=color_by,
        ax=ax,
        palette=palette,
        title=title or "t-SNE",
        xlabel="t-SNE 1",
        ylabel="t-SNE 2" if n_components >= 2 else None,
        figsize=figsize,
        alpha=alpha,
        s=s,
        legend=legend,
        show=show,
    )


def plot_umap(
    X: pd.DataFrame,
    color_by: pd.Series | np.ndarray | None = None,
    n_components: int = 2,
    *,
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    ax: plt.Axes | None = None,
    palette: str | dict | None = None,
    title: str | None = None,
    figsize: tuple[float, float] = (8, 6),
    alpha: float = 0.7,
    s: float = 20,
    legend: bool = True,
    show: bool = True,
    **umap_kwargs,
) -> tuple[plt.Figure, plt.Axes]:
    """Scatter plot of a UMAP embedding colored by metadata.

    Requires the optional ``umap-learn`` package.  Install it with::

        pip install maldiamrkit[batch]

    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix of shape ``(n_samples, n_features)``.
    color_by : pd.Series, np.ndarray, or None
        Categorical labels used to color points.
    n_components : int, default=2
        Number of UMAP dimensions.
    n_neighbors : int, default=15
        UMAP ``n_neighbors`` parameter.
    min_dist : float, default=0.1
        UMAP ``min_dist`` parameter.
    ax : matplotlib.axes.Axes, optional
        Axes to draw on.
    palette : str or dict, optional
        Colormap name or ``{label: color}`` mapping.
    title : str, optional
        Plot title. Defaults to ``"UMAP"``.
    figsize : tuple of float, default=(8, 6)
        Figure size.
    alpha : float, default=0.7
        Point transparency.
    s : float, default=20
        Marker size.
    legend : bool, default=True
        Whether to show a legend.
    show : bool, default=True
        Call ``plt.show()`` at the end.
    **umap_kwargs : dict
        Extra keyword arguments forwarded to :class:`umap.UMAP`.

    Returns
    -------
    fig : matplotlib.figure.Figure
    ax : matplotlib.axes.Axes

    Raises
    ------
    ImportError
        If ``umap-learn`` is not installed.

    Examples
    --------
    >>> from maldiamrkit.visualization import plot_umap
    >>> fig, ax = plot_umap(dataset.X, color_by=dataset.meta["Species"])
    """
    embedding, _ = _reduce_dimensions(
        X,
        "umap",
        n_components,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        **umap_kwargs,
    )

    return _scatter_embedding(
        embedding,
        color_by=color_by,
        ax=ax,
        palette=palette,
        title=title or "UMAP",
        xlabel="UMAP 1",
        ylabel="UMAP 2" if n_components >= 2 else None,
        figsize=figsize,
        alpha=alpha,
        s=s,
        legend=legend,
        show=show,
    )
