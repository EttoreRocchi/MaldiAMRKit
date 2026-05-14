"""Dimensionality reduction and exploratory scatter plots."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from ._common import order_labels, resolve_display_label, show_with_warning

if TYPE_CHECKING:
    import matplotlib.pyplot as plt


def _auto_marker_size(n: int) -> float:
    """Scale marker size inversely with sample count."""
    return max(8.0, 2000.0 / max(n, 1))


def _reduce_dimensions(
    X: pd.DataFrame | np.ndarray,
    method: str,
    n_components: int = 2,
    *,
    standardize: bool = True,
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
    standardize : bool, default=True
        If ``True``, features are zero-mean / unit-variance scaled
        before the reducer is fit, so no single high-intensity bin can
        dominate the embedding.  For ``"pca"`` this yields a
        ``Pipeline(StandardScaler -> PCA)`` whose ``transform`` applies
        both steps consistently at inference time (see
        :class:`maldiamrkit.drift.DriftMonitor`).  Set to ``False`` only
        if features are already on a comparable scale.
    **kwargs : dict
        Extra keyword arguments forwarded to the reducer constructor.

    Returns
    -------
    embedding : np.ndarray
        Array of shape ``(n_samples, n_components)``.
    reducer : object
        The fitted reducer.  For ``"pca"`` with ``standardize=True``,
        returns a :class:`sklearn.pipeline.Pipeline` whose last step is
        the PCA estimator (accessible via
        ``reducer.named_steps['pca']`` for e.g.
        ``explained_variance_ratio_``).

    Raises
    ------
    ImportError
        If ``method="umap"`` and *umap-learn* is not installed.
    ValueError
        If *method* is not one of the supported algorithms.
    """
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    arr = np.asarray(X, dtype=float)

    if method == "pca":
        from sklearn.decomposition import PCA

        pca = PCA(n_components=n_components, **kwargs)
        if standardize:
            reducer: object = Pipeline([("scaler", StandardScaler()), ("pca", pca)])
        else:
            reducer = pca
        embedding = reducer.fit_transform(arr)

    elif method == "tsne":
        from sklearn.manifold import TSNE

        tsne = TSNE(n_components=n_components, **kwargs)
        if standardize:
            arr = StandardScaler().fit_transform(arr)
        reducer = tsne
        embedding = reducer.fit_transform(arr)

    elif method == "umap":
        try:
            import umap
        except ImportError:
            raise ImportError(
                "umap-learn is required for UMAP plots. "
                "Install it with: pip install maldiamrkit[batch]"
            ) from None
        umap_reducer = umap.UMAP(n_components=n_components, **kwargs)
        if standardize:
            arr = StandardScaler().fit_transform(arr)
        reducer = umap_reducer
        embedding = reducer.fit_transform(arr)

    else:
        raise ValueError(
            f"Unknown method {method!r}. Choose from 'pca', 'tsne', 'umap'."
        )

    return embedding, reducer


def _resolve_colors(unique_labels: list, palette: str | dict | None) -> dict:
    """Build a label-to-color mapping from a palette specification."""
    import matplotlib.pyplot as plt

    if isinstance(palette, dict):
        return palette
    cmap = plt.get_cmap(palette or "tab10")
    return {lab: cmap(i % cmap.N) for i, lab in enumerate(unique_labels)}


def _scatter_embedding(
    embedding: np.ndarray,
    color_by: pd.Series | np.ndarray | None = None,
    *,
    ax: plt.Axes | None = None,
    palette: str | dict | None = None,
    label_map: dict | None = None,
    title: str | None = None,
    xlabel: str | None = None,
    ylabel: str | None = None,
    figsize: tuple[float, float] = (8, 6),
    alpha: float = 0.7,
    s: float | None = None,
    legend: bool = True,
    legend_loc: str = "best",
    grid: bool = True,
    show: bool = True,
) -> tuple[plt.Figure, plt.Axes]:
    """Render a 2-D scatter plot of an embedding.

    Parameters
    ----------
    embedding : np.ndarray
        Array of shape ``(n_samples, 2)``.
    color_by : pd.Series, np.ndarray, or None
        Categorical labels used to color points.  Always rendered as
        discrete categories (no continuous colorbar branch); numeric
        ``color_by`` with many unique values is accepted but will
        produce one legend entry per value.
    ax : matplotlib.axes.Axes, optional
        Axes to draw on. If ``None``, a new figure is created.
    palette : str or dict, optional
        Matplotlib colormap name or ``{label: color}`` mapping.
    label_map : dict, optional
        Map raw label values to display strings shown in the legend.
        Defaults to the S/I/R and 0/1 → susceptible/resistant mapping;
        user values override entries in the default map.
    title : str, optional
        Plot title.
    xlabel, ylabel : str, optional
        Axis labels.
    figsize : tuple of float, default=(8, 6)
        Figure size (only used when *ax* is ``None``).
    alpha : float, default=0.7
        Point transparency.
    s : float, optional
        Marker size.  When ``None``, auto-scales with sample count
        (``max(8, 2000/n)``).
    legend : bool, default=True
        Whether to show a legend when *color_by* is provided.
    legend_loc : str, default="best"
        ``matplotlib`` legend location string (e.g. ``"upper right"``)
        or ``"outside"`` to place the legend outside the axes.
    grid : bool, default=True
        Draw a faint background grid.
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

    n = embedding.shape[0]
    marker_size = _auto_marker_size(n) if s is None else s

    if color_by is None:
        ax.scatter(embedding[:, 0], embedding[:, 1], alpha=alpha, s=marker_size)
    else:
        labels = np.asarray(color_by)
        unique_labels = order_labels(list(dict.fromkeys(labels.tolist())))
        colors = _resolve_colors(unique_labels, palette)

        for lab in unique_labels:
            mask = labels == lab
            display = resolve_display_label(lab, label_map)
            ax.scatter(
                embedding[mask, 0],
                embedding[mask, 1],
                label=display,
                color=colors[lab],
                alpha=alpha,
                s=marker_size,
            )
        if legend:
            legend_title = color_by.name if isinstance(color_by, pd.Series) else None
            if legend_loc == "outside":
                ax.legend(
                    title=legend_title,
                    bbox_to_anchor=(1.02, 1.0),
                    loc="upper left",
                    borderaxespad=0.0,
                )
            else:
                ax.legend(title=legend_title, loc=legend_loc)

    if title:
        ax.set_title(title)
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    if grid:
        ax.grid(True, alpha=0.3)

    show_with_warning(show)

    return fig, ax


def plot_pca(
    X: pd.DataFrame,
    color_by: pd.Series | np.ndarray | None = None,
    n_components: int = 2,
    *,
    random_state: int | None = 42,
    ax: plt.Axes | None = None,
    palette: str | dict | None = None,
    label_map: dict | None = None,
    title: str | None = None,
    figsize: tuple[float, float] = (8, 6),
    alpha: float = 0.7,
    s: float | None = None,
    legend: bool = True,
    legend_loc: str = "best",
    grid: bool = True,
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
    random_state : int or None, default=42
        Random seed forwarded to :class:`sklearn.decomposition.PCA` for
        reproducibility.  Pass ``None`` for non-deterministic results.
    ax : matplotlib.axes.Axes, optional
        Axes to draw on. If ``None``, a new figure is created.
    palette : str or dict, optional
        Colormap name or ``{label: color}`` mapping.
    label_map : dict, optional
        Map raw label values to display strings shown in the legend.
        Defaults to the S/I/R and 0/1 → susceptible/resistant mapping;
        user values override entries in the default map.
    title : str, optional
        Plot title. Defaults to ``"PCA"``.
    figsize : tuple of float, default=(8, 6)
        Figure size (only used when *ax* is ``None``).
    alpha : float, default=0.7
        Point transparency.
    s : float, optional
        Marker size.  When ``None`` (the default), auto-scales with
        sample count (``max(8, 2000/n)``).
    legend : bool, default=True
        Whether to show a legend.
    legend_loc : str, default="best"
        ``matplotlib`` legend location string (e.g. ``"upper right"``)
        or ``"outside"`` to place the legend outside the axes.
    grid : bool, default=True
        Draw a faint background grid.
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
    embedding, reducer = _reduce_dimensions(
        X, "pca", n_components, random_state=random_state, **pca_kwargs
    )
    pca_estimator = (
        reducer.named_steps["pca"] if hasattr(reducer, "named_steps") else reducer
    )
    var = pca_estimator.explained_variance_ratio_ * 100

    xlabel = f"PC1 ({var[0]:.1f}%)"
    ylabel = f"PC2 ({var[1]:.1f}%)" if n_components >= 2 else None

    return _scatter_embedding(
        embedding,
        color_by=color_by,
        ax=ax,
        palette=palette,
        label_map=label_map,
        title=title or "PCA",
        xlabel=xlabel,
        ylabel=ylabel,
        figsize=figsize,
        alpha=alpha,
        s=s,
        legend=legend,
        legend_loc=legend_loc,
        grid=grid,
        show=show,
    )


def plot_tsne(
    X: pd.DataFrame,
    color_by: pd.Series | np.ndarray | None = None,
    n_components: int = 2,
    *,
    perplexity: float = 30.0,
    random_state: int | None = 42,
    ax: plt.Axes | None = None,
    palette: str | dict | None = None,
    label_map: dict | None = None,
    title: str | None = None,
    figsize: tuple[float, float] = (8, 6),
    alpha: float = 0.7,
    s: float | None = None,
    legend: bool = True,
    legend_loc: str = "best",
    grid: bool = True,
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
    random_state : int or None, default=42
        Random seed for reproducibility.  Pass ``None`` for
        non-deterministic results.
    ax : matplotlib.axes.Axes, optional
        Axes to draw on.
    palette : str or dict, optional
        Colormap name or ``{label: color}`` mapping.
    label_map : dict, optional
        Map raw label values to display strings shown in the legend.
        Defaults to the S/I/R and 0/1 → susceptible/resistant mapping;
        user values override entries in the default map.
    title : str, optional
        Plot title. Defaults to ``"t-SNE"``.
    figsize : tuple of float, default=(8, 6)
        Figure size.
    alpha : float, default=0.7
        Point transparency.
    s : float, optional
        Marker size.  When ``None`` (the default), auto-scales with
        sample count (``max(8, 2000/n)``).
    legend : bool, default=True
        Whether to show a legend.
    legend_loc : str, default="best"
        ``matplotlib`` legend location string (e.g. ``"upper right"``)
        or ``"outside"`` to place the legend outside the axes.
    grid : bool, default=True
        Draw a faint background grid.
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
        X,
        "tsne",
        n_components,
        perplexity=perplexity,
        random_state=random_state,
        **tsne_kwargs,
    )

    return _scatter_embedding(
        embedding,
        color_by=color_by,
        ax=ax,
        palette=palette,
        label_map=label_map,
        title=title or "t-SNE",
        xlabel="t-SNE 1",
        ylabel="t-SNE 2" if n_components >= 2 else None,
        figsize=figsize,
        alpha=alpha,
        s=s,
        legend=legend,
        legend_loc=legend_loc,
        grid=grid,
        show=show,
    )


def plot_umap(
    X: pd.DataFrame,
    color_by: pd.Series | np.ndarray | None = None,
    n_components: int = 2,
    *,
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    random_state: int | None = 42,
    ax: plt.Axes | None = None,
    palette: str | dict | None = None,
    label_map: dict | None = None,
    title: str | None = None,
    figsize: tuple[float, float] = (8, 6),
    alpha: float = 0.7,
    s: float | None = None,
    legend: bool = True,
    legend_loc: str = "best",
    grid: bool = True,
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
    random_state : int or None, default=42
        Random seed for reproducibility.  Pass ``None`` for
        non-deterministic results.
    ax : matplotlib.axes.Axes, optional
        Axes to draw on.
    palette : str or dict, optional
        Colormap name or ``{label: color}`` mapping.
    label_map : dict, optional
        Map raw label values to display strings shown in the legend.
        Defaults to the S/I/R and 0/1 → susceptible/resistant mapping;
        user values override entries in the default map.
    title : str, optional
        Plot title. Defaults to ``"UMAP"``.
    figsize : tuple of float, default=(8, 6)
        Figure size.
    alpha : float, default=0.7
        Point transparency.
    s : float, optional
        Marker size.  When ``None`` (the default), auto-scales with
        sample count (``max(8, 2000/n)``).
    legend : bool, default=True
        Whether to show a legend.
    legend_loc : str, default="best"
        ``matplotlib`` legend location string (e.g. ``"upper right"``)
        or ``"outside"`` to place the legend outside the axes.
    grid : bool, default=True
        Draw a faint background grid.
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
        random_state=random_state,
        **umap_kwargs,
    )

    return _scatter_embedding(
        embedding,
        color_by=color_by,
        ax=ax,
        palette=palette,
        label_map=label_map,
        title=title or "UMAP",
        xlabel="UMAP 1",
        ylabel="UMAP 2" if n_components >= 2 else None,
        figsize=figsize,
        alpha=alpha,
        s=s,
        legend=legend,
        legend_loc=legend_loc,
        grid=grid,
        show=show,
    )
