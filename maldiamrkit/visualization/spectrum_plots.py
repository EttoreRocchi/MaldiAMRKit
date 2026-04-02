"""Spectrum and dataset plotting functions."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

    from ..dataset import MaldiSet
    from ..spectrum import MaldiSpectrum


def plot_spectrum(
    spectrum: MaldiSpectrum,
    binned: bool = True,
    ax: Axes | None = None,
    **kwargs: Any,
) -> Axes:
    """Plot a single MALDI-TOF spectrum.

    Parameters
    ----------
    spectrum : MaldiSpectrum
        Spectrum to plot.
    binned : bool, default=True
        If True, plot the binned spectrum. Otherwise, plot preprocessed
        or raw spectrum.
    ax : matplotlib.axes.Axes, optional
        Axes to plot on. If None, creates a new figure.
    **kwargs : dict
        Additional keyword arguments passed to the plotting function.

    Returns
    -------
    matplotlib.axes.Axes
        The axes containing the plot.
    """
    import matplotlib.pyplot as plt
    import seaborn as sns

    _ax = ax or plt.subplots(figsize=(10, 4))[1]
    data = spectrum.binned if binned else (spectrum.get_data(prefer="preprocessed"))
    if binned:
        sns.barplot(data=data, x="mass", y="intensity", ax=_ax, **kwargs)
    else:
        _ax.plot(data.mass, data.intensity, **kwargs)
    _ax.set(
        title=f"{spectrum.id}{' (binned)' if binned else ''}",
        xlabel="m/z",
        ylabel="intensity",
        xticks=[],
        ylim=[0, (data.intensity.max()) * 1.05],
    )
    return _ax


def plot_pseudogel(
    dataset: MaldiSet,
    *,
    antibiotic: str | None = None,
    regions: tuple[float, float] | list[tuple[float, float]] | None = None,
    cmap: str = "inferno",
    vmin: float | None = None,
    vmax: float | None = None,
    figsize: tuple[float, float] | None = None,
    log_scale: bool = True,
    sort_by_intensity: bool = True,
    title: str | None = None,
    show: bool = True,
) -> tuple[Figure, np.ndarray]:
    """Display a pseudogel heatmap of the spectra.

    Creates one subplot for each unique value of the antibiotic column.

    Parameters
    ----------
    dataset : MaldiSet
        Dataset to visualize.
    antibiotic : str, optional
        Target column to group by. Defaults to first antibiotic.
    regions : tuple or list of tuples, optional
        m/z region(s) to display. None shows all.
    cmap : str, default="inferno"
        Matplotlib colormap name.
    vmin, vmax : float, optional
        Color scale limits.
    figsize : tuple, optional
        Figure size. Auto-calculated if None.
    log_scale : bool, default=True
        Apply log1p to intensities.
    sort_by_intensity : bool, default=True
        Sort samples by average intensity.
    title : str, optional
        Figure title.
    show : bool, default=True
        If True, call plt.show().

    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure object.
    axes : ndarray of Axes
        The subplot axes.

    Raises
    ------
    ValueError
        If antibiotic column is not defined, if a region has
        min_mz > max_mz, or if no m/z values are found in a
        specified region.
    """
    import matplotlib.pyplot as plt

    if antibiotic is None:
        antibiotic = dataset.antibiotics[0] if dataset.antibiotics else None
    if antibiotic is None:
        raise ValueError("Antibiotic column not defined.")

    X = dataset.X.copy()
    y = dataset.get_y_single(antibiotic)

    X = _apply_region_filter(X, regions)

    groups = y.groupby(y).groups
    n_groups = len(groups)
    if figsize is None:
        figsize = (6.0, 2.5 * n_groups)

    fig, axes = plt.subplots(
        n_groups, 1, figsize=figsize, sharex=True, constrained_layout=True
    )
    if n_groups == 1:
        axes = np.asarray([axes])

    cmap_obj = plt.get_cmap(cmap).copy()
    cmap_obj.set_bad(color="white", alpha=1.0)

    for ax, (label, idx) in zip(
        axes, sorted(groups.items(), key=lambda t: str(t[0])), strict=True
    ):
        im = _render_pseudogel_group(
            ax,
            X.loc[idx].to_numpy(),
            label,
            log_scale,
            sort_by_intensity,
            cmap_obj,
            vmin,
            vmax,
        )

    _set_pseudogel_xaxis(axes, X)

    cbar = fig.colorbar(im, ax=axes, orientation="vertical", pad=0.01)
    cbar.set_label("Log(intensity + 1)" if log_scale else "intensity")

    if title:
        fig.suptitle(title, y=1.02)

    if show:
        plt.show()

    return fig, axes


def _render_pseudogel_group(
    ax, M, label, log_scale, sort_by_intensity, cmap_obj, vmin, vmax
):
    """Render a single group panel in a pseudogel heatmap."""
    if sort_by_intensity:
        order = np.argsort(np.nanmean(M, axis=1))[::-1]
        M = M[order]
    if log_scale:
        M = np.log1p(M)

    im = ax.imshow(
        M,
        aspect="auto",
        interpolation="nearest",
        cmap=cmap_obj,
        vmin=vmin,
        vmax=vmax,
    )
    ax.set_ylabel(f"{label}\n(n={M.shape[0]})", rotation=0, ha="right", va="center")
    ax.set_yticks([])
    return im


def _apply_region_filter(
    X: pd.DataFrame,
    regions: tuple[float, float] | list[tuple[float, float]] | None,
) -> pd.DataFrame:
    """Filter feature matrix to specified m/z regions."""
    if regions is None:
        return X

    if (
        isinstance(regions, tuple)
        and len(regions) == 2
        and not isinstance(regions[0], (tuple, list))
    ):
        regions = [regions]

    mz_values = X.columns.astype(float)
    region_dfs = []

    for min_mz, max_mz in regions:
        if min_mz > max_mz:
            raise ValueError(f"Invalid region: min_mz ({min_mz}) > max_mz ({max_mz})")

        mask = (mz_values >= min_mz) & (mz_values <= max_mz)
        if not mask.any():
            raise ValueError(f"No m/z values found in region ({min_mz}, {max_mz})")

        region_dfs.append(X.iloc[:, mask])

        if len(region_dfs) < len(regions):
            blank_col = pd.DataFrame(
                np.nan, index=X.index, columns=[f"_blank_{len(region_dfs)}"]
            )
            region_dfs.append(blank_col)

    return pd.concat(region_dfs, axis=1)


def _set_pseudogel_xaxis(axes: np.ndarray, X: pd.DataFrame) -> None:
    """Set x-axis ticks and labels for pseudogel plot."""
    n_ticks = min(10, X.shape[1])
    xticks = np.linspace(0, X.shape[1] - 1, n_ticks, dtype=int)

    xticklabels = []
    for i in xticks:
        col_name = str(X.columns[i])
        if col_name.startswith("_blank_"):
            xticklabels.append("")
        else:
            xticklabels.append(col_name)

    axes[-1].set_xticks(xticks)
    axes[-1].set_xticklabels(xticklabels, rotation=90)
    axes[-1].set_xlabel("m/z (binned)")
