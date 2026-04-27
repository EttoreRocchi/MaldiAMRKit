"""Spectrum and dataset plotting functions."""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
import pandas as pd

from ._common import DEFAULT_LABEL_MAP, order_labels, show_with_warning

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

    from ..dataset import MaldiSet
    from ..spectrum import MaldiSpectrum


SpectrumStage = Literal["binned", "preprocessed", "raw"]


def _resolve_stage(
    spectrum: MaldiSpectrum,
    stage: SpectrumStage,
    binned: bool | None,
) -> tuple[SpectrumStage, pd.DataFrame]:
    """Pick the stage to plot and fetch its DataFrame.

    Handles the deprecated ``binned`` boolean: ``binned=True`` maps to
    ``"binned"`` and ``binned=False`` falls back to the best available
    non-binned stage (``"preprocessed"`` over ``"raw"``).
    """
    if binned is not None:
        warnings.warn(
            "plot_spectrum(binned=...) is deprecated; use "
            "stage='binned'|'preprocessed'|'raw' instead.",
            DeprecationWarning,
            stacklevel=3,
        )
        if binned:
            stage = "binned"
        else:
            stage = "preprocessed" if spectrum.is_preprocessed else "raw"

    if stage == "binned":
        return stage, spectrum.binned
    if stage == "preprocessed":
        return stage, spectrum.preprocessed
    if stage == "raw":
        return stage, spectrum.raw
    raise ValueError(
        f"Unknown stage {stage!r}; expected 'binned', 'preprocessed', or 'raw'."
    )


def plot_spectrum(
    spectrum: MaldiSpectrum,
    *,
    stage: SpectrumStage = "binned",
    peaks: list[float] | np.ndarray | None = None,
    highlight_regions: list[tuple[float, float]] | None = None,
    ax: Axes | None = None,
    color: str | None = None,
    figsize: tuple[float, float] = (10, 4),
    title: str | None = None,
    log_y: bool = False,
    ylim: tuple[float, float] | None = None,
    show: bool = True,
    binned: bool | None = None,
    **kwargs: Any,
) -> tuple[Figure, Axes]:
    """Plot a single MALDI-TOF spectrum with real m/z axis.

    Parameters
    ----------
    spectrum : MaldiSpectrum
        Spectrum to plot.
    stage : {"binned", "preprocessed", "raw"}, default="binned"
        Processing stage to render. ``"binned"`` uses a bar plot with
        bar width inferred from the bin spacing; ``"preprocessed"`` and
        ``"raw"`` use a line plot.
    peaks : list of float or ndarray, optional
        If given, draw a scatter marker above the spectrum at each
        peak m/z.
    highlight_regions : list of (mz_min, mz_max) tuples, optional
        Shaded m/z bands drawn behind the spectrum (e.g. regions of
        interest from differential analysis).
    ax : matplotlib.axes.Axes, optional
        Axes to plot on. If None, creates a new figure.
    color : str, optional
        Colour for the spectrum (bars / line). Matplotlib default used
        when None.
    figsize : tuple of float, default=(10, 4)
        Figure size in inches (only used when ``ax`` is None).
    title : str, optional
        Overrides the auto-generated title
        (``"{spectrum.id} ({stage})"``).
    log_y : bool, default=False
        Use a logarithmic y-axis.
    ylim : tuple of float, optional
        Override y-axis limits.  Defaults to matplotlib autoscaling
        (no clipping of negatives).
    show : bool, default=True
        Call ``plt.show()`` at the end.
    binned : bool, optional
        *Deprecated.* Use ``stage=`` instead. ``binned=True`` maps to
        ``stage="binned"``; ``binned=False`` maps to ``"preprocessed"``
        if available, else ``"raw"``.
    **kwargs : dict
        Additional keyword arguments forwarded to ``ax.bar`` (binned
        stage) or ``ax.plot`` (raw / preprocessed).

    Returns
    -------
    fig : matplotlib.figure.Figure
    ax : matplotlib.axes.Axes
    """
    import matplotlib.pyplot as plt

    resolved_stage, data = _resolve_stage(spectrum, stage, binned)

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()

    mass = data["mass"].to_numpy(dtype=float)
    intensity = data["intensity"].to_numpy(dtype=float)

    if highlight_regions:
        for low, high in highlight_regions:
            ax.axvspan(low, high, color="goldenrod", alpha=0.15, zorder=0)

    if resolved_stage == "binned":
        # Bar width derived from bin spacing: median diff is robust to
        # non-uniform binning (adaptive / custom), uniform falls through.
        if len(mass) >= 2:
            bar_width = float(np.median(np.diff(mass)))
        else:
            bar_width = 1.0
        ax.bar(
            mass,
            intensity,
            width=bar_width,
            align="center",
            color=color,
            linewidth=0,
            **kwargs,
        )
    else:
        ax.plot(mass, intensity, color=color, **kwargs)

    if peaks is not None:
        peaks = np.asarray(peaks, dtype=float)
        if peaks.size:
            y_at_peaks = np.interp(peaks, mass, intensity)
            ax.scatter(
                peaks,
                y_at_peaks,
                marker="v",
                color="crimson",
                s=30,
                zorder=5,
                label="peak",
            )

    ax.set_xlabel("m/z (Da)")
    ax.set_ylabel("intensity")
    ax.set_title(title or f"{spectrum.id} ({resolved_stage})")
    if log_y:
        ax.set_yscale("log")
    if ylim is not None:
        ax.set_ylim(ylim)

    show_with_warning(show)

    return fig, ax


def plot_pseudogel(
    dataset: MaldiSet,
    *,
    antibiotic: str | None = None,
    species: str | None = None,
    regions: tuple[float, float] | list[tuple[float, float]] | None = None,
    cmap: str = "inferno",
    vmin: float | None = None,
    vmax: float | None = None,
    figsize: tuple[float, float] | None = None,
    log_scale: bool = True,
    sort_by: str | None = "intensity",
    label_map: dict | None = None,
    title: str | None = None,
    show: bool = True,
    sort_by_intensity: bool | None = None,
) -> tuple[Figure, np.ndarray]:
    """Display a pseudogel heatmap of the spectra.

    Creates one subplot per unique value of the antibiotic column, in
    susceptibility order (S, I, R) with unknown labels appended
    alphabetically.

    Parameters
    ----------
    dataset : MaldiSet
        Dataset to visualize.
    antibiotic : str, optional
        Target column to group by. Defaults to the first configured
        antibiotic in the MaldiSet.
    species : str, optional
        When given, restrict the pseudogel to that species via
        :class:`~maldiamrkit.filters.SpeciesFilter`.  Default ``None``
        keeps all samples.
    regions : tuple or list of tuples, optional
        m/z region(s) to display. None shows all.
    cmap : str, default="inferno"
        Matplotlib colormap name.
    vmin, vmax : float, optional
        Colour-scale limits in the *raw intensity* units the caller
        is familiar with.  When ``log_scale=True`` both values are
        automatically mapped through ``np.log1p`` before being passed
        to ``imshow``, so the plotted range matches what the user
        specified.
    figsize : tuple, optional
        Figure size.  Defaults to ``(14.0, 2.5 * n_groups)`` so the
        m/z axis is wide enough for typical binned data (thousands of
        columns).
    log_scale : bool, default=True
        Apply ``np.log1p`` to intensities.
    sort_by : {"intensity", "id", None}, default="intensity"
        How to order samples within each group:

        - ``"intensity"``: sort by mean intensity (descending).
        - ``"id"``: sort by the sample's index value (deterministic).
        - ``None``: keep the order encountered in the MaldiSet.
    label_map : dict, optional
        Mapping from raw group label to display name.  Default maps
        0/1 and R/I/S to ``"Susceptible (S)"`` / ``"Intermediate (I)"``
        / ``"Resistant (R)"``; any other value is stringified as-is.
        Pass a dict to override.
    title : str, optional
        Figure title.  Defaults to ``f"Pseudogel: {antibiotic}"`` when
        omitted.
    show : bool, default=True
        Call ``plt.show()`` at the end.
    sort_by_intensity : bool, optional
        *Deprecated.* Use ``sort_by=`` instead.  Retained for
        backwards-compatibility; ``True`` maps to ``sort_by="intensity"``
        and ``False`` maps to ``sort_by=None``.

    Returns
    -------
    fig : matplotlib.figure.Figure
    axes : ndarray of Axes

    Raises
    ------
    ValueError
        If the antibiotic column is not defined, if a region has
        min_mz > max_mz, if no m/z values lie within a specified
        region, or if ``sort_by`` is not one of the recognised values.
    """
    import matplotlib.pyplot as plt

    from ..filters import SpeciesFilter

    if sort_by_intensity is not None:
        warnings.warn(
            "plot_pseudogel(sort_by_intensity=...) is deprecated; use "
            "sort_by='intensity'|'id'|None instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        sort_by = "intensity" if sort_by_intensity else None

    if sort_by not in (None, "intensity", "id"):
        raise ValueError(
            f"sort_by must be 'intensity', 'id', or None; got {sort_by!r}."
        )

    if antibiotic is None:
        antibiotic = dataset.antibiotics[0] if dataset.antibiotics else None
    if antibiotic is None:
        raise ValueError("Antibiotic column not defined.")

    if species is not None:
        dataset = dataset.filter(SpeciesFilter(species))

    X = dataset.X.copy()
    y = dataset.get_y_single(antibiotic)

    X = _apply_region_filter(X, regions)

    groups = y.groupby(y).groups
    n_groups = len(groups)
    if figsize is None:
        figsize = (14.0, 2.5 * max(1, n_groups))

    fig, axes = plt.subplots(
        n_groups, 1, figsize=figsize, sharex=True, constrained_layout=True
    )
    if n_groups == 1:
        axes = np.asarray([axes])

    cmap_obj = plt.get_cmap(cmap).copy()
    cmap_obj.set_bad(color="white", alpha=1.0)

    # Map user-supplied vmin/vmax from raw-intensity to display units
    # so they behave consistently with log_scale.
    display_vmin = np.log1p(vmin) if (log_scale and vmin is not None) else vmin
    display_vmax = np.log1p(vmax) if (log_scale and vmax is not None) else vmax

    merged_label_map: dict = dict(DEFAULT_LABEL_MAP)
    if label_map:
        merged_label_map.update(label_map)

    ordered_items = [(lab, groups[lab]) for lab in order_labels(list(groups))]

    im = None
    for ax, (label, idx) in zip(axes, ordered_items, strict=True):
        display_label = merged_label_map.get(label, str(label))
        im = _render_pseudogel_group(
            ax,
            X.loc[idx].to_numpy(),
            display_label,
            log_scale,
            sort_by,
            cmap_obj,
            display_vmin,
            display_vmax,
            sample_ids=list(idx),
        )

    _set_pseudogel_xaxis(axes, X)

    if im is not None:
        cbar = fig.colorbar(im, ax=axes, orientation="vertical", pad=0.01)
        cbar.set_label("Log(intensity + 1)" if log_scale else "intensity")

    fig.suptitle(title or f"Pseudogel: {antibiotic}")

    if show:
        plt.show()

    return fig, axes


def _render_pseudogel_group(
    ax,
    M,
    label,
    log_scale,
    sort_by,
    cmap_obj,
    vmin,
    vmax,
    *,
    sample_ids=None,
):
    """Render a single group panel in a pseudogel heatmap."""
    order: np.ndarray | None = None
    if sort_by == "intensity":
        order = np.argsort(np.nanmean(M, axis=1))[::-1]
    elif sort_by == "id" and sample_ids is not None:
        order = np.argsort([str(sid) for sid in sample_ids])
    if order is not None:
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
