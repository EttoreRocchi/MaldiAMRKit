"""Visualizations for temporal drift monitoring."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from ..visualization._common import show_with_warning

if TYPE_CHECKING:
    import matplotlib.pyplot as plt


def plot_reference_drift(
    monitoring_df: pd.DataFrame,
    *,
    baseline_end: pd.Timestamp | str | None = None,
    warning_threshold: float | None = None,
    ax: plt.Axes | None = None,
    title: str | None = None,
    figsize: tuple[float, float] = (10, 4),
    show: bool = True,
) -> tuple[plt.Figure, plt.Axes]:
    """Line plot of reference-similarity distance over time.

    Parameters
    ----------
    monitoring_df : pd.DataFrame
        Output of :meth:`DriftMonitor.monitor`.  Must contain
        ``window_start`` and ``distance_to_reference`` columns.
    baseline_end : pd.Timestamp or str, optional
        If given, draw a dashed vertical line at this timestamp so the
        reader can tell where the baseline period ends and monitoring
        begins.
    warning_threshold : float, optional
        If given, draw a horizontal dashed line at this distance so
        windows exceeding the threshold are visually flagged.
    ax : Axes or None, default=None
        Pre-existing axes.
    title : str or None, default=None
        Plot title.  Defaults to ``"Reference drift"``.
    figsize : tuple of float, default=(10, 4)
        Figure size in inches (only used when ``ax`` is ``None``).
    show : bool, default=True
        Whether to call ``plt.show()``.

    Returns
    -------
    fig : matplotlib.figure.Figure
    ax : matplotlib.axes.Axes
    """
    import matplotlib.pyplot as plt

    required = {"window_start", "distance_to_reference"}
    missing = required - set(monitoring_df.columns)
    if missing:
        raise ValueError(f"'monitoring_df' is missing columns: {sorted(missing)}")

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()

    x = pd.to_datetime(monitoring_df["window_start"])
    y = monitoring_df["distance_to_reference"].to_numpy(dtype=float)
    ax.plot(x, y, marker="o", color="#111827", linewidth=1.5)

    if baseline_end is not None:
        ax.axvline(
            pd.to_datetime(baseline_end),
            color="#6b7280",
            linestyle="--",
            linewidth=0.8,
            label="baseline end",
        )
    if warning_threshold is not None:
        ax.axhline(
            warning_threshold,
            color="#ef4444",
            linestyle="--",
            linewidth=0.8,
            label=f"warning (>{warning_threshold:g})",
        )
    if baseline_end is not None or warning_threshold is not None:
        ax.legend(loc="best", frameon=False, fontsize=8)

    ax.set_xlabel("Window start")
    ax.set_ylabel("Distance to reference")
    ax.grid(True, linestyle=":", linewidth=0.5, color="#bdbdbd")
    ax.set_axisbelow(True)
    ax.set_title(title or "Reference drift")
    fig.autofmt_xdate()

    show_with_warning(show)
    return fig, ax


def plot_pca_drift(
    pca_df: pd.DataFrame,
    *,
    baseline_end: pd.Timestamp | str | None = None,
    ax: plt.Axes | None = None,
    title: str | None = None,
    figsize: tuple[float, float] = (8, 6),
    show: bool = True,
) -> tuple[plt.Figure, plt.Axes]:
    """PCA centroid trajectory colored by time.

    Consecutive windows are connected by a thin grey polyline so the
    reader can follow the temporal order; time direction is encoded by
    the colorbar (early → late).  Marker size encodes per-window
    dispersion (mean distance from centroid) when the ``dispersion``
    column is present.

    Parameters
    ----------
    pca_df : pd.DataFrame
        Output of :meth:`DriftMonitor.monitor_pca`.  Must contain
        ``window_start``, ``centroid_pc1``, and ``centroid_pc2`` columns;
        ``dispersion`` is used for marker sizing when available.
    baseline_end : pd.Timestamp or str, optional
        If given, ring the first post-baseline point with a thicker
        black outline and annotate it ``"post-baseline start"``.
    ax : Axes or None, default=None
        Pre-existing axes.
    title : str or None, default=None
        Plot title.  Defaults to ``"PCA centroid drift"``.
    figsize : tuple of float, default=(8, 6)
        Figure size in inches (only used when ``ax`` is ``None``).
    show : bool, default=True
        Whether to call ``plt.show()``.

    Returns
    -------
    fig : matplotlib.figure.Figure
    ax : matplotlib.axes.Axes
    """
    import matplotlib.pyplot as plt

    required = {"window_start", "centroid_pc1", "centroid_pc2"}
    missing = required - set(pca_df.columns)
    if missing:
        raise ValueError(f"'pca_df' is missing columns: {sorted(missing)}")

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()

    df = pca_df.copy()
    df["window_start"] = pd.to_datetime(df["window_start"])
    df = df.sort_values("window_start").reset_index(drop=True)

    x = df["centroid_pc1"].to_numpy(dtype=float)
    y = df["centroid_pc2"].to_numpy(dtype=float)
    time_idx = np.arange(len(df))

    if "dispersion" in df.columns and len(df) > 0:
        disp = df["dispersion"].to_numpy(dtype=float)
        max_disp = disp.max() if disp.size and disp.max() > 0 else 1.0
        sizes = 30.0 + 120.0 * (disp / max_disp)
    else:
        sizes = 60.0

    # Polyline connecting consecutive windows (behind the scatter so
    # markers stay readable). No arrowheads -- direction comes from
    # the colorbar + optional baseline-end marker.
    if len(df) >= 2:
        ax.plot(
            x,
            y,
            color="#9ca3af",
            linewidth=1.0,
            alpha=0.7,
            zorder=1,
        )

    scatter = ax.scatter(
        x,
        y,
        c=time_idx,
        cmap="viridis",
        s=sizes,
        zorder=3,
        edgecolors="black",
        linewidths=0.4,
    )

    if baseline_end is not None and len(df) > 0:
        baseline_ts = pd.to_datetime(baseline_end)
        post = df["window_start"] > baseline_ts
        if post.any():
            first_post = int(np.argmax(post.to_numpy()))
            # Highlight the first post-baseline point with a thicker ring.
            ax.scatter(
                [x[first_post]],
                [y[first_post]],
                s=float(sizes[first_post])
                if hasattr(sizes, "__len__")
                else float(sizes),
                facecolors="none",
                edgecolors="black",
                linewidths=1.8,
                zorder=4,
            )
            ax.annotate(
                "post-baseline start",
                xy=(x[first_post], y[first_post]),
                xytext=(6, 6),
                textcoords="offset points",
                fontsize=8,
                color="black",
            )

    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.grid(True, linestyle=":", linewidth=0.5, color="#bdbdbd")
    ax.set_axisbelow(True)
    ax.set_title(title or "PCA centroid drift")

    if len(df) >= 2:
        cbar = fig.colorbar(scatter, ax=ax, fraction=0.04, pad=0.02)
        cbar.set_label("Window index (early → late)")
        n = len(df)
        n_ticks = min(5, n)
        tick_positions = np.linspace(0, n - 1, n_ticks).astype(int)
        cbar.set_ticks(tick_positions.tolist())
        cbar.set_ticklabels(
            [
                df["window_start"].iloc[int(t)].strftime("%Y-%m-%d")
                for t in tick_positions
            ]
        )

    show_with_warning(show)
    return fig, ax


def plot_peak_stability(
    stability_df: pd.DataFrame,
    *,
    drug: str | None = None,
    threshold: float | None = 0.5,
    ax: plt.Axes | None = None,
    title: str | None = None,
    figsize: tuple[float, float] = (10, 4),
    show: bool = True,
) -> tuple[plt.Figure, plt.Axes]:
    """Line plot of peak-selection Jaccard stability over time.

    Parameters
    ----------
    stability_df : pd.DataFrame
        Output of :meth:`DriftMonitor.monitor_peak_stability`.  Must
        contain ``window_start`` and ``stability_score`` columns.
    drug : str, optional
        Drug name appended to the default title.
    threshold : float or None, default=0.5
        Horizontal dashed line at this Jaccard value (conventional
        "still-stable" cut-off).  Pass ``None`` to omit.
    ax : Axes or None, default=None
        Pre-existing axes.
    title : str or None, default=None
        Plot title.  Defaults to ``"Peak stability"`` (optionally
        ``f"Peak stability - {drug}"``).
    figsize : tuple of float, default=(10, 4)
        Figure size in inches.
    show : bool, default=True
        Whether to call ``plt.show()``.

    Returns
    -------
    fig : matplotlib.figure.Figure
    ax : matplotlib.axes.Axes
    """
    import matplotlib.pyplot as plt

    required = {"window_start", "stability_score"}
    missing = required - set(stability_df.columns)
    if missing:
        raise ValueError(f"'stability_df' is missing columns: {sorted(missing)}")

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()

    x = pd.to_datetime(stability_df["window_start"])
    y = stability_df["stability_score"].to_numpy(dtype=float)
    ax.plot(x, y, marker="o", color="#111827", linewidth=1.5)

    if threshold is not None:
        ax.axhline(
            threshold,
            color="#ef4444",
            linestyle="--",
            linewidth=0.8,
            label=f"threshold (Jaccard={threshold:g})",
        )
        ax.legend(loc="best", frameon=False, fontsize=8)

    ax.set_ylim(-0.05, 1.05)
    ax.set_xlabel("Window start")
    ax.set_ylabel("Jaccard stability")
    ax.grid(True, linestyle=":", linewidth=0.5, color="#bdbdbd")
    ax.set_axisbelow(True)
    if title is None:
        title = "Peak stability" + (f" - {drug}" if drug else "")
    ax.set_title(title)
    fig.autofmt_xdate()

    show_with_warning(show)
    return fig, ax


def plot_effect_size_drift(
    effect_df: pd.DataFrame,
    peaks: list[str] | None = None,
    *,
    drug: str | None = None,
    ax: plt.Axes | None = None,
    title: str | None = None,
    figsize: tuple[float, float] = (10, 4),
    legend_loc: str = "best",
    reference_lines: bool = True,
    show: bool = True,
) -> tuple[plt.Figure, plt.Axes]:
    """Multi-line plot of per-peak Cohen's d over time.

    Parameters
    ----------
    effect_df : pd.DataFrame
        Output of :meth:`DriftMonitor.monitor_effect_sizes`.  Must
        contain a ``window_start`` column plus one column per tracked
        peak.
    peaks : list of str or None, default=None
        Subset of peak columns to plot.  ``None`` plots every peak
        column present in ``effect_df``.
    drug : str, optional
        Drug name appended to the default title.
    ax : Axes or None, default=None
        Pre-existing axes.
    title : str or None, default=None
        Plot title.  Defaults to ``"Effect size drift"`` (optionally
        ``f"Effect size drift - {drug}"``).
    figsize : tuple of float, default=(10, 4)
        Figure size in inches.
    legend_loc : str, default="best"
        ``matplotlib`` legend location or ``"outside"`` to place the
        legend to the right of the axes (useful for many peaks).
    reference_lines : bool, default=True
        Draw dashed guides at Cohen's d = ±0.5 (medium effect) and
        ±0.8 (large effect).
    show : bool, default=True
        Whether to call ``plt.show()``.

    Returns
    -------
    fig : matplotlib.figure.Figure
    ax : matplotlib.axes.Axes
    """
    import matplotlib.pyplot as plt

    if "window_start" not in effect_df.columns:
        raise ValueError("'effect_df' must have a 'window_start' column")

    peak_cols = [c for c in effect_df.columns if c != "window_start"]
    if peaks is None:
        selected = peak_cols
    else:
        missing = [p for p in peaks if p not in peak_cols]
        if missing:
            raise ValueError(
                f"Peaks not found in effect_df columns: {missing[:5]}"
                + ("..." if len(missing) > 5 else "")
            )
        selected = list(peaks)

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()

    x = pd.to_datetime(effect_df["window_start"])
    for peak in selected:
        ax.plot(
            x,
            effect_df[peak].to_numpy(dtype=float),
            marker="o",
            linewidth=1.2,
            label=str(peak),
        )
    ax.axhline(0.0, color="#9ca3af", linewidth=0.8, linestyle="--")

    if reference_lines:
        for level in (0.5, 0.8):
            for sign in (+1, -1):
                ax.axhline(
                    sign * level,
                    color="#d1d5db",
                    linewidth=0.6,
                    linestyle=":",
                )

    ax.set_xlabel("Window start")
    ax.set_ylabel("Cohen's d (R vs S)")
    ax.grid(True, linestyle=":", linewidth=0.5, color="#bdbdbd")
    ax.set_axisbelow(True)
    if title is None:
        title = "Effect size drift" + (f" - {drug}" if drug else "")
    ax.set_title(title)

    if selected:
        legend_kwargs: dict = {
            "frameon": False,
            "fontsize": 8,
            "ncols": min(3, len(selected)),
        }
        if legend_loc == "outside":
            legend_kwargs.update(
                loc="upper left",
                bbox_to_anchor=(1.02, 1.0),
                borderaxespad=0.0,
                ncols=1,
            )
        else:
            legend_kwargs["loc"] = legend_loc
        ax.legend(**legend_kwargs)

    fig.autofmt_xdate()

    show_with_warning(show)
    return fig, ax
