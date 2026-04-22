"""Visualizations for temporal drift monitoring."""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    import matplotlib.pyplot as plt


def _show_with_warning(show: bool) -> None:
    """Call ``plt.show()`` with a backend-compatibility warning.

    Mirrors the pattern used in other ``maldiamrkit`` plot modules.
    """
    import matplotlib
    import matplotlib.pyplot as plt

    if show:
        if not matplotlib.is_interactive():
            warnings.warn(
                "matplotlib is using a non-interactive backend; "
                "plt.show() may not display the figure",
                UserWarning,
                stacklevel=3,
            )
        plt.show()


def plot_reference_drift(
    monitoring_df: pd.DataFrame,
    *,
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
    ax : Axes or None, default=None
        Pre-existing axes.
    title : str or None, default=None
        Optional plot title.
    figsize : tuple of float, default=(10, 4)
        Figure size in inches (used only when ``ax`` is ``None``).
    show : bool, default=True
        Whether to call :func:`matplotlib.pyplot.show`.

    Returns
    -------
    tuple[Figure, Axes]
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
    ax.set_xlabel("Window start")
    ax.set_ylabel("Distance to reference")
    ax.grid(True, linestyle=":", linewidth=0.5, color="#bdbdbd")
    ax.set_axisbelow(True)
    if title is not None:
        ax.set_title(title)
    fig.autofmt_xdate()

    _show_with_warning(show)
    return fig, ax


def plot_pca_drift(
    pca_df: pd.DataFrame,
    *,
    ax: plt.Axes | None = None,
    title: str | None = None,
    figsize: tuple[float, float] = (8, 6),
    show: bool = True,
) -> tuple[plt.Figure, plt.Axes]:
    """PCA centroid trajectory colored by time, with arrows between windows.

    Marker size encodes per-window dispersion (mean distance from
    centroid) when the ``dispersion`` column is present.

    Parameters
    ----------
    pca_df : pd.DataFrame
        Output of :meth:`DriftMonitor.monitor_pca`.  Must contain
        ``window_start``, ``centroid_pc1``, and ``centroid_pc2`` columns;
        ``dispersion`` is used for marker sizing when available.
    ax : Axes or None, default=None
        Pre-existing axes.  If ``None``, a new figure and axes are
        created.
    title : str or None, default=None
        Optional plot title.
    figsize : tuple of float, default=(8, 6)
        Figure size in inches (used only when ``ax`` is ``None``).
    show : bool, default=True
        Whether to call :func:`matplotlib.pyplot.show`.

    Returns
    -------
    tuple[Figure, Axes]
    """
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyArrowPatch

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

    for i in range(len(df) - 1):
        arrow = FancyArrowPatch(
            (x[i], y[i]),
            (x[i + 1], y[i + 1]),
            arrowstyle="->",
            mutation_scale=12,
            color="#6b7280",
            linewidth=1.0,
            zorder=2,
        )
        ax.add_patch(arrow)

    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.grid(True, linestyle=":", linewidth=0.5, color="#bdbdbd")
    ax.set_axisbelow(True)
    if title is not None:
        ax.set_title(title)

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

    _show_with_warning(show)
    return fig, ax


def plot_peak_stability(
    stability_df: pd.DataFrame,
    *,
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
    ax : Axes or None, default=None
        Pre-existing axes.  If ``None``, a new figure and axes are
        created.
    title : str or None, default=None
        Optional plot title.
    figsize : tuple of float, default=(10, 4)
        Figure size in inches (used only when ``ax`` is ``None``).
    show : bool, default=True
        Whether to call :func:`matplotlib.pyplot.show`.

    Returns
    -------
    tuple[Figure, Axes]
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
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlabel("Window start")
    ax.set_ylabel("Jaccard stability")
    ax.grid(True, linestyle=":", linewidth=0.5, color="#bdbdbd")
    ax.set_axisbelow(True)
    if title is not None:
        ax.set_title(title)
    fig.autofmt_xdate()

    _show_with_warning(show)
    return fig, ax


def plot_effect_size_drift(
    effect_df: pd.DataFrame,
    peaks: list[str] | None = None,
    *,
    ax: plt.Axes | None = None,
    title: str | None = None,
    figsize: tuple[float, float] = (10, 4),
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
    ax : Axes or None, default=None
        Pre-existing axes.  If ``None``, a new figure and axes are
        created.
    title : str or None, default=None
        Optional plot title.
    figsize : tuple of float, default=(10, 4)
        Figure size in inches (used only when ``ax`` is ``None``).
    show : bool, default=True
        Whether to call :func:`matplotlib.pyplot.show`.

    Returns
    -------
    tuple[Figure, Axes]
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
    ax.set_xlabel("Window start")
    ax.set_ylabel("Cohen's d (R vs S)")
    ax.grid(True, linestyle=":", linewidth=0.5, color="#bdbdbd")
    ax.set_axisbelow(True)
    if title is not None:
        ax.set_title(title)
    if selected:
        ax.legend(loc="best", frameon=False, fontsize=8, ncols=min(3, len(selected)))
    fig.autofmt_xdate()

    _show_with_warning(show)
    return fig, ax
