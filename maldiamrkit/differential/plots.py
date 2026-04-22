"""Visualizations for differential analysis."""

from __future__ import annotations

import warnings
from enum import Enum
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    import matplotlib.pyplot as plt


_VOLCANO_EPS = 1e-300


class DrugComparisonKind(str, Enum):
    """Rendering kind for :func:`plot_drug_comparison`.

    Attributes
    ----------
    heatmap : str
        Boolean ``rows x drugs`` heatmap (compact, precise positions).
    upset : str
        UpSet-style intersection plot: bar chart of intersection sizes
        plus a dot matrix of drug membership.
    """

    heatmap = "heatmap"
    upset = "upset"


def _show_with_warning(show: bool) -> None:
    """Call ``plt.show()`` with a backend-compatibility warning.

    Mirrors the pattern used across the other ``maldiamrkit`` plot
    modules so that headless test runs and notebooks behave consistently.
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


def plot_volcano(
    results: pd.DataFrame,
    fc_threshold: float = 1.0,
    p_threshold: float = 0.05,
    *,
    ax: plt.Axes | None = None,
    title: str | None = None,
    figsize: tuple[float, float] = (8, 6),
    show: bool = True,
) -> tuple[plt.Figure, plt.Axes]:
    r"""Volcano plot of log2 fold change vs. -log10 adjusted p-value.

    Points are coloured by direction and significance: grey for
    non-significant, red for up in resistant (``fold_change > fc_threshold``
    and ``adjusted_p_value <= p_threshold``), blue for up in susceptible
    (``fold_change < -fc_threshold`` and ``adjusted_p_value <= p_threshold``).
    Horizontal and vertical dashed lines mark the thresholds.

    Parameters
    ----------
    results : pd.DataFrame
        Output of :attr:`DifferentialAnalysis.results`.  Must contain
        ``fold_change`` and ``adjusted_p_value`` columns.
    fc_threshold : float, default=1.0
        Absolute log2 fold-change threshold (drawn as vertical dashed
        lines at :math:`\pm` ``fc_threshold``).
    p_threshold : float, default=0.05
        Adjusted p-value threshold (drawn as a horizontal dashed line at
        ``-log10(p_threshold)``).
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

    required = {"fold_change", "adjusted_p_value"}
    missing = required - set(results.columns)
    if missing:
        raise ValueError(f"'results' is missing required columns: {sorted(missing)}")

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()

    fc = results["fold_change"].to_numpy(dtype=float)
    adj_p = results["adjusted_p_value"].to_numpy(dtype=float)
    neg_log10_p = -np.log10(np.clip(adj_p, _VOLCANO_EPS, 1.0))

    sig = adj_p <= p_threshold
    up_r = sig & (fc > fc_threshold)
    up_s = sig & (fc < -fc_threshold)
    ns = ~(up_r | up_s)

    ax.scatter(fc[ns], neg_log10_p[ns], s=10, color="lightgrey", label="NS", alpha=0.6)
    ax.scatter(
        fc[up_s], neg_log10_p[up_s], s=14, color="#3b82f6", label="Up in S", alpha=0.85
    )
    ax.scatter(
        fc[up_r], neg_log10_p[up_r], s=14, color="#ef4444", label="Up in R", alpha=0.85
    )

    ax.axhline(-np.log10(p_threshold), color="black", linestyle="--", linewidth=0.8)
    ax.axvline(fc_threshold, color="black", linestyle="--", linewidth=0.8)
    ax.axvline(-fc_threshold, color="black", linestyle="--", linewidth=0.8)

    ax.set_xlabel("log2 fold change (R / S)")
    ax.set_ylabel(r"$-\log_{10}$(adjusted p-value)")
    if title is not None:
        ax.set_title(title)
    ax.legend(loc="best", frameon=False)

    _show_with_warning(show)
    return fig, ax


def plot_manhattan(
    results: pd.DataFrame,
    p_threshold: float = 0.05,
    *,
    ax: plt.Axes | None = None,
    title: str | None = None,
    figsize: tuple[float, float] = (12, 4),
    show: bool = True,
) -> tuple[plt.Figure, plt.Axes]:
    """Manhattan plot along the m/z axis.

    x-axis is the numeric m/z bin value; y-axis is
    ``-log10(adjusted_p_value)``.  Points with
    ``adjusted_p_value <= p_threshold`` are highlighted in red.  A
    horizontal dashed line marks the threshold.

    Parameters
    ----------
    results : pd.DataFrame
        Output of :attr:`DifferentialAnalysis.results`.  Must contain
        ``mz_bin`` and ``adjusted_p_value`` columns.  ``mz_bin`` values
        that cannot be coerced to float are excluded.
    p_threshold : float, default=0.05
        Adjusted p-value threshold.
    ax : Axes or None, default=None
        Pre-existing axes.  If ``None``, a new figure and axes are
        created.
    title : str or None, default=None
        Optional plot title.
    figsize : tuple of float, default=(12, 4)
        Figure size in inches (used only when ``ax`` is ``None``).
    show : bool, default=True
        Whether to call :func:`matplotlib.pyplot.show`.

    Returns
    -------
    tuple[Figure, Axes]
    """
    import matplotlib.pyplot as plt

    required = {"mz_bin", "adjusted_p_value"}
    missing = required - set(results.columns)
    if missing:
        raise ValueError(f"'results' is missing required columns: {sorted(missing)}")

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()

    mz_values = pd.to_numeric(results["mz_bin"], errors="coerce").to_numpy()
    adj_p = results["adjusted_p_value"].to_numpy(dtype=float)
    valid = ~np.isnan(mz_values)
    mz_values = mz_values[valid]
    adj_p = adj_p[valid]

    neg_log10_p = -np.log10(np.clip(adj_p, _VOLCANO_EPS, 1.0))
    sig = adj_p <= p_threshold

    ax.scatter(
        mz_values[~sig],
        neg_log10_p[~sig],
        s=8,
        color="#6b7280",
        alpha=0.6,
        label="NS",
    )
    ax.scatter(
        mz_values[sig],
        neg_log10_p[sig],
        s=14,
        color="#ef4444",
        alpha=0.9,
        label="Significant",
    )

    ax.axhline(-np.log10(p_threshold), color="black", linestyle="--", linewidth=0.8)

    ax.set_xlabel("m/z")
    ax.set_ylabel(r"$-\log_{10}$(adjusted p-value)")
    if title is not None:
        ax.set_title(title)
    ax.legend(loc="best", frameon=False)

    _show_with_warning(show)
    return fig, ax


def _plot_drug_comparison_heatmap(
    comparison_df: pd.DataFrame,
    ax: plt.Axes | None,
    title: str | None,
    figsize: tuple[float, float],
) -> tuple[plt.Figure, plt.Axes]:
    """Render the boolean comparison as a seaborn binary heatmap."""
    import matplotlib.pyplot as plt
    import seaborn as sns

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()

    data = comparison_df.astype(int)
    sns.heatmap(
        data,
        ax=ax,
        cmap=["#f3f4f6", "#ef4444"],
        cbar=False,
        linewidths=0.25,
        linecolor="white",
        xticklabels=True,
        yticklabels=data.shape[0] <= 60,
    )
    ax.set_xlabel("Drug")
    ax.set_ylabel("m/z bin")
    if title is not None:
        ax.set_title(title)
    return fig, ax


def _plot_drug_comparison_upset(
    comparison_df: pd.DataFrame,
    title: str | None,
    figsize: tuple[float, float],
) -> tuple[plt.Figure, plt.Axes]:
    """Render an UpSet-style intersection plot using matplotlib only.

    Layout follows the ``UpSetPlot`` convention (Nothman, 2018):

    - top-right: intersection-size bar chart
    - bottom-left: per-set totals bar chart (horizontal, right-to-left)
    - bottom-right: dot matrix of set membership, with alternating row
      shading for readability

    The returned Axes is the intersection-size bar chart at the top.
    """
    import matplotlib.pyplot as plt

    bool_df = comparison_df.astype(bool)
    drugs = list(bool_df.columns)

    if bool_df.empty or not drugs:
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(
            0.5,
            0.5,
            "No significant peaks",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
        ax.set_axis_off()
        if title is not None:
            ax.set_title(title)
        return fig, ax

    signatures = bool_df.apply(lambda row: tuple(row.values), axis=1)
    sig_counts = signatures.value_counts()
    all_false = tuple(False for _ in drugs)
    if all_false in sig_counts.index:
        sig_counts = sig_counts.drop(all_false)

    sig_counts = sig_counts.sort_values(ascending=False)
    combos = list(sig_counts.index)
    counts = sig_counts.to_numpy()

    set_totals = bool_df.sum(axis=0).to_numpy()

    # Palette close to UpSetPlot defaults
    dot_on = "#1f1f1f"
    dot_off = "#d9d9d9"
    bar_color = "#1f1f1f"
    row_shade = "#f2f2f2"

    fig = plt.figure(figsize=figsize)
    matrix_height = max(1.5, 0.45 * len(drugs))
    gs = fig.add_gridspec(
        nrows=2,
        ncols=2,
        height_ratios=[3.0, matrix_height],
        width_ratios=[1.0, max(3.0, 0.55 * len(combos))],
        hspace=0.06,
        wspace=0.04,
    )
    ax_bars = fig.add_subplot(gs[0, 1])
    ax_totals = fig.add_subplot(gs[1, 0])
    ax_matrix = fig.add_subplot(gs[1, 1], sharex=ax_bars, sharey=ax_totals)

    ax_corner = fig.add_subplot(gs[0, 0])
    ax_corner.set_axis_off()

    x = np.arange(len(combos))
    ax_bars.bar(x, counts, color=bar_color, width=0.55, zorder=2)
    for xi, ci in zip(x, counts, strict=True):
        ax_bars.text(
            float(xi),
            float(ci),
            str(int(ci)),
            ha="center",
            va="bottom",
            fontsize=9,
            color=bar_color,
        )
    ax_bars.set_ylabel("Intersection size")
    ax_bars.grid(axis="y", linestyle=":", linewidth=0.6, color="#bdbdbd", zorder=0)
    ax_bars.set_axisbelow(True)
    for spine in ("top", "right"):
        ax_bars.spines[spine].set_visible(False)
    ax_bars.tick_params(axis="x", bottom=False, labelbottom=False)
    if title is not None:
        ax_bars.set_title(title)

    y_pos = np.arange(len(drugs))
    ax_totals.barh(y_pos, set_totals, color=bar_color, height=0.55, zorder=2)
    ax_totals.set_xlabel("Set size")
    ax_totals.invert_xaxis()
    ax_totals.set_yticks(y_pos)
    ax_totals.set_yticklabels(drugs)
    ax_totals.grid(axis="x", linestyle=":", linewidth=0.6, color="#bdbdbd", zorder=0)
    ax_totals.set_axisbelow(True)
    for spine in ("top", "right", "left"):
        ax_totals.spines[spine].set_visible(False)
    ax_totals.tick_params(axis="y", left=False)

    for yi in range(len(drugs)):
        if yi % 2 == 0:
            ax_matrix.axhspan(yi - 0.5, yi + 0.5, color=row_shade, zorder=0)

    for col_idx, combo in enumerate(combos):
        col_x: float = float(col_idx)
        members: list[float] = [float(yi) for yi, flag in enumerate(combo) if flag]
        non_members: list[float] = [
            float(yi) for yi in range(len(drugs)) if float(yi) not in members
        ]
        if non_members:
            ax_matrix.scatter(
                [col_x] * len(non_members),
                non_members,
                s=80,
                color=dot_off,
                edgecolors="none",
                zorder=1,
            )
        if members:
            ax_matrix.scatter(
                [col_x] * len(members),
                members,
                s=80,
                color=dot_on,
                edgecolors="none",
                zorder=3,
            )
        if len(members) > 1:
            ax_matrix.plot(
                [col_x, col_x],
                [min(members), max(members)],
                color=dot_on,
                linewidth=1.8,
                zorder=2,
            )

    ax_matrix.set_xticks(x)
    ax_matrix.set_xticklabels([])
    ax_matrix.set_xlim(-0.5, len(combos) - 0.5)
    ax_matrix.set_ylim(-0.5, len(drugs) - 0.5)
    ax_matrix.invert_yaxis()
    for spine in ("top", "right", "bottom", "left"):
        ax_matrix.spines[spine].set_visible(False)
    ax_matrix.tick_params(axis="x", bottom=False)
    ax_matrix.tick_params(axis="y", left=False, labelleft=False)

    return fig, ax_bars


def plot_drug_comparison(
    comparison_df: pd.DataFrame,
    *,
    kind: str | DrugComparisonKind = DrugComparisonKind.heatmap,
    ax: plt.Axes | None = None,
    title: str | None = None,
    figsize: tuple[float, float] = (10, 8),
    show: bool = True,
) -> tuple[plt.Figure, plt.Axes]:
    """Visualise a multi-drug differential-peak comparison matrix.

    Parameters
    ----------
    comparison_df : pd.DataFrame
        Boolean significance matrix from
        :meth:`DifferentialAnalysis.compare_drugs`.  Index = m/z bins,
        columns = drug names, values coerced to ``bool``.
    kind : {"heatmap", "upset"} or DrugComparisonKind, default="heatmap"
        Rendering style.

        - ``"heatmap"``: compact binary heatmap of peaks x drugs.
        - ``"upset"``: UpSet-style plot showing intersection counts
          across drug combinations.
    ax : Axes or None, default=None
        Pre-existing axes (used only by ``kind="heatmap"``; ignored for
        ``"upset"`` which needs its own composite figure).
    title : str or None, default=None
        Optional plot title.
    figsize : tuple of float, default=(10, 8)
        Figure size in inches (used only when ``ax`` is ``None``).
    show : bool, default=True
        Whether to call :func:`matplotlib.pyplot.show`.

    Returns
    -------
    tuple[Figure, Axes]
        For ``kind="upset"``, the returned Axes is the intersection-size
        bar chart; the drug-membership matrix is drawn on a second
        Axes inside the same Figure.
    """
    kind = DrugComparisonKind(kind)

    if kind == DrugComparisonKind.heatmap:
        fig, ax = _plot_drug_comparison_heatmap(
            comparison_df, ax=ax, title=title, figsize=figsize
        )
    else:
        if ax is not None:
            warnings.warn(
                "plot_drug_comparison(kind='upset') creates its own composite "
                "figure; the provided 'ax' is ignored.",
                UserWarning,
                stacklevel=2,
            )
        fig, ax = _plot_drug_comparison_upset(
            comparison_df, title=title, figsize=figsize
        )

    _show_with_warning(show)
    return fig, ax
