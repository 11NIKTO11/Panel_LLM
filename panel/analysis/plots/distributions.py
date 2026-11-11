"""How values are spread over subjects: a violin per column, and histograms side by side."""
from typing import Mapping, Sequence, Union

import matplotlib.ticker as mtick
import numpy as np
import pandas as pd
from matplotlib.figure import Figure

from .style import EDGE, INK, MUTED, NEUTRAL, PANEL, SERIES, SURFACE, clean, tint, wrap


def violins(
    values: pd.DataFrame,
    rows: Sequence[str],
    head: Sequence[str] = (),
    colors: Union[Mapping[str, str], None] = None,
    labels: Union[Mapping[str, str], None] = None,
    title: Union[str, None] = None,
    note: Union[str, None] = None,
) -> Figure:
    """
    One horizontal violin per column of shares, with the median as a dot and the middle
    half as a bar.

    Args:
        values: one row per subject, shares between 0 and 1, no missing values.
        rows: the columns of the main panel, top to bottom.
        head: columns drawn in a panel of their own above it, on their own scale (e.g. turnout).
        colors: column -> colour; grey for a column without one.
        labels: column -> display name; the column itself by default.
        title: figure title.
        note: a line under the title.
    """
    colors, labels = colors or {}, labels or {}
    panels = [list(part) for part in (head, rows) if len(part)]
    height = 1.5 + 0.5 * sum(len(part) for part in panels) + 0.4 * len(panels)
    fig = Figure(figsize=(9, height), facecolor=SURFACE)
    axes = fig.subplots(len(panels), 1, squeeze=False, gridspec_kw={"height_ratios": [len(part) for part in panels]})[:, 0]
    for ax, part in zip(axes, panels):
        clean(ax)
        positions = np.arange(len(part))
        bodies = ax.violinplot([values[column].to_numpy(dtype=float) for column in part], positions, orientation="horizontal", widths=0.85, showextrema=False)["bodies"]
        for body, column in zip(bodies, part):
            body.set_facecolor(tint(colors.get(column, NEUTRAL), 0.55))
            body.set_edgecolor(colors.get(column, NEUTRAL))
            body.set_alpha(1)
        low, middle, high = (values[part].quantile(q).to_numpy() for q in (0.25, 0.5, 0.75))
        ax.hlines(positions, low, high, color=INK, linewidth=2.5)
        ax.scatter(middle, positions, s=26, color=SURFACE, edgecolor=INK, linewidth=1.2, zorder=3)
        ax.set_yticks(positions, wrap(labels.get(column, column) for column in part))
        ax.set_ylim(len(part) - 0.4, -0.6)
        ax.set_xlim(0, max(float(values[part].max().max()) * 1.03, 0.05))
        ax.xaxis.set_major_formatter(mtick.PercentFormatter(1.0, decimals=0))
    if title:
        fig.suptitle(title, x=0.02, ha="left", fontsize=13, color=INK, fontweight="bold")
    if note:
        fig.text(0.02, 1 - 0.42 / height, note, ha="left", va="top", fontsize=9, color=MUTED)
    fig.tight_layout(rect=(0, 0, 1, 1 - (0.5 if note else 0.3) / height))
    return fig


def histograms(
    values: pd.DataFrame,
    ylabel: str,
    bins: int = 30,
    title: Union[str, None] = None,
    note: Union[str, None] = None,
) -> Figure:
    """
    One framed histogram per column, side by side, each on its own scale, the mean marked.

    Args:
        values: a column per panel; missing values are left out.
        ylabel: what is counted.
        bins: bars per histogram.
        title: figure title.
        note: a line under the title.
    """
    height = 4.2
    fig = Figure(figsize=(4.4 * values.shape[1] + 0.6, height), facecolor=SURFACE)
    axes = fig.subplots(1, values.shape[1], squeeze=False)[0]
    for ax, column in zip(axes, values.columns):
        clean(ax)
        ax.set_facecolor(PANEL)
        ax.grid(axis="x", visible=False)
        ax.grid(axis="y", color=SURFACE, linewidth=0.8)
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_color(EDGE)
        found = values[column].dropna().to_numpy(dtype=float)
        ax.hist(found, bins=bins, color=SERIES[0], edgecolor=PANEL, linewidth=0.6)
        ax.axvline(found.mean(), color=INK, linewidth=1)
        ax.annotate(f"mean {found.mean():.3f}", (found.mean(), 1), xycoords=("data", "axes fraction"), xytext=(4, -12), textcoords="offset points", fontsize=8, color=MUTED)
        ax.set_title(column, fontsize=10, color=INK, loc="left", fontweight="bold")
        ax.tick_params(axis="y", colors=MUTED, labelsize=9)
    axes[0].set_ylabel(ylabel, fontsize=9, color=MUTED)
    if title:
        fig.suptitle(title, x=0.02, ha="left", fontsize=13, color=INK, fontweight="bold")
    if note:
        fig.text(0.02, 1 - 0.42 / height, note, ha="left", va="top", fontsize=9, color=MUTED)
    fig.tight_layout(rect=(0, 0, 1, 1 - (0.5 if note else 0.3) / height), w_pad=2.2)
    return fig
