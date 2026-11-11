"""Points on a plane beside where they were expected: one small map per panel."""
from typing import Mapping, Tuple, Union

import pandas as pd
from matplotlib.figure import Figure
from matplotlib.lines import Line2D

from .style import EDGE, INK, MUTED, NEUTRAL, PANEL, SURFACE, clean


def maps(
    panels: Mapping[str, pd.DataFrame],
    target: pd.DataFrame,
    limits: Tuple[float, float],
    xlabel: str,
    ylabel: str,
    names: Tuple[str, str],
    colors: Union[Mapping[str, str], None] = None,
    labels: Union[Mapping[str, str], None] = None,
    per_row: int = 2,
    width: float = 10.0,
    title: Union[str, None] = None,
    note: Union[str, None] = None,
) -> Figure:
    """
    One square map per panel: each point a dot where the panel puts it, a cross where the
    target has it, the two joined by a line.

    Args:
        panels: panel title -> two columns (x, y), indexed by point.
        target: the same points and columns, where they were expected.
        limits: both axes' range.
        xlabel, ylabel: of the axes.
        names: legend entries for the dot and the cross.
        colors: point -> colour; grey without one.
        labels: point -> the text beside it; the point itself by default.
        per_row: maps in a row; further maps go on further rows.
        width: of the figure, in inches, whatever the number of maps: one map fills it, two
            in a row are half the size each. Text keeps its size.
        title: figure title.
        note: a line under the title.
    """
    colors, labels = colors or {}, labels or {}
    across = min(per_row, len(panels))
    down = -(-len(panels) // across)
    height = 2.3 + down * ((width - 1.0) / across + 0.5)
    fig = Figure(figsize=(width, height), facecolor=SURFACE)
    axes = fig.subplots(down, across, squeeze=False, sharex=True, sharey=True).ravel()
    for ax in axes[len(panels):]:
        ax.set_visible(False)
    for ax, (name, points) in zip(axes, panels.items()):
        clean(ax)
        ax.set_facecolor(PANEL)
        ax.grid(axis="y", color=SURFACE, linewidth=0.8)
        ax.grid(axis="x", color=SURFACE, linewidth=0.8)
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_color(EDGE)
        for point, (x, y) in points.iterrows():
            color = colors.get(point, NEUTRAL)
            wanted = target.loc[point]
            ax.plot([x, wanted.iloc[0]], [y, wanted.iloc[1]], color=color, linewidth=1, alpha=0.5)
            ax.scatter(*wanted, s=90, marker="x", color=color, linewidth=2.2, zorder=3)
            ax.scatter(x, y, s=90, color=color, zorder=4)
            ax.annotate(labels.get(point, point), (x, y), xytext=(7, 6), textcoords="offset points", fontsize=10, color=INK)
        ax.set_xlim(*limits)
        ax.set_ylim(*limits)
        ax.set_aspect("equal")
        ax.set_title(name, fontsize=10, color=INK, loc="left", fontweight="bold")
        ax.tick_params(axis="y", colors=MUTED, labelsize=9)
    for ax in axes[:len(panels)][-across:]:
        ax.set_xlabel(xlabel, fontsize=9, color=MUTED)
    for ax in axes[:len(panels):across]:
        ax.set_ylabel(ylabel, fontsize=9, color=MUTED)
    fig.legend(
        [Line2D([], [], marker="o", linestyle="", color=NEUTRAL), Line2D([], [], marker="x", linestyle="", color=NEUTRAL, markeredgewidth=2.2)],
        list(names), loc="lower center", ncols=2, frameon=False, fontsize=10, bbox_to_anchor=(0.5, 0),
    )
    if title:
        fig.suptitle(title, x=0.02, ha="left", fontsize=13, color=INK, fontweight="bold")
    if note:
        fig.text(0.02, 1 - 0.42 / height, note, ha="left", va="top", fontsize=9, color=MUTED)
    fig.tight_layout(rect=(0, 0.95 / height, 1, 1 - (0.6 if note else 0.3) / height))
    return fig
