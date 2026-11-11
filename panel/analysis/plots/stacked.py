"""Compositions as stacked horizontal bars: one bar per row, one segment per part."""
from typing import Mapping, Union

import matplotlib.ticker as mtick
import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from matplotlib.patches import Patch

from .style import INK, MUTED, NEUTRAL, SERIES, SURFACE, clean, wrap


def stacked_bars(
    table: pd.DataFrame,
    colors: Union[Mapping[str, str], None] = None,
    labels: Union[Mapping[str, str], None] = None,
    least: float = 0.04,
    title: Union[str, None] = None,
    note: Union[str, None] = None,
) -> Figure:
    """
    One horizontal bar per row of the table, its columns stacked left to right.

    Args:
        table: indexed by what a bar stands for; a column per part, rows summing to about 1.
        colors: part -> colour; the default palette in column order without it.
        labels: part -> legend entry; the part itself by default.
        least: a segment at least this wide carries its percentage.
        title: figure title.
        note: a line under the title.

    Raises:
        ValueError: without colours, on more parts than the palette has.
    """
    labels = labels or {}
    if not colors:
        if table.shape[1] > len(SERIES):
            raise ValueError(f"{table.shape[1]} parts, the palette has {len(SERIES)} colours")
        colors = dict(zip(table.columns, SERIES))
    height = 1.9 + 0.42 * len(table)
    fig = Figure(figsize=(9, height), facecolor=SURFACE)
    ax = fig.subplots()
    clean(ax)
    ax.grid(visible=False)
    positions = np.arange(len(table))
    left = np.zeros(len(table))
    for part in table.columns:
        values = table[part].to_numpy(dtype=float)
        ax.barh(positions, values, 0.72, left=left, color=colors.get(part, NEUTRAL), edgecolor=SURFACE, linewidth=1)
        for at, start, value in zip(positions, left, values):
            if value >= least:
                ax.text(start + value / 2, at, f"{value:.0%}", ha="center", va="center", fontsize=8, color=SURFACE)
        left += values
    ax.set_yticks(positions, wrap(str(row) for row in table.index))
    ax.set_ylim(len(table) - 0.5, -0.5)
    ax.set_xlim(0, max(1.0, float(left.max())))
    ax.xaxis.set_major_formatter(mtick.PercentFormatter(1.0, decimals=0))
    fig.legend(
        [Patch(color=colors.get(part, NEUTRAL)) for part in table.columns], [labels.get(part, part) for part in table.columns],
        loc="lower center", ncols=min(len(table.columns), 9), frameon=False, fontsize=9, bbox_to_anchor=(0.5, 0), handlelength=1, columnspacing=1.2,
    )
    if title:
        fig.suptitle(title, x=0.02, ha="left", fontsize=13, color=INK, fontweight="bold")
    if note:
        fig.text(0.02, 1 - 0.42 / height, note, ha="left", va="top", fontsize=9, color=MUTED)
    fig.tight_layout(rect=(0, 0.45 / height, 1, 1 - (0.5 if note else 0.3) / height))
    return fig
