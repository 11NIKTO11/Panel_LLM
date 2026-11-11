"""Numbers with an interval each: one framed panel per measure, one row per thing measured."""
from typing import Mapping, Sequence, Union

import matplotlib.ticker as mtick
import numpy as np
import pandas as pd
from matplotlib.figure import Figure

from .style import EDGE, INK, MUTED, PANEL, SERIES, SURFACE, clean


def interval_panels(
    table: pd.DataFrame,
    columns: Sequence[str],
    percent: Sequence[str] = (),
    zero: Sequence[str] = (),
    names: Union[Mapping[str, str], None] = None,
    same_scale: bool = False,
    per_row: Union[int, None] = None,
    unit: str = "pp",
    marks: Union[Mapping[str, Sequence[str]], None] = None,
    title: Union[str, None] = None,
    note: Union[str, None] = None,
) -> Figure:
    """
    One panel per column, in a grid; in each, a dot per row of the table with its interval.

    Args:
        table: indexed by what is measured; a column per measure, and `<column>_low`,
            `<column>_high` where there is an interval.
        columns: the measures, one panel each, left to right, then down.
        percent: columns holding shares, shown times 100, the title saying `unit`.
        zero: columns that are signed, drawn with a line at zero.
        names: column -> panel title; the column itself by default.
        same_scale: one value axis for all panels, for measures in the same unit; each
            panel has its own otherwise.
        per_row: panels in a row of the grid; all in one row by default.
        unit: what a `percent` column is in: percentage points, or `%` for a share of a whole.
        marks: column -> a text per row, written under the row's value (e.g. a p-value).
        title: figure title.
        note: a line under the title.
    """
    names, marks = names or {}, marks or {}
    across = min(per_row or len(columns), len(columns))
    down = -(-len(columns) // across)
    height = 1.0 + down * (0.6 + (0.58 if marks else 0.42) * len(table))
    fig = Figure(figsize=(2.5 * across + 2.2, height), facecolor=SURFACE)
    grid = fig.subplots(down, across, sharey=True, sharex=same_scale, squeeze=False)
    axes = grid.ravel()
    for ax in axes[len(columns):]:
        ax.set_visible(False)
    y = np.arange(len(table))
    for ax, column in zip(axes, columns):
        clean(ax)
        ax.set_facecolor(PANEL)
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_color(EDGE)
        scale = 100 if column in percent else 1
        values = table[column].to_numpy(dtype=float) * scale
        if f"{column}_low" in table.columns:
            ax.hlines(y, table[f"{column}_low"] * scale, table[f"{column}_high"] * scale, color=SERIES[0], linewidth=2, alpha=0.45)
        ax.scatter(values, y, s=42, color=SERIES[0], zorder=3)
        if column in zero:
            ax.axvline(0, color=MUTED, linewidth=0.8)
        for at, value, mark in zip(y, values, marks.get(column, [""] * len(y))):
            if np.isnan(value):
                continue
            text = f"{value:+.1f}" if column in zero else f"{value:.1f}" if column in percent else f"{value:.3f}"
            ax.annotate("\n".join(part for part in (text, mark) if part), (value, at), linespacing=1.1, xytext=(0, 8), textcoords="offset points", ha="center", fontsize=8, color=MUTED)
        ax.set_title(names.get(column, column) + (f" ({unit})" if column in percent else ""), fontsize=10, color=INK, loc="left", fontweight="bold")
        ax.xaxis.set_major_locator(mtick.MaxNLocator(4))
        ax.margins(x=0.15)
    for ax in grid[:, 0]:
        ax.set_yticks(y, [str(row) for row in table.index])
    axes[0].set_ylim(len(table) - 0.5, -0.6)
    if title:
        fig.suptitle(title, x=0.02, ha="left", fontsize=13, color=INK, fontweight="bold")
    if note:
        fig.text(0.02, 1 - 0.42 / height, note, ha="left", va="top", fontsize=9, color=MUTED)
    fig.tight_layout(rect=(0, 0, 1, 1 - (0.5 if note else 0.3) / height), w_pad=2.2, h_pad=1.6)
    return fig
