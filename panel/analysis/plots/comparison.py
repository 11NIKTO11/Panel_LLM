"""Several sources' distributions over the same rows, side by side."""
from typing import Mapping, Sequence, Union

import matplotlib.ticker as mtick
import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from matplotlib.patches import Patch

from .style import INK, MUTED, NEUTRAL, SURFACE, clean, tint, wrap


def source_bars(
    table: pd.DataFrame,
    sources: Sequence[str],
    rows: Sequence[str],
    colors: Mapping[str, str],
    head: Sequence[str] = (),
    row_colors: Union[Mapping[str, str], None] = None,
    labels: Union[Mapping[str, str], None] = None,
    names: Union[Mapping[str, str], None] = None,
    title: Union[str, None] = None,
    note: Union[str, None] = None,
) -> Figure:
    """
    One group of horizontal bars per row, one bar per source, shares as percentages.

    Args:
        table: indexed by row; a column per source, and optionally `<source>_low` and
            `<source>_high`, drawn as a whisker on that source's bar.
        sources: the columns drawn, in order from the top of each group.
        rows: the rows of the main panel, in display order.
        colors: source -> colour.
        head: rows drawn in a panel of their own above the main one (e.g. turnout).
        row_colors: row -> colour. When given, a bar wears its row's colour (grey for a row
            without one) and the sources differ by strength, the first the strongest;
            `colors` is then not used.
        labels: row -> display name; the row itself by default.
        names: source -> legend entry; the source itself by default.
        title: figure title.
        note: a line under the title.
    """
    labels, names = labels or {}, names or {}
    panels = [list(part) for part in (head, rows) if len(part)]
    shown = [row for part in panels for row in part]
    thickness = 0.8 / len(sources)
    strength = dict(zip(sources, np.linspace(1.0, 0.3, len(sources))))
    height = 1.5 + 0.21 * len(sources) * len(shown) + 0.4 * len(panels)
    fig = Figure(figsize=(9, height), facecolor=SURFACE)
    axes = fig.subplots(len(panels), 1, squeeze=False, gridspec_kw={"height_ratios": [len(part) for part in panels]})[:, 0]

    for ax, part in zip(axes, panels):
        clean(ax)
        top = float(table.loc[part].max().max())
        positions = np.arange(len(part))
        for i, source in enumerate(sources):
            values = table.loc[part, source].to_numpy(dtype=float)
            y = positions + (i - (len(sources) - 1) / 2) * thickness
            if row_colors:
                ax.barh(y, values, thickness * 0.88, color=[tint(row_colors.get(row, NEUTRAL), strength[source]) for row in part])
            else:
                ax.barh(y, values, thickness * 0.88, color=colors[source])
            ends = values
            if f"{source}_low" in table.columns:
                ends = table.loc[part, f"{source}_high"].to_numpy(dtype=float)
                ax.hlines(y, table.loc[part, f"{source}_low"].to_numpy(dtype=float), ends, color=INK, linewidth=1)
            for at, end, value in zip(y, ends, values):
                ax.text(end + top * 0.012, at, f"{value:.1%}", va="center", ha="left", fontsize=8, color=MUTED)
        ax.set_yticks(positions, wrap(labels.get(row, row) for row in part))
        ax.set_ylim(len(part) - 0.5, -0.5)
        ax.set_xlim(0, top * 1.1)
        ax.xaxis.set_major_formatter(mtick.PercentFormatter(1.0, decimals=0))

    handles = [Patch(color=tint(NEUTRAL, strength[source])) if row_colors else Patch(color=colors[source]) for source in sources]
    fig.legend(handles, [names.get(source, source) for source in sources], loc="lower center", ncols=len(sources), frameon=False, fontsize=10, bbox_to_anchor=(0.5, 0))
    if title:
        fig.suptitle(title, x=0.02, ha="left", fontsize=13, color=INK, fontweight="bold")
    if note:
        fig.text(0.02, 1 - 0.42 / height, note, ha="left", va="top", fontsize=9, color=MUTED)
    fig.tight_layout(rect=(0, 0.4 / height, 1, 1 - (0.5 if note else 0.3) / height))
    return fig
