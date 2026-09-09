"""Distances between sources and a reference, by group: lines, bars, and MANOVA effect sizes."""
from math import ceil
from typing import Dict, Sequence, Union

import matplotlib.pyplot as plt
import pandas as pd
import plotly.graph_objects as go
import seaborn as sns
from plotly.subplots import make_subplots

from .style import colors


def distance_lines(
    source_to_metrics: Dict[str, pd.DataFrame],
    group_order: Sequence[str],
    metrics: Sequence[str],
    total: Union[str, None] = None,
    ncols: int = 2,
    height_per_row: int = 350,
    template: str = "plotly_white",
    figure_width: Union[int, None] = None,
    col_width: int = 520,
    title: str = "Distance per metric by group",
) -> go.Figure:
    """
    One subplot per metric; in each, one line per source over the groups in `group_order`.

    Args:
        source_to_metrics: source -> distance table indexed by group (`compare.distance_table`).
        group_order: groups along the x axis, in order.
        metrics: which columns to plot, one subplot each.
        total: the group standing for the whole; the first source's value there is drawn as a dashed line.
    """
    nrows = ceil(len(metrics) / max(1, ncols))
    fig = make_subplots(rows=nrows, cols=ncols, subplot_titles=[str(metric) for metric in metrics], horizontal_spacing=0.08, vertical_spacing=0.12)
    sources = list(source_to_metrics)
    palette = colors(sources)
    labels = [str(group) for group in group_order]
    for i, metric in enumerate(metrics):
        row, col = i // ncols + 1, i % ncols + 1
        for source in sources:
            series = source_to_metrics[source][metric].reindex(group_order).replace([float("inf"), float("-inf")], pd.NA)
            fig.add_trace(go.Scatter(
                x=labels, y=series.values, mode="lines+markers", name=source, line=dict(width=2, color=palette[source]),
                hovertemplate="Group=%{x}<br>" + f"{metric}: " + "%{y:.6g}<br>" + f"Source={source}<extra></extra>",
                legendgroup=source, showlegend=(i == 0),
            ), row=row, col=col)
        if total is not None and total in source_to_metrics[sources[0]].index:
            baseline = source_to_metrics[sources[0]].loc[total, metric]
            if pd.notna(baseline):
                fig.add_hline(y=float(baseline), line_dash="dash", line_color="#C44E52", line_width=1.5, row=row, col=col)
        fig.update_xaxes(categoryorder="array", categoryarray=labels, tickangle=45, row=row, col=col)
        fig.update_yaxes(title_text="Distance", row=row, col=col)
    for j in range(len(metrics), nrows * ncols):
        fig.update_xaxes(visible=False, row=j // ncols + 1, col=j % ncols + 1)
        fig.update_yaxes(visible=False, row=j // ncols + 1, col=j % ncols + 1)
    fig.update_layout(
        height=max(350, nrows * height_per_row), width=figure_width or max(800, ncols * col_width), title_text=title,
        showlegend=True, hovermode="x unified", template=template, legend_title_text="Source", margin=dict(l=60, r=20, t=60, b=60),
    )
    return fig


def distance_bars(
    source_to_metrics: Dict[str, pd.DataFrame],
    group: str,
    metrics: Sequence[str],
    template: str = "plotly_white",
    figure_width: int = 1000,
    figure_height: int = 520,
    scale: float = 100.0,
    title: Union[str, None] = None,
) -> go.Figure:
    """
    For one group, one bar per source and metric, annotated with the value times `scale`.

    Raises:
        ValueError: if a source's table lacks the group.
    """
    missing = [source for source, table in source_to_metrics.items() if group not in table.index]
    if missing:
        raise ValueError(f"group '{group}' missing from the tables of {missing}")
    sources = list(source_to_metrics)
    palette = colors(sources)
    fig = go.Figure()
    annotations = []
    for source in sources:
        values = [float(source_to_metrics[source].loc[group, metric]) for metric in metrics]
        fig.add_bar(name=source, x=list(metrics), y=values, marker_color=palette[source])
        annotations += [dict(x=metric, y=value, text=f"{value * scale:.1f}", showarrow=False, xanchor="center", yanchor="bottom", font=dict(size=11))
                        for metric, value in zip(metrics, values)]
    fig.update_layout(
        title_text=title or f"Distances for {group} by source", barmode="group", template=template, width=figure_width,
        height=figure_height, margin=dict(l=60, r=20, t=60, b=60), legend_title_text="Source", annotations=annotations,
    )
    fig.update_xaxes(title_text="Metric")
    fig.update_yaxes(title_text="Distance")
    return fig


def effects_bars(effects: pd.DataFrame, stat: str = "Pillai's trace", figsize=(10, 6)) -> plt.Figure:
    """
    MANOVA effect sizes as grouped bars: one group per predictor, one bar per model.

    Raises:
        ValueError: if the statistic is not a column of `effects`.
    """
    if stat not in effects.columns:
        raise ValueError(f"statistic {stat!r} not in {list(effects.columns)}")
    fig, ax = plt.subplots(figsize=figsize)
    sns.barplot(data=effects.reset_index(), x="predictor", y=stat, hue="model", dodge=True, ax=ax)
    ax.tick_params(axis="x", rotation=45)
    plt.setp(ax.get_xticklabels(), ha="right")
    ax.set_ylabel(stat)
    ax.set_xlabel("Predictor")
    ax.set_title(f"{stat} by model and predictor")
    fig.tight_layout()
    return fig


def effects_heatmap(effects: pd.DataFrame, stat: str = "Wilks' lambda", figsize=(10, 6)) -> plt.Figure:
    """
    MANOVA effect sizes as a models x predictors matrix.

    Raises:
        ValueError: if the statistic is not a column of `effects`.
    """
    if stat not in effects.columns:
        raise ValueError(f"statistic {stat!r} not in {list(effects.columns)}")
    fig, ax = plt.subplots(figsize=figsize)
    pivot = effects.reset_index().pivot(index="model", columns="predictor", values=stat)
    sns.heatmap(pivot, annot=True, fmt=".3f", cmap="viridis", cbar_kws={"label": stat}, ax=ax)
    ax.set_ylabel("Model")
    ax.set_xlabel("Predictor")
    ax.set_title(f"{stat} (MANOVA) across models and predictors")
    fig.tight_layout()
    return fig
