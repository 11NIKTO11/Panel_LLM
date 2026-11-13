"""Every plot function returns a figure for small inputs and leaves global state alone."""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from matplotlib.figure import Figure

from panel.analysis.plots import comparison, curves, distributions, intervals, maps, matrices, stacked
from panel.analysis.plots.style import ERRORS, colors, legible

OPTIONS = ["a", "b", "c"]
RNG = np.random.default_rng(0)


@pytest.fixture(autouse=True)
def global_style_untouched():
    before = dict(plt.rcParams)
    yield
    assert dict(plt.rcParams) == before, "a plot function changed global style"
    assert not plt.get_fignums(), "a plot function opened a pyplot figure"


def shares(rows: int) -> pd.DataFrame:
    return pd.DataFrame(RNG.dirichlet(np.ones(len(OPTIONS)), rows), columns=OPTIONS)


def test_source_bars():
    table = pd.DataFrame({"predicted": [0.5, 0.3, 0.2, 0.7], "actual": [0.4, 0.4, 0.2, 0.6]}, index=OPTIONS + ["voted"])
    table["predicted_low"], table["predicted_high"] = table["predicted"] - 0.05, table["predicted"] + 0.05
    plain = comparison.source_bars(table, ["predicted", "actual"], OPTIONS, colors(["predicted", "actual"]), head=["voted"], title="t", note="n")
    styled = comparison.source_bars(table, ["predicted", "actual"], OPTIONS, {}, row_colors={"a": "#261060"})
    assert isinstance(plain, Figure) and isinstance(styled, Figure) and len(plain.axes) == 2


def test_interval_panels():
    table = pd.DataFrame({"m": [0.1, 0.2, np.nan], "m_low": [0.05, 0.1, np.nan], "m_high": [0.2, 0.3, np.nan], "d": [-0.1, 0.0, 0.1]}, index=["x", "y", "z"])
    figure = intervals.interval_panels(table, ["m", "d"], percent=["m"], zero=["d"], marks={"m": ["(p<0.001)", "", ""]}, per_row=1, same_scale=False, title="t")
    assert isinstance(figure, Figure) and len(figure.axes) == 2
    assert isinstance(intervals.interval_panels(table, ["m", "d"], percent=["m"], zero=["d"], bars=True), Figure)


def test_violins_and_histograms():
    values = shares(40).assign(voted=RNG.uniform(size=40))
    assert isinstance(distributions.violins(values, OPTIONS, head=["voted"], colors={"a": "#261060"}, title="t", note="n"), Figure)
    assert isinstance(distributions.histograms(values[["a", "b"]], "subjects", title="t"), Figure)


def test_heatmap_and_grid():
    assert isinstance(matrices.heatmap(shares(40).corr(), bound=1, title="t", note="n"), Figure)
    table = pd.DataFrame({"one": [0.1, 0.9], "two": [0.5, np.nan]}, index=["x", "y"])
    assert isinstance(matrices.grid(table, below=table.round(1).astype(str), scale=100, decimals=1, cmap=ERRORS, title="t"), Figure)
    assert isinstance(matrices.grid(table - 0.5, signed=True), Figure)
    stacked_tables = matrices.grids({"first": table, "second": table * 2}, scale=100, decimals=1, cmap=ERRORS, title="t", note="n")
    assert isinstance(stacked_tables, Figure) and len(stacked_tables.axes) == 2
    assert legible((0.0, 0.0, 0.2)) != legible((1.0, 1.0, 0.8))


def test_stacked_bars():
    assert isinstance(stacked.stacked_bars(shares(4), title="t", note="n"), Figure)
    with pytest.raises(ValueError):
        stacked.stacked_bars(pd.DataFrame(np.full((2, 9), 1 / 9)))


def test_maps():
    target = pd.DataFrame(RNG.uniform(0, 10, (3, 2)), index=OPTIONS)
    panels = {name: target + 1 for name in ("one", "two", "three")}
    figure = maps.maps(panels, target, (0, 10), "x", "y", ("model", "experts"), title="t", note="n")
    assert isinstance(figure, Figure) and sum(ax.get_visible() for ax in figure.axes) == 3


def test_lines():
    places = pd.DataFrame({"one": [0.1, 0.3, 0.2], "two": [0.2, -0.1, 0.4]}, index=["x", "y", "z"])
    figure = curves.line_panels({"m": places, "d": places - 0.2, "k": places}, levels={"m": places.mean()}, percent=["m"], zero=["d"], title="t", note="n")
    assert isinstance(figure, Figure) and sum(ax.get_visible() for ax in figure.axes) == 3
    assert isinstance(curves.curves({"one": places, "two": places * 2}, "one", "two", diagonal=True, title="t", note="n"), Figure)
