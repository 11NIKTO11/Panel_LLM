"""Every plot function returns a figure for small inputs and leaves global state alone."""
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pytest

from panel.analysis.plots import comparison, distances, distributions, matrices

OPTIONS = ["a", "b", "c"]
GROUPS = ["g1", "g2", "total"]


@pytest.fixture(autouse=True)
def close_figures():
    before = dict(plt.rcParams)
    yield
    plt.close("all")
    assert dict(plt.rcParams) == before, "a plot function changed global style"


def metrics_table():
    return pd.DataFrame({"MAE": [0.02, 0.03, 0.01], "KL Divergence": [0.1, 0.2, 0.05]}, index=pd.Index(GROUPS, name="group"))


def test_distribution_bars():
    predicted = pd.Series({"Voted": 0.7, "Not Voted": 0.3, "a": 0.5, "b": 0.3, "c": 0.2})
    reference = pd.Series({"Voted": 0.65, "Not Voted": 0.35, "a": 0.4, "b": 0.4, "c": 0.2})
    assert isinstance(comparison.distribution_bars(predicted, reference, OPTIONS, ("Voted", "Not Voted"), labels={"a": "Option A"}, title="t"), plt.Figure)
    assert isinstance(comparison.distribution_bars(predicted, reference, OPTIONS, None), plt.Figure)


def test_bars_by_source():
    assert isinstance(comparison.bars_by_source(pd.Series({"m1": 0.02, "m2": 0.03}), "MAE", scale=100), plt.Figure)


def test_heatmaps():
    frame = pd.DataFrame(np.random.default_rng(0).dirichlet([1, 1, 1], size=20), columns=OPTIONS)
    assert isinstance(matrices.heatmap(frame.cov(), title="cov"), plt.Figure)
    assert isinstance(matrices.heatmap(frame.corr(), vmin=-1, vmax=1), plt.Figure)
    spread = pd.DataFrame(np.random.default_rng(1).random((3, 4)), index=OPTIONS, columns=["r1", "r2", "r3", "r4"])
    assert isinstance(matrices.spread_heatmap(spread, title="spread"), plt.Figure)
    assert isinstance(matrices.spread_heatmap_grid({"m1": spread, "m2": spread * 2}, ncols=2), plt.Figure)


def test_error_grid():
    table = pd.DataFrame({"P vs A": [0.02, 0.03], "P vs C": [0.03, 0.05], "A vs C": [0.01, 0.02]}, index=pd.Index(["g1", "g2"], name="group"))
    assert isinstance(matrices.error_grid(table, ["P", "C", "A"], ncols=2), plt.Figure)


def test_distributions():
    frame = pd.DataFrame(np.random.default_rng(0).dirichlet([1, 1, 1], size=30), columns=OPTIONS)
    assert isinstance(distributions.option_distribution(frame, OPTIONS, kind="violin", add_points=False), plt.Figure)
    assert isinstance(distributions.option_distribution(frame, OPTIONS, sample=10, labels={"a": "A"}), plt.Figure)
    assert isinstance(distributions.error_histogram([0.01, 0.02, 0.05]), plt.Figure)
    assert isinstance(distributions.spread_histograms(frame, OPTIONS, title="spread"), plt.Figure)
    with pytest.raises(ValueError):
        distributions.option_distribution(frame, OPTIONS, kind="pie")


def test_distance_figures():
    tables = {"m1": metrics_table(), "m2": metrics_table() * 1.5}
    lines = distances.distance_lines(tables, ["total", "g1", "g2"], ["MAE", "KL Divergence"], total="total", ncols=1)
    assert isinstance(lines, go.Figure) and len(lines.data) == 4
    bars = distances.distance_bars(tables, "total", ["MAE", "KL Divergence"])
    assert isinstance(bars, go.Figure) and len(bars.data) == 2
    with pytest.raises(ValueError):
        distances.distance_bars(tables, "nowhere", ["MAE"])


def test_effects_figures():
    effects = pd.DataFrame({"model": ["m1", "m1", "m2", "m2"], "predictor": ["age", "region"] * 2,
                            "Pillai's trace": [0.1, 0.4, 0.2, 0.5], "Wilks' lambda": [0.9, 0.6, 0.8, 0.5]}).set_index(["model", "predictor"])
    assert isinstance(distances.effects_bars(effects), plt.Figure)
    assert isinstance(distances.effects_heatmap(effects), plt.Figure)
    with pytest.raises(ValueError):
        distances.effects_bars(effects, stat="Nonsense")
