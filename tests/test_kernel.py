"""
Runs read as kernels: what a kernel holds for a grid and for a survey run, and the analyses
reproducing `CoRe/llm_pushforward` on the legacy runs it read (`fixtures/pushforward.json`).
"""
import numpy as np
import pandas as pd
import pytest

from panel.analysis import Output
from panel.analysis.comparison import load_runs
from panel.analysis.kernel import KERNEL_ANALYSES, load_kernels, push
from panel.analysis.kernel.noise import noise_floor
from panel.analysis.kernel.pairs import FACTOR, minimal_pairs
from panel.analysis.kernel.variance import INTERACTIONS, MAIN, NOISE, POOLED, decompose
from panel.analysis.stats.spectral import EFFECTIVE_AXES
from panel.run import ResultStore

from tests.conftest import RESULTS_DIR, load_fixture, require

PUSHFORWARD = load_fixture("pushforward")
LEGACY = {model: f"sod_legacy_cslegacy_legacy_OpenAI_{model}_Prob_t=0.0" for model in PUSHFORWARD["scorecard"]}
GRIDS = {"sod": "gridsod_default", "simple": "gridsimple", "colleague": "gridcolleague_delivered_colleague_OpenRouter_gpt-4o-2024-11-20_Prob_t=0.3_x3"}


@pytest.fixture(scope="module")
def store() -> ResultStore:
    require(RESULTS_DIR / "election_2021" / f"{LEGACY['gpt-4o']}_run.json")
    return ResultStore(RESULTS_DIR, "election_2021")


@pytest.fixture(scope="module")
def grids(store):
    return load_kernels(load_runs(store, GRIDS))


@pytest.fixture(scope="module")
def legacy(store):
    return load_kernels(load_runs(store, LEGACY))


def test_a_grid_is_a_designed_kernel(grids):
    sod, simple, colleague = (grids.kernels[label] for label in GRIDS)
    assert sod.matrix.shape == (10, 360) and simple.matrix.shape == colleague.matrix.shape == (10, 384)
    assert sod.designed and simple.designed and colleague.designed
    assert sod.repeats == 3 and simple.repeats == 1 and colleague.averaged == 3
    for kernel in (sod, simple, colleague):
        np.testing.assert_allclose(kernel.matrix.sum(), 1)
        assert (kernel.weights == 1).all()


def test_a_survey_run_is_a_sampled_kernel(legacy):
    kernel = legacy.kernels["gpt-4o"]
    assert not kernel.designed and kernel.matrix.shape == (10, 3854) and kernel.weights.sum() == 3880
    run = kernel.run
    np.testing.assert_allclose(push(kernel, kernel.weights).to_numpy(), kernel.calls[kernel.outcomes].mean().to_numpy(), atol=1e-12)
    assert run.topic.turnout[1] in kernel.outcomes


def test_minimal_pairs_of_a_grid_are_what_its_design_says(grids):
    kernel = grids.kernels["sod"]
    counts = minimal_pairs(kernel)[FACTOR].value_counts()
    for column in kernel.factors.columns:
        assert counts[column] == len(kernel.factors) * (kernel.factors[column].nunique() - 1) // 2


def test_noise_floor_comes_from_repeats_then_copies(grids, legacy):
    assert noise_floor(grids.kernels["sod"]).kind == "repeats" and len(noise_floor(grids.kernels["sod"]).pairs) == 360 * 3
    assert noise_floor(grids.kernels["simple"]).kind is None and noise_floor(legacy.kernels["gpt-4o"]).kind == "copies"


@pytest.mark.parametrize("model", sorted(LEGACY))
def test_scales_reproduce_pushforward(model, legacy):
    expected = PUSHFORWARD["scorecard"][model]
    out = Output(None, "test", save=False)
    table, _ = KERNEL_ANALYSES["scales"](type(legacy)(legacy.topic, {model: legacy.kernels[model]}, legacy.style), out)
    row = table.loc[model]
    assert row["Dobrushin δ"] == pytest.approx(expected["delta (Dobrushin)"], abs=1e-6)
    assert row["q99 pairwise TV"] == pytest.approx(expected["delta_robust (q99)"], abs=1e-4)
    assert row["mean TV from the mean answer"] == pytest.approx(expected["mean TV to mean column"], abs=1e-6)
    assert row["noise floor"] == pytest.approx(expected["noise floor (TV)"], abs=1e-6)


@pytest.mark.parametrize("model", sorted(LEGACY))
def test_variance_and_axes_reproduce_pushforward(model, legacy):
    expected, kernel = PUSHFORWARD["scorecard"][model], legacy.kernels[model]
    split = decompose(kernel, [column for column in kernel.factors.columns if column != "district"]).loc[POOLED]
    assert split[MAIN] + split[INTERACTIONS] == pytest.approx(expected["var explained by properties (main+inter)"], abs=2e-3)
    assert split[NOISE] == pytest.approx(expected["var: noise"], abs=2e-3)
    single = type(legacy)(legacy.topic, {model: kernel}, legacy.style)
    table, _ = KERNEL_ANALYSES["axes"](single, Output(None, "test", save=False))
    assert table.loc[model, EFFECTIVE_AXES] == pytest.approx(expected["effective number of axes"], abs=1e-6)


def test_shapley_reproduces_pushforward(legacy):
    model = "gpt-4o"
    kernel, expected = legacy.kernels[model], PUSHFORWARD["shapley"][model]
    named = {name: next(column for column in kernel.players() if set(columns) == set(kernel.run.dataset.together.get(column, (column,))))
             for name, columns in PUSHFORWARD["players"].items()}
    table, _ = KERNEL_ANALYSES["shapley"](type(legacy)(legacy.topic, {model: kernel}, legacy.style), Output(None, "test", save=False), draws=0)
    found = table.loc[model]
    for name, effect in expected["effects"].items():
        assert found.loc[named[name], "effect"] == pytest.approx(effect, abs=1e-6), name
        for axis, effects in PUSHFORWARD["shapley_axes"][model].items():
            assert found.loc[named[name], f"axis {axis}"] == pytest.approx(effects[name], abs=1e-6), (name, axis)
    assert found["explained by all"].iloc[0] == pytest.approx(expected["explained"], abs=1e-6)


def test_intervals_pair_kernels_of_the_same_profiles(store, legacy, grids):
    out = Output(None, "test", save=False)
    table, _ = KERNEL_ANALYSES["intervals"](legacy, out, draws=20, reference="gpt-4o")
    estimate, low, high = (table.xs("gpt-4o")[column] for column in ("estimate", "low", "high"))
    assert (low <= estimate + 1e-9).all() and (estimate <= high + 1e-9).all()
    found, _ = KERNEL_ANALYSES["intervals"](grids, out, draws=10)
    assert found.xs("sod")["interval"].iloc[0].startswith("resampled repeats") and found.xs("simple")["interval"].iloc[0].startswith("none")


def test_surrogate_on_a_grid_observes_the_whole_product(grids):
    table, _ = KERNEL_ANALYSES["surrogate"](type(grids)(grids.topic, {"sod": grids.kernels["sod"]}, grids.style), Output(None, "test", save=False))
    row = table.loc["sod"]
    assert row["additive R² (out of sample)"] > 0.9 and row["independent factors: share observed"] == 1.0
    assert row["δ surrogate, full product"] == pytest.approx(row["δ surrogate, observed profiles"])


def test_agreement_holds_the_runs_against_the_trained_model(legacy):
    table, _ = KERNEL_ANALYSES["agreement"](legacy, Output(None, "test", save=False))
    assert set(table.index) == set(LEGACY) and (table["shared profiles"] > 2000).all()
    assert table.loc["gpt-4.1-nano", "mean TV"] > table.loc["gpt-4o", "mean TV"]


def test_every_analysis_runs_on_grids_and_survey_runs(store):
    """Shapley is left out: `test_shapley_reproduces_pushforward` runs it, and on a survey run it takes a minute."""
    kernels = load_kernels(load_runs(store, {**GRIDS, "gpt-4o": LEGACY["gpt-4o"], "gpt-4.1": LEGACY["gpt-4.1"]}))
    out = Output(None, "test", save=False)
    settings = {"intervals": {"draws": 5}, "agreement": {"reference": "gpt-4o"}}
    for name, analysis in KERNEL_ANALYSES.items():
        if name == "shapley":
            continue
        table, figures = analysis(kernels, out, **settings.get(name, {}))
        assert isinstance(table, pd.DataFrame) and len(table) and figures, name
