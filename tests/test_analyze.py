"""The analysis layer, offline: a fake run is read back and every analysis makes its table and figures."""
import json

import pandas as pd
import pytest
from matplotlib.figure import Figure

from panel.analysis import ANALYSES, NotApplicable, Output, describe, find_stem, load_run
from panel.analysis.benchmark import BENCHMARKS, load_ground
from panel.analysis.benchmark.ground import SHARES, TRAINED
from panel.analysis.comparison import COMPARISONS, load_comparisons, load_runs
from panel.analysis.comparison import describe as describe_runs
from panel.analysis.levels import members
from panel.analysis.area import ordered
from panel.analysis.sources import PREDICTED, estimated, frame, ranked
from panel import paths
from panel.dataset import Levels
from panel.run import ResultStore, execute
from panel.run.fake import FakeClient

from tests.test_run import REPEATED, SOD, prepared


@pytest.fixture
def saved(tmp_path):
    store = ResultStore(tmp_path, SOD.topic)
    for spec in (SOD, REPEATED):
        execute(prepared(spec), store, FakeClient(), max_workers=4)
    return store


def test_a_run_is_found_and_read_back(saved):
    with pytest.raises(ValueError):
        find_stem(saved, "fake-model")
    stem = find_stem(saved, "x3")
    run = load_run(saved, stem)
    assert len(run.subjects) == 5 and len(run.results) == 15 and run.claims and run.reference is not None
    assert stem in describe(run) and ranked(run)[-1] == "Other" and set(ranked(run)) == set(run.options)
    assert load_run(saved, stem, styled=False).style.colors == {}
    plain = load_run(saved, stem, weighted=False)
    assert not plain.weighted and run.weighted and not estimated(plain, frame(plain))[PREDICTED].equals(estimated(run, frame(run))[PREDICTED])
    assert ordered(run) == [name for name in run.dataset.levels[run.dataset.area].order if name in set(run.subjects[run.dataset.area])]


@pytest.mark.parametrize("text", ["subset=20", "x3"])
def test_every_analysis_gives_a_table_and_figures(saved, text):
    run = load_run(saved, find_stem(saved, text))
    out = Output(saved, run.stem)
    for name, analysis in ANALYSES.items():
        try:
            table, figures = analysis(run, out)
        except NotApplicable:
            assert name == "effects"
            continue
        assert isinstance(table, pd.DataFrame) and not table.empty, name
        assert all(isinstance(figure, Figure) for figure in figures), name
    assert out.written and all(path.exists() for path in out.written)
    assert Output(saved, run.stem, save=False).table("nothing", table) is table and not saved.analysis("nothing_" + run.stem + ".csv").exists()


def test_runs_side_by_side(saved):
    with pytest.raises(ValueError):
        load_runs(saved, ["x3", "x3"])
    runs = load_runs(saved, ["subset=20", "x3"])
    assert len(set(runs.runs)) == 2 and len(runs.claimed) == 2 and list(runs.polls.columns) == list(runs.topic.options)
    out = Output(saved, "pair")
    for name, comparison in COMPARISONS.items():
        try:
            table, figures = comparison(runs, out)
        except NotApplicable:
            assert name == "effects"
            continue
        assert isinstance(table, pd.DataFrame) and not table.empty, name
        assert figures and all(isinstance(figure, Figure) for figure in figures), name
    assert all(path.exists() and "_pair." in path.name for path in out.written)


def test_levels_of_a_grouping_column():
    values = pd.Series([20, 30, 40, 30], index=list("abcd"), name="age")
    assert {name: list(at) for name, at in members(values, Levels({"young": 25, "old": 99}, (), None)).items()} == {"young": ["a"], "old": ["b", "c", "d"]}
    answers = pd.Series(["low", "high", "low", None], index=list("abcd"), name="scale")
    assert list(members(answers, Levels({}, ("low", "mid", "high"), None))) == ["low", "high"]
    assert list(members(answers)) == ["low", "high"]
    held = pd.Series(["x", "x; y", "y", "z; x"], index=list("abcd"), name="held")
    assert {name: len(at) for name, at in members(held, Levels({}, (), "; ")).items()} == {"x": 3, "y": 2, "z": 1}
    with pytest.raises(ValueError):
        members(answers, Levels({}, ("low",), None))
    with pytest.raises(ValueError):
        members(values, Levels({"young": 25}, (), None))


def test_runs_against_a_model_trained_on_the_answers(saved):
    ground = load_ground(load_runs(saved, ["subset=20"]), ["gender", "age"], folds=2)
    assert list(ground.predicted) == ["fake-model"] and ground.outcomes[-1] == ground.topic.turnout[1]
    assert len(ground.answers) <= 20 and all(abs(shares.sum(axis=1) - 1).max() < 1e-9 for shares in ground.predicted.values())
    assert ground.sources[:2] == [SHARES, TRAINED] and len(ground.sources) == 5
    out = Output(saved, "trained")
    for name, benchmark in BENCHMARKS.items():
        table, figures = benchmark(ground, out)
        assert isinstance(table, pd.DataFrame) and not table.empty, name
        assert figures and all(isinstance(figure, Figure) for figure in figures), name
    assert all(path.exists() and "benchmark_" in path.name for path in out.written)
    with pytest.raises(ValueError):
        load_ground(load_runs(saved, ["subset=20"]), ["nope"])


def test_named_sets_of_runs(saved, tmp_path):
    named = load_comparisons(paths.CONFIG_DIR / "comparisons.json")
    assert named and all(chosen.name == name and chosen.runs for name, chosen in named.items())
    runs = load_runs(saved, {"small": "subset=20", "repeated": "x3"})
    assert list(runs.runs) == ["small", "repeated"] and "small" in describe_runs(runs)
    for wrong in ({"a": {"topic": "t"}}, {"a": {"topic": "t", "runs": {}}}, {"a": {"topic": "t", "runs": {"x": "s", "y": "s"}}}):
        path = tmp_path / "comparisons.json"
        path.write_text(json.dumps(wrong), encoding="utf-8")
        with pytest.raises(ValueError):
            load_comparisons(path)
