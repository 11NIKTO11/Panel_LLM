"""The analysis layer, offline: a fake run is read back and every analysis makes its table and figures."""
import pandas as pd
import pytest
from matplotlib.figure import Figure

from panel.analysis import ANALYSES, NotApplicable, Output, describe, find_stem, load_run
from panel.analysis.comparison import COMPARISONS, load_runs
from panel.analysis.levels import members
from panel.analysis.sources import ranked
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
