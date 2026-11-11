"""The run layer, offline: a spec resolves, executes through the fake client, and its record round-trips."""
import json
import shutil

import pytest

from panel import paths
from panel.analysis import validate
from panel.dataset import load_dataset
from panel.run import ResultStore, RunSpec, execute, resolve
from panel.run.check import check, report
from panel.run.fake import FakeClient
from panel.run.named import FIELDS, OWN, build_spec, load_runs, parse
from panel.run.registry import client_class

from tests.conftest import TOPICS_DIR, require

SOD = RunSpec(dataset="sod", description="default", topic="election_2021", framing="cs", question="default",
              client="Fake", model="fake-model", shape="Prob", temperature=0.0, sample=20)
REPEATED = SOD.variant(sample=5, repeats=3)
FOUR = ("results_csv", "results_json", "normalized_csv", "record")
RUNS = paths.CONFIG_DIR / "runs.json"


def prepared(spec: RunSpec):
    dataset = load_dataset(spec.dataset)
    require(dataset.values_path)
    require(dataset.prompts_path(spec.description))
    return resolve(spec)


def write_runs(tmp_path, base: dict, runs: dict):
    path = tmp_path / "runs.json"
    path.write_text(json.dumps({"base": base, "runs": runs}), encoding="utf-8")
    return path


def test_stem_encodes_every_axis():
    assert prepared(SOD).stem == "sod_default_cs_default_Fake_fake-model_Prob_t=0.0_subset=20"
    assert prepared(REPEATED).stem == "sod_default_cs_default_Fake_fake-model_Prob_t=0.0_x3_subset=5"
    assert prepared(SOD.variant(shape="Pick", temperature=0.3, sample=None)).stem == "sod_default_cs_default_Fake_fake-model_Pick_t=0.3"
    assert prepared(SOD.variant(temperature=None)).stem == "sod_default_cs_default_Fake_fake-model_Prob_t=default_subset=20"
    assert prepared(SOD.variant(question="neutral")).stem == "sod_default_cs_neutral_Fake_fake-model_Prob_t=0.0_subset=20"
    effort = SOD.variant(client="Anthropic", model="claude-haiku-5-5", temperature=None, effort="low")
    assert prepared(effort).stem == "sod_default_cs_default_Anthropic_claude-haiku-5-5_Prob_t=default_e=low_subset=20"


def test_execute_offline(tmp_path):
    run = prepared(SOD)
    store = ResultStore(tmp_path, SOD.topic)
    written = execute(run, store, FakeClient(), max_workers=4)

    plain = store.paths(run.stem)
    archived = list((store.topic_dir / store.ARCHIVE_DIR).glob(f"{run.stem}_*"))
    assert all(getattr(plain, name).exists() for name in FOUR) and len(archived) == 2
    frame = store.frame(run.stem)
    assert frame.index.name == "id" and len(frame) == 20 and list(frame.columns) == run.topic.columns + ["Prompt"]
    assert set(frame.index) == set(run.rows.index)
    normalized = store.frame(run.stem, normalized=True)
    assert abs(normalized[list(run.topic.options)].sum(axis=1) - 1).max() < 1e-9

    record = store.record(run.stem)
    assert RunSpec.from_dict(record["spec"]) == SOD
    assert resolve(RunSpec.from_dict(record["spec"])).stem == run.stem
    assert record["response_schema"] == run.response_model.model_json_schema()
    assert record["framing"]["instructions"] == run.framing.instructions and record["n"] == 20
    assert "question" not in record["framing"] and record["question"]["forms"] == run.question.forms
    assert {name: record["dataset"][f"{name}_sha256"] for name in run.digests} == run.digests
    assert sorted(record["dataset"]["rows"]) == sorted(int(i) for i in run.rows.index)
    assert record["model_resolved"] == "fake-model-fake" and "pydantic" in record["versions"]

    loaded = store.load(run.stem, run.response_model)
    assert set(loaded) == {(int(i), 1) for i in run.rows.index}
    assert all(result.options for _, result in loaded.values())
    bad_turnout, bad_options, duplicates = validate.evaluate_voting_results(loaded)
    assert not bad_turnout and not bad_options and not duplicates
    assert store.runs() == [run.stem] and written.results_csv.exists()


def test_check_of_a_saved_run(tmp_path):
    run = prepared(SOD)
    store = ResultStore(tmp_path, SOD.topic)
    with pytest.raises(FileNotFoundError):
        check(run, store)
    execute(run, store, FakeClient(), max_workers=4)
    result = check(run, store)
    total = len(run.topic.options)
    assert result.answers == result.calls == 20 and result.same_calls and result.model_resolved == "fake-model-fake"
    assert result.missing == 0 and result.out_of_range == 0 and result.duplicates == 0
    sent = run.rows.apply(run.prompt, axis=1)
    assert result.subjects == 20 and result.identical_prompts == sent.duplicated(keep=False).sum()
    assert result.turnout.off == 0 and result.options.off == 0 and result.listed == {total: 20}
    assert list(result.means) == run.topic.columns and abs(sum(result.top.values()) - 1) < 1e-9
    assert run.stem in report(result, total)

    pick = prepared(SOD.variant(shape="Pick"))
    execute(pick, store, FakeClient(), max_workers=4)
    picked = check(pick, store)
    assert picked.options is None and picked.top is None and picked.turnout.off == 0
    assert abs(sum(picked.means[option] for option in pick.topic.options) - 1) < 1e-9
    assert pick.stem in report(picked, total)


def test_execute_repeats(tmp_path):
    run = prepared(REPEATED)
    store = ResultStore(tmp_path, REPEATED.topic)
    execute(run, store, FakeClient(), max_workers=4)
    frame = store.frame(run.stem)
    assert len(frame) == 15 and list(frame.columns) == ["repeat"] + run.topic.columns + ["Prompt"]
    assert frame.groupby(level=0)["repeat"].apply(list).map(lambda r: r == [1, 2, 3]).all()
    loaded = store.load(run.stem, run.response_model)
    first = int(run.rows.index[0])
    assert len(loaded) == 15 and (first, 3) in loaded
    raw = json.loads(store.paths(run.stem).results_json.read_text(encoding="utf-8"))
    assert set(raw[f"{first}:1"]) == {"prompt", "result"}


def test_execute_reruns_skipped(tmp_path):
    run = prepared(SOD.variant(sample=5))
    store = ResultStore(tmp_path, SOD.topic)
    execute(run, store, FakeClient(fail_calls=3), max_workers=2, retries=3, rerun_rounds=2)
    assert len(store.frame(run.stem)) == 5
    with pytest.raises(RuntimeError):
        execute(run, ResultStore(tmp_path / "again", SOD.topic), FakeClient(fail_calls=99), max_workers=2, retries=2, rerun_rounds=1)


def test_store_collision(tmp_path):
    store = ResultStore(tmp_path, SOD.topic)
    execute(prepared(SOD), store, FakeClient(), max_workers=4)
    other = prepared(SOD.variant(random_state=7))
    with pytest.raises(FileExistsError):
        execute(other, store, FakeClient(), max_workers=4)
    execute(other, store, FakeClient(), max_workers=4, overwrite=True)
    assert RunSpec.from_dict(store.record(other.stem)["spec"]).random_state == 7


def test_store_refuses_another_topic(tmp_path):
    with pytest.raises(ValueError):
        execute(prepared(SOD), ResultStore(tmp_path, "election_2025"), FakeClient())


def test_question_is_worded_per_subject(people):
    run = prepared(SOD.variant(sample=None))
    forms = run.question.forms
    assert run.question.by == "gender" and forms["male"] != forms["female"]
    worded = run.rows["question"].groupby(people["gender"]).unique()
    assert {answer: list(texts) for answer, texts in worded.items()} == {"Muž": [forms["male"]], "Žena": [forms["female"]]}
    row = run.calls.iloc[0]
    assert run.prompt(row) == f"{row['default']} {row['question']}"


def test_one_form_question_reads_the_same_for_everyone():
    run = prepared(SOD.variant(question="neutral"))
    assert run.question.by is None and run.rows["question"].nunique() == 1


def test_resolve_refuses_a_missing_or_unknown_question():
    with pytest.raises(ValueError):
        prepared(SOD.variant(question=None))
    with pytest.raises(FileNotFoundError):
        prepared(SOD.variant(question="nobody"))


def test_resolve_language_mismatch():
    with pytest.raises(ValueError):
        prepared(SOD.variant(framing="en"))


def test_question_needs_the_framings_language(tmp_path):
    shutil.copytree(TOPICS_DIR / "election_2021", tmp_path / "election_2021")
    path = tmp_path / "election_2021" / "questions" / "default.json"
    config = json.loads(path.read_text(encoding="utf-8"))
    del config["languages"]["cs"]
    path.write_text(json.dumps(config, ensure_ascii=False), encoding="utf-8")
    with pytest.raises(ValueError):
        resolve(SOD, topics_dir=tmp_path)


def test_resolve_refuses_undeclared_description():
    with pytest.raises(ValueError):
        resolve(SOD.variant(description="nobody"))


def test_effort_needs_a_client_that_takes_one():
    with pytest.raises(ValueError):
        prepared(SOD.variant(effort="low"))
    with pytest.raises(ValueError):
        SOD.variant(effort="huge")


def test_registry():
    assert client_class("OpenRouter") is client_class("OpenRouterClient")
    assert client_class("Fake") is FakeClient
    with pytest.raises(KeyError):
        client_class("Nobody")
    with pytest.raises(KeyError):
        prepared(SOD.variant(client="Nobody"))


def test_spec_rejects_nonsense():
    with pytest.raises(ValueError):
        SOD.variant(shape="VProb")
    with pytest.raises(ValueError):
        SOD.variant(repeats=0)
    with pytest.raises(TypeError):
        RunSpec.from_dict({"name": "sod", "topic": "election_2021_legacy"})


def test_named_runs_are_complete_specs():
    runs = load_runs(RUNS)
    assert runs and all(set(named) == set(FIELDS) for named in runs.values())
    assert build_spec(runs["fake"], {"sample": 5}).sample == 5 and build_spec(runs["fake"], {}).client == "Fake"
    stems = [prepared(build_spec(named, {})).stem for named in runs.values()]
    assert len(set(stems)) == len(stems)


def test_named_runs_keep_model_settings_out_of_the_base(tmp_path):
    base = {field: getattr(SOD, field) for field in FIELDS if field not in OWN}
    own = {field: getattr(SOD, field) for field in OWN}
    assert load_runs(write_runs(tmp_path, base, {"a": own, "b": {**own, "shape": "Pick"}}))["b"]["shape"] == "Pick"
    with pytest.raises(ValueError):
        load_runs(write_runs(tmp_path, {**base, "model": "fake-model"}, {"a": own}))
    with pytest.raises(ValueError):
        load_runs(write_runs(tmp_path, base, {"a": {field: own[field] for field in OWN[1:]}}))
    with pytest.raises(ValueError):
        load_runs(write_runs(tmp_path, base, {"a": {**own, "nobody": 1}}))
    with pytest.raises(ValueError):
        build_spec(None, {"dataset": "sod"})


def test_parse_reads_what_a_person_typed():
    assert parse("sample", "none") is None and parse("sample", "200") == 200
    assert parse("temperature", "0.3") == 0.3 and parse("effort", "low") == "low"
    with pytest.raises(ValueError):
        parse("repeats", "many")
