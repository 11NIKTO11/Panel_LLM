"""The run layer, offline: a spec resolves, executes through the fake client, and its record round-trips."""
import json
import shutil

import pytest

from panel import paths
from panel.dataset import load_dataset
from panel.analysis.saved import describe, load_run
from panel.llm import BatchFailed
from panel.run import ResultStore, RunSpec, execute, resolve
from panel.run.accept import judge
from panel.run.check import MEAN, TOP, check, means, report
from panel.run.fake import FakeClient
from panel.run.presets import DATA, FIELDS, MODEL, build_spec, load_presets, parse
from panel.run.record import KEPT
from panel.run.registry import client_class

from tests.conftest import TOPICS_DIR, require

SOD = RunSpec(dataset="sod", description="default", topic="election_2021", framing="cs", question="default",
              client="Fake", model="fake-model", shape="Prob", temperature=0.0, sample=20)
REPEATED = SOD.variant(sample=5, repeats=3)
CITIZEN = SOD.variant(dataset="regions", description="citizen", sample=None, repeats=3)
DIRECT = CITIZEN.variant(description="direct", question=None)
EFFORT = SOD.variant(client="Anthropic", model="claude-haiku-5-5", temperature=None, effort="low", sample=5)
FOUR = ("results_csv", "results_json", "normalized_csv", "record")
PRESETS = paths.CONFIG_DIR / "presets.json"


def prepared(spec: RunSpec):
    dataset = load_dataset(spec.dataset)
    require(dataset.values_path)
    require(dataset.prompts_path(spec.description))
    return resolve(spec)


def write_presets(tmp_path, base: dict, data: dict, models: dict):
    path = tmp_path / "presets.json"
    path.write_text(json.dumps({"base": base, "data": data, "models": models}), encoding="utf-8")
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
    archived = list((store.run_dir(run.stem) / store.ARCHIVE_DIR).iterdir())
    assert all(getattr(plain, name).exists() for name in FOUR) and len(archived) == 2
    frame = store.frame(run.stem)
    assert frame.index.name == "id" and len(frame) == 20 and list(frame.columns) == run.topic.columns + ["Prompt"]
    assert set(frame.index) == set(run.rows.index)
    normalized = store.frame(run.stem, normalized=True)
    assert abs(normalized[list(run.topic.options)].sum(axis=1) - 1).max() < 1e-9

    record = store.record(run.stem)
    assert RunSpec.from_dict(record["spec"]) == SOD
    assert resolve(RunSpec.from_dict(record["spec"])).stem == run.stem
    assert record["response_schema"] == record["response_schema_sent"] == run.response_model.model_json_schema()
    assert record["framing"]["instructions"] == run.framing.instructions and record["n"] == 20
    assert "question" not in record["framing"] and record["question"]["forms"] == run.question.forms
    assert {name: record["dataset"][f"{name}_sha256"] for name in run.digests} == run.digests
    assert sorted(record["dataset"]["rows"]) == sorted(int(i) for i in run.rows.index)
    assert record["client_parameters"] == {} and record["model_resolved"] == "fake-model-fake" and "pydantic" in record["versions"]

    loaded = store.load(run.stem, run.response_model)
    assert set(loaded) == {(int(i), 1) for i in run.rows.index}
    assert all(result.options for _, result in loaded.values())
    assert store.runs() == [run.stem] and written.results_csv.exists()


def test_check_of_a_saved_run(tmp_path):
    run = prepared(SOD)
    store = ResultStore(tmp_path, SOD.topic)
    with pytest.raises(FileNotFoundError):
        check(store, run.stem, run.topic, run.framing)
    execute(run, store, FakeClient(), max_workers=4)
    result = check(store, run.stem, run.topic, run.framing)
    total = len(run.topic.options)
    assert result.answers == result.calls == 20 and result.same_calls and result.model_resolved == "fake-model-fake"
    assert result.missing == 0 and result.out_of_range == 0 and result.duplicates == 0
    sent = run.rows.apply(run.prompt, axis=1)
    assert result.subjects == 20 and result.identical_prompts == sent.duplicated(keep=False).sum()
    assert result.turnout.off == 0 and result.options.off == 0 and result.listed == {total: 20}
    assert result.abstained == 0 and result.undefined == 0 and result.both == 0
    table = means(store.frame(run.stem), run.topic)
    assert list(table.index) == run.topic.columns and abs(table[TOP].sum() - 1) < 1e-9
    assert run.stem in report(result, total)

    pick = prepared(SOD.variant(shape="Pick"))
    execute(pick, store, FakeClient(), max_workers=4)
    picked = check(store, pick.stem, pick.topic, pick.framing)
    assert picked.options is None and picked.undefined is None and picked.both is None and picked.turnout.off == 0
    assert abs(means(store.frame(pick.stem), pick.topic).loc[list(pick.topic.options), MEAN].sum() - 1) < 1e-9
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
    assert store.record(run.stem)["sending"] == {
        "calls_sent": 20, "refused": {}, "failed": 15, "rate_limited": 0, "answered": 5, "rerun_rounds": 1,
        "settings": {"max_workers": 2, "retries": 3, "rerun_rounds": 2},
    }
    with pytest.raises(RuntimeError):
        execute(run, ResultStore(tmp_path / "again", SOD.topic), FakeClient(fail_calls=99), max_workers=2, retries=2, rerun_rounds=1)


class Altered(FakeClient):
    """A fake whose answers to the first `subjects` prompts it sees are changed by `change`, on their first `times` calls."""

    def __init__(self, change, subjects: int = 1, times: int = 99):
        super().__init__()
        self.change, self.subjects, self.times, self.chosen = change, subjects, times, {}

    def call_llm(self, model, prompt, response_format, temperature, instructions=None):
        answer = super().call_llm(model, prompt, response_format, temperature, instructions)
        with self._lock:
            if prompt not in self.chosen and len(self.chosen) < self.subjects:
                self.chosen[prompt] = 0
            if prompt not in self.chosen or self.chosen[prompt] >= self.times:
                return answer
            self.chosen[prompt] += 1
        return self.change(answer)


def scaled(factor: float):
    def change(answer):
        options = [{"name": option.name, "probability": min(1.0, option.probability * factor)} for option in answer.options]
        return type(answer).model_validate({**answer.model_dump(), "options": options})
    return change


def doubled(answer):
    options = [option.model_dump() for option in answer.options]
    return type(answer).model_validate({**answer.model_dump(), "options": options + options[:1]})


def no_preference(answer):
    options = [{"name": option.name, "probability": 0.0} for option in answer.options]
    return type(answer).model_validate({"voted_or_not": {"voted": 0.0, "not_voted": 1.0}, "options": options})


BOUNDED = SOD.variant(sample=5, sum_low=0.95, sum_high=1.05)


@pytest.mark.parametrize("change, low, high, reasons, usable", [
    (lambda answer: answer, 0.95, 1.05, None, None),
    (scaled(0.5), 0.95, 1.05, ("options_sum",), True),
    (scaled(0.5), None, None, None, None),
    (scaled(0.0), None, None, ("options_zero",), False),
    (no_preference, 0.95, 1.05, ("no_preference",), True),
    (doubled, None, None, ("duplicates",), False),
    (lambda answer: type(answer).model_validate({**answer.model_dump(), "voted_or_not": {"voted": 0.0, "not_voted": 0.0}}), None, None, ("turnout_zero",), False),
    (lambda answer: type(answer).model_validate({**answer.model_dump(), "voted_or_not": {"voted": 0.6, "not_voted": 0.6}}), 0.95, 1.05, ("turnout_sum",), True),
], ids=["fine", "low sum", "no bounds", "zero options", "abstainer without preference", "duplicate", "zero turnout", "high turnout"])
def test_an_answer_is_judged(change, low, high, reasons, usable):
    run = prepared(SOD.variant(sample=1))
    answer = change(FakeClient().call_llm("fake-model", run.prompt(run.calls.iloc[0]), run.response_model, 0.0))
    flaw = judge(answer, low, high)
    assert (flaw and (flaw.reasons, flaw.usable)) == (reasons and (reasons, usable))


def test_a_flawed_answer_is_tried_again(tmp_path):
    run = prepared(BOUNDED)
    store = ResultStore(tmp_path, BOUNDED.topic)
    client = Altered(scaled(0.5), times=2)
    execute(run, store, client, max_workers=1)
    record = store.record(run.stem)
    assert record["faulty"] == {} and "incomplete" not in record and client._seen[next(iter(client.chosen))] == 3
    assert list(record["attempts"].values()) == [["options_sum", "options_sum"]]
    assert record["sending"] == {
        "calls_sent": 7, "refused": {"options_sum": 2}, "failed": 0, "rate_limited": 0, "answered": 5, "rerun_rounds": 0,
        "settings": {"max_workers": 1, "retries": 3, "rerun_rounds": 5},
    }
    assert "7 for 5 answers; refused: options_sum 2" in report(check(store, run.stem, run.topic, run.framing), len(run.topic.options))


def test_the_best_flawed_answer_is_kept_and_marked(tmp_path):
    run = prepared(BOUNDED)
    store = ResultStore(tmp_path, BOUNDED.topic)
    client = Altered(scaled(0.5), subjects=2)
    execute(run, store, client, max_workers=1, retries=2, rerun_rounds=3)
    record = store.record(run.stem)
    marked = {str(int(run.calls.loc[run.calls.apply(run.prompt, axis=1) == prompt, "id"].iloc[0])) for prompt in client.chosen}
    assert set(record["faulty"]) == marked and set(map(tuple, record["faulty"].values())) == {("options_sum",)}
    assert all(client._seen[prompt] == 2 for prompt in client.chosen) and len(store.frame(run.stem)) == 5
    looked = check(store, run.stem, run.topic, run.framing)
    assert looked.faulty == {"options_sum": 2} and "kept though flawed" in report(looked, len(run.topic.options))
    assert record["sending"]["calls_sent"] == 3 + 2 * 2 and record["sending"]["refused"] == {"options_sum": 4}
    assert "2 kept though flawed" in describe(load_run(store, run.stem))


def test_a_run_with_a_call_missing_is_saved_as_incomplete(tmp_path):
    run = prepared(BOUNDED)
    store = ResultStore(tmp_path, BOUNDED.topic)
    with pytest.raises(RuntimeError, match="_incomplete"):
        execute(run, store, Altered(doubled), max_workers=1, retries=2, rerun_rounds=1)
    stem = run.stem + store.INCOMPLETE
    record = store.record(stem)
    assert store.runs() == [stem] and record["incomplete"] and record["stem"] == stem and record["n"] == 4
    assert len(record["missing"]) == 1 and len(store.frame(stem)) == 4
    assert record["sending"] == {
        "calls_sent": 4 + 2 * 2, "refused": {"duplicates": 4}, "failed": 0, "rate_limited": 0, "answered": 4, "rerun_rounds": 1,
        "settings": {"max_workers": 1, "retries": 2, "rerun_rounds": 1},
    }
    assert not check(store, stem, run.topic, run.framing).same_calls
    with pytest.raises(ValueError, match="incomplete"):
        load_run(store, stem)
    execute(run, store, FakeClient(), max_workers=1)
    assert store.runs() == [run.stem, stem] and "incomplete" not in store.record(run.stem)
    with pytest.raises(FileExistsError, match="whole already"):
        execute(run, store, FakeClient(), max_workers=1, missing=True)


def test_missing_sends_only_what_the_incomplete_run_lacks(tmp_path):
    run = prepared(SOD)
    store = ResultStore(tmp_path, SOD.topic)
    with pytest.raises(FileNotFoundError):
        execute(run, store, FakeClient(), max_workers=1, missing=True)
    with pytest.raises(BatchFailed):
        execute(run, store, Broke(paid=7), max_workers=1)
    partial = store.record(run.stem + store.INCOMPLETE)

    with pytest.raises(BatchFailed):
        execute(run, store, Broke(paid=5), max_workers=1, missing=True)
    again = store.record(run.stem + store.INCOMPLETE)
    assert again["n"] == 12 and again["resumed"]["answers"] == 7 and store.runs() == [run.stem + store.INCOMPLETE]

    client = FakeClient()
    execute(run, store, client, max_workers=1, missing=True)
    record = store.record(run.stem)
    assert sum(client._seen.values()) == 8 and record["n"] == 20 and "incomplete" not in record
    earlier = [{key: sending.get(key) for key in ("saved_at", "model_resolved", "sending", "attempts")} for sending in (partial, again)]
    assert record["resumed"] == {"answers": 12, "sendings": earlier}
    assert [sending["sending"]["calls_sent"] for sending in earlier] == [7, 5] and record["sending"]["calls_sent"] == 8
    assert record["sending"]["answered"] == 8 and record["attempts"] == {}
    assert partial["saved_at"] <= again["saved_at"]
    archived = sorted(path.name for path in (store.run_dir(run.stem) / store.ARCHIVE_DIR).iterdir())
    assert store.runs() == [run.stem] and not store.run_dir(run.stem + store.INCOMPLETE).exists()
    earlier = [name for name in archived if store.INCOMPLETE in name]
    assert len(archived) == len(earlier) + 2 and len(earlier) in (2, 4) and any(name.endswith("_incomplete_run.json") for name in earlier)
    with pytest.raises(FileExistsError, match="whole already"):
        execute(run, store, FakeClient(), max_workers=1, missing=True)

    whole = ResultStore(tmp_path / "whole", SOD.topic)
    execute(run, whole, FakeClient(), max_workers=1)
    assert store.raw(run.stem) == whole.raw(run.stem) and store.frame(run.stem).equals(whole.frame(run.stem))
    assert check(store, run.stem, run.topic, run.framing).same_calls


def test_missing_refuses_another_run(tmp_path):
    run = prepared(BOUNDED)
    store = ResultStore(tmp_path, BOUNDED.topic)
    with pytest.raises(BatchFailed):
        execute(run, store, Broke(paid=2), max_workers=1)
    with pytest.raises(ValueError, match="different spec"):
        execute(prepared(BOUNDED.variant(sum_low=0.9)), store, FakeClient(), max_workers=1, missing=True)
    kept = Altered(scaled(0.5), subjects=5)
    execute(run, store, kept, max_workers=1, missing=True)
    assert len(store.record(run.stem)["faulty"]) == 3


class Broke(FakeClient):
    """A fake that runs out of credit after a number of calls."""

    def __init__(self, paid: int):
        super().__init__()
        self.paid = paid

    def call_llm(self, model, prompt, response_format, temperature, instructions=None):
        with self._lock:
            self.paid -= 1
            if self.paid < 0:
                raise BatchFailed("no credit")
        return super().call_llm(model, prompt, response_format, temperature, instructions)


def test_a_stopped_run_saves_what_came_back_and_raises_what_stopped_it(tmp_path):
    run = prepared(SOD)
    store = ResultStore(tmp_path, SOD.topic)
    with pytest.raises(BatchFailed, match="no credit"):
        execute(run, store, Broke(paid=7), max_workers=1)
    record = store.record(run.stem + store.INCOMPLETE)
    assert record["n"] == 7 and len(record["missing"]) == 13
    with pytest.raises(BatchFailed):
        execute(run, ResultStore(tmp_path / "nothing", SOD.topic), Broke(paid=0), max_workers=1)
    assert not (tmp_path / "nothing").exists()


def test_store_collision(tmp_path):
    store = ResultStore(tmp_path, SOD.topic)
    execute(prepared(SOD), store, FakeClient(), max_workers=4)
    other = prepared(SOD.variant(random_state=7))
    with pytest.raises(FileExistsError):
        execute(other, store, FakeClient(), max_workers=4)
    execute(other, store, FakeClient(), max_workers=4, overwrite=True)
    assert RunSpec.from_dict(store.record(other.stem)["spec"]).random_state == 7


@pytest.mark.parametrize("block", KEPT)
def test_a_run_brought_in_is_never_saved_over(tmp_path, block):
    store = ResultStore(tmp_path, SOD.topic)
    run = prepared(SOD)
    record = execute(run, store, FakeClient(), max_workers=4).record
    record.write_text(json.dumps({**json.loads(record.read_text(encoding="utf-8")), block: {}}), encoding="utf-8")
    with pytest.raises(FileExistsError):
        execute(run, store, FakeClient(fail_calls=len(run.calls)), max_workers=4, overwrite=True)


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


def test_a_dataset_without_the_vocabulary_reads_the_fallback():
    run = prepared(CITIZEN)
    assert run.stem == "regions_citizen_cs_default_Fake_fake-model_Prob_t=0.0_x3"
    assert set(run.rows["question"]) == {run.question.forms[run.question.fallback]}
    assert run.prompt(run.calls.iloc[0]) == f"Jsem obyvatel/ka Česka. {run.question.forms[run.question.fallback]}"


def test_standalone_description_is_the_whole_prompt(tmp_path):
    run = prepared(DIRECT)
    assert run.stem == "regions_direct_cs_Fake_fake-model_Prob_t=0.0_x3" and run.question is None
    assert run.prompt(run.calls.iloc[0]) == "Účast a výsledek voleb do poslanecké sněmovny v Česku v roce 2021 budou:"
    with pytest.raises(ValueError):
        prepared(DIRECT.variant(question="default"))
    store = ResultStore(tmp_path, DIRECT.topic)
    execute(run, store, FakeClient(), max_workers=4)
    assert len(store.frame(run.stem)) == 45 and store.record(run.stem)["question"] is None
    result = check(store, run.stem, run.topic, run.framing)
    assert result.answers == result.calls == 45 and result.same_calls and result.subjects == 15 and result.identical_prompts == 0


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
    with pytest.raises(ValueError):
        SOD.variant(effort="none")
    assert prepared(SOD.variant(client="OpenRouter", model="gpt-5", effort="minimal")).stem.endswith("_e=minimal_subset=20")


class Tiered(FakeClient):
    """A fake that was built with an effort and reports the levels its model takes."""

    def __init__(self, effort, levels):
        super().__init__()
        self.effort, self.levels, self.asked = effort, levels, []

    def efforts(self, model):
        self.asked.append(model)
        return self.levels


@pytest.mark.parametrize("levels, tier", [
    (("minimal", "low", "medium", "high"), {"level": "low", "rank": 2, "of": 4, "levels": ["minimal", "low", "medium", "high"]}),
    (None, {"level": "low", "rank": None, "of": None, "levels": None}),
], ids=["listed", "no list"])
def test_the_record_says_where_the_effort_stands(tmp_path, levels, tier):
    run = prepared(EFFORT)
    store = ResultStore(tmp_path, EFFORT.topic)
    client = Tiered("low", levels)
    execute(run, store, client, max_workers=2)
    assert store.record(run.stem)["effort"] == tier and client.asked == [EFFORT.model]


@pytest.mark.parametrize("built, levels", [("low", ("medium", "high")), ("low", ()), ("high", ("low", "high")), (None, ("low",))],
                         ids=["not a level", "takes none", "another effort", "no effort"])
def test_an_effort_the_model_does_not_take_is_refused_before_any_call(tmp_path, built, levels):
    run = prepared(EFFORT)
    store = ResultStore(tmp_path, EFFORT.topic)
    client = Tiered(built, levels)
    with pytest.raises(ValueError, match="effort"):
        execute(run, store, client, max_workers=2)
    assert not client._seen and not store.runs()


def test_a_run_without_an_effort_asks_nothing(tmp_path):
    run = prepared(SOD.variant(sample=5))
    store = ResultStore(tmp_path, SOD.topic)
    client = Tiered(None, ("low",))
    execute(run, store, client, max_workers=2)
    assert store.record(run.stem)["effort"] is None and client.asked == []


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
    with pytest.raises(ValueError):
        SOD.variant(sum_low=1.01)
    with pytest.raises(ValueError):
        SOD.variant(sum_high=0.99)


def test_presets_combine_into_complete_specs():
    presets = load_presets(PRESETS)
    assert set(DATA) | set(MODEL) | {"sample", "random_state"} == set(FIELDS)
    base = presets.base
    assert build_spec(base, None, None, {}).client == "Fake"
    for name, data in presets.data.items():
        spec = build_spec(base, data, None, {})
        assert spec.sample is None and prepared(spec).stem, name
    assert build_spec(base, presets.data["sod"], None, {"sample": 5, "question": "neutral"}).question == "neutral"
    stems = {prepared(build_spec(base, data, None, {})).stem for data in presets.data.values()}
    assert len(stems) == len(presets.data)
    for name, model in presets.models.items():
        assert prepared(build_spec(base, None, model, {"sample": 5})).spec.model == model["model"], name


def test_presets_lay_over_a_complete_base(tmp_path):
    base = {field: getattr(SOD, field) for field in DATA + MODEL}
    data, model = {"description": "extended"}, {"model": "other", "temperature": None}
    loaded = load_presets(write_presets(tmp_path, base, {"a": data}, {"m": model}))
    assert build_spec(loaded.base, None, None, {"sample": 20}) == SOD
    spec = build_spec(loaded.base, loaded.data["a"], loaded.models["m"], {"model": "third"})
    assert spec == SOD.variant(description="extended", model="third", temperature=None, sample=None)
    for bad_base, bad_data, bad_models in (
        ({field: base[field] for field in DATA}, {"a": data}, {"m": model}),
        ({**base, "sample": 5}, {"a": data}, {"m": model}),
        (base, {"a": {**data, "model": "fake-model"}}, {"m": model}),
        (base, {"a": {}}, {"m": model}),
        (base, {"a": data}, {"m": {**model, "sample": 5}}),
        (base, {"a": data}, {"m": {**model, "repeats": 2}}),
    ):
        with pytest.raises(ValueError):
            load_presets(write_presets(tmp_path, bad_base, bad_data, bad_models))


def test_parse_reads_what_a_person_typed():
    assert parse("sample", "none") is None and parse("sample", "200") == 200
    assert parse("temperature", "0.3") == 0.3 and parse("effort", "low") == "low"
    with pytest.raises(ValueError):
        parse("repeats", "many")
