"""The run layer, offline: a spec resolves, executes through the fake client, and its record round-trips."""
import json
import shutil

import pandas as pd
import pytest

from panel.analysis import validate
from panel.dataset import load_dataset, profile_path
from panel.run import ResultStore, RunSpec, execute, resolve
from panel.run.fake import FakeClient
from panel.run.registry import client_class

from tests.conftest import TOPICS_DIR, require

SOD = RunSpec(dataset="sod", description="self_v1", topic="election_2021", framing="cs",
              client="Fake", model="fake-model", shape="VProb", temperature=0.0, sample=20)
REGIONS = SOD.variant(dataset="regions", description="latent_citizen_v1", sample=None, repeats=3)
SIX = ("results_csv", "results_json", "normalized_csv", "record")


def prepared(spec: RunSpec):
    require(profile_path(load_dataset(spec.dataset)))
    return resolve(spec)


def files_of(store: ResultStore, stem: str):
    plain = store.paths(stem)
    archived = list((store.topic_dir / store.ARCHIVE_DIR).glob(f"{stem}_*"))
    return [getattr(plain, name) for name in SIX], archived


def test_stem_encodes_every_axis():
    assert prepared(SOD).stem == "sod_v1_cs_Fake_fake-model_VProb_t=0.0_subset=20"
    assert prepared(REGIONS).stem == "regions_citizen-v1_cs_Fake_fake-model_VProb_t=0.0_x3"
    assert prepared(SOD.variant(shape="VPick", temperature=0.3, sample=None)).stem == "sod_v1_cs_Fake_fake-model_VPick_t=0.3"
    assert prepared(SOD.variant(temperature=None)).stem == "sod_v1_cs_Fake_fake-model_VProb_t=default_subset=20"


def test_execute_offline(tmp_path):
    run = prepared(SOD)
    store = ResultStore(tmp_path, SOD.topic)
    paths = execute(run, store, FakeClient(), max_workers=4)

    plain, archived = files_of(store, run.stem)
    assert all(path.exists() for path in plain) and len(archived) == 2
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
    assert sorted(record["dataset"]["rows"]) == sorted(int(i) for i in run.rows.index)
    assert record["model_resolved"] == "fake-model-fake" and "pydantic" in record["versions"]

    loaded = store.load(run.stem, run.response_model)
    assert set(loaded) == {(int(i), 1) for i in run.rows.index}
    bad_turnout, bad_parties, duplicates = validate.evaluate_voting_results(loaded)
    assert not bad_turnout and not bad_parties and not duplicates
    assert store.runs() == [run.stem] and paths.results_csv.exists()


def test_execute_repeats(tmp_path):
    run = prepared(REGIONS)
    store = ResultStore(tmp_path, REGIONS.topic)
    execute(run, store, FakeClient(), max_workers=4)
    frame = store.frame(run.stem)
    assert len(frame) == 45 and list(frame.columns) == ["repeat"] + run.topic.columns + ["Prompt"]
    assert frame.groupby(level=0)["repeat"].apply(list).map(lambda r: r == [1, 2, 3]).all()
    loaded = store.load(run.stem, run.response_model)
    assert len(loaded) == 45 and (1, 3) in loaded
    raw = json.loads(store.paths(run.stem).results_json.read_text(encoding="utf-8"))
    assert "1:1" in raw and set(raw["1:1"]) == {"prompt", "result"}


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


def test_resolve_language_mismatch(tmp_path):
    shutil.copytree(TOPICS_DIR / "election_2021", tmp_path / "election_2021")
    english = json.loads((tmp_path / "election_2021" / "cs.json").read_text(encoding="utf-8"))
    english.update(name="en", language="en", short="en")
    (tmp_path / "election_2021" / "en.json").write_text(json.dumps(english, ensure_ascii=False), encoding="utf-8")
    with pytest.raises(ValueError):
        resolve(SOD.variant(framing="en"), topics_dir=tmp_path)


def test_resolve_refuses_undeclared_description():
    with pytest.raises(ValueError):
        prepared(SOD.variant(dataset="grid", description="latent_citizen_v1"))


def test_standalone_description_appends_no_question():
    run = prepared(REGIONS.variant(description="latent_direct_v1"))
    assert run.question == "" and run.prompt(run.rows.iloc[0]).endswith("budou:")
    assert prepared(REGIONS).question.startswith(" Ve volbách")


def test_registry():
    assert client_class("OpenRouter") is client_class("OpenRouterClient")
    assert client_class("Fake") is FakeClient
    with pytest.raises(KeyError):
        client_class("Nobody")
    with pytest.raises(KeyError):
        prepared(SOD.variant(client="Nobody"))


def test_spec_rejects_nonsense():
    with pytest.raises(ValueError):
        SOD.variant(shape="VWhat")
    with pytest.raises(ValueError):
        SOD.variant(repeats=0)
    with pytest.raises(TypeError):
        RunSpec.from_dict({"name": "sod", "topic": "election_2021_legacy"})
