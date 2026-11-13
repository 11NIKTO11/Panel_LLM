"""
A legacy run written again as a run of today holds the same numbers and is read by the analysis.

Needs runs still in the legacy shape; where they were all migrated, these tests skip.
"""
import json

import pytest

from panel.analysis import find_stem, load_run
from panel.dataset import load_dataset, read_values
from panel.run import ResultStore, resolve
from panel.run.check import check
from panel.run.legacy import Legacy, legacy_stems, migrate, spec_of

from tests.conftest import require

LEGACY = Legacy(dataset="sod", description="legacy", framing="cs", question="default")
SMALL = "sod_OpenRouter_claude-sonnet-4-5_VProb_t=0.0_subset=200"
PICK = "sod_OpenAI_gpt-4.1_VPick_t=0.0"


@pytest.fixture(scope="module")
def sod():
    dataset = load_dataset("sod")
    require(dataset.values_path)
    require(dataset.prompts_path("legacy"))
    return dataset


def unmigrated(store, stem):
    if stem not in legacy_stems(store):
        pytest.skip(f"'{stem}' is not in {store.topic_dir} in the legacy shape")


def test_a_legacy_record_amounts_to_a_spec(legacy_store):
    unmigrated(legacy_store, SMALL)
    unmigrated(legacy_store, PICK)
    spec = spec_of(legacy_store.record(SMALL), LEGACY)
    assert (spec.client, spec.model, spec.shape, spec.sample, spec.random_state) == ("OpenRouter", "claude-sonnet-4-5", "Prob", 200, 42)
    assert spec_of(legacy_store.record(PICK), LEGACY).random_state is None


@pytest.mark.parametrize("stem", [SMALL, PICK])
def test_a_migrated_run_is_a_run_of_today(stem, legacy_store, legacy_topic, sod, tmp_path):
    unmigrated(legacy_store, stem)
    target = ResultStore(tmp_path, legacy_topic.name)
    done = migrate(legacy_store, stem, target, LEGACY)
    assert done.source == stem and done.stem in target.runs() and done.normalized_differs in (0, None)
    assert legacy_stems(target) == []
    with pytest.raises(FileExistsError):
        migrate(legacy_store, stem, target, LEGACY)

    record = target.record(done.stem)
    assert record["migrated"]["from"] == stem and record["saved_at"] == legacy_store.record(stem)["saved_at"]
    frozen = json.loads(legacy_store.path("config.json").read_text(encoding="utf-8"))["schemas"]
    assert record["response_schema"] in frozen.values()

    ids = read_values(sod).index
    old = legacy_topic.rename(legacy_store.frame(stem))[legacy_topic.columns]
    run = load_run(target, find_stem(target, done.stem), styled=False)
    assert run.results.loc[ids[old.index]].to_numpy(dtype=float).tolist() == old.to_numpy(dtype=float).tolist()
    assert set(run.subjects.index) == set(ids[old.index])
    found = check(resolve(run.spec), target)
    assert found.same_calls and found.answers == len(old) and found.missing == 0
