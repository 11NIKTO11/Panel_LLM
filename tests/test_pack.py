"""A pack: the run, the files it names, and a manifest whose hashes are those of the files."""
import json
import shutil

import pytest

from panel.digest import content_sha256, file_sha256
from panel.pack import DATASET_DIR, MANIFEST_NAME, RUN_DIR, TOPIC_DIR, archive, pack, verify
from panel.run import ResultStore

from tests.conftest import RESULTS_DIR, require

TOPIC = "election_2021"
STEM = "regions_citizen_cs-legacy_legacy_OpenAI_gpt-4o_Prob_t=0.0_x10"


@pytest.fixture(scope="module")
def store() -> ResultStore:
    found = ResultStore(RESULTS_DIR, TOPIC)
    require(found.paths(STEM).record)
    return found


def test_a_pack_holds_the_run_and_what_it_names(store, tmp_path):
    folder = pack(store, STEM, tmp_path, derive=False)
    manifest = json.loads((folder / MANIFEST_NAME).read_text(encoding="utf-8"))
    files = manifest["files"]
    assert folder == tmp_path / TOPIC / STEM and manifest["missing"] == [] and manifest["unverified"] == []
    for name in (f"{RUN_DIR}/results.csv", f"{RUN_DIR}/run.json", f"{DATASET_DIR}/prepared/values.csv",
                 f"{DATASET_DIR}/prepared/prompts/citizen.csv", f"{DATASET_DIR}/descriptions/citizen.json",
                 f"{TOPIC_DIR}/topic.json", f"{TOPIC_DIR}/framings/cs_legacy.json", f"{TOPIC_DIR}/questions/legacy.json",
                 f"{TOPIC_DIR}/reference.csv"):
        assert files[name]["sha256"] == file_sha256(folder / name) and "from" in files[name]
    assert all(entry["sha256"] == content_sha256(folder / entry["file"]) for entry in manifest["built_from"].values())
    assert verify(folder) == []
    assert archive(folder).stat().st_size > 0


def test_a_changed_or_stray_file_is_found(store, tmp_path):
    folder = pack(store, STEM, tmp_path, derive=False)
    (folder / RUN_DIR / "results.csv").write_text("changed", encoding="utf-8")
    (folder / "stray.txt").write_text("stray", encoding="utf-8")
    assert verify(folder) == [f"{RUN_DIR}/results.csv", "stray.txt"]


def test_derived_tables_and_figures_are_made_in_the_pack(store, tmp_path):
    before = {path: path.stat().st_mtime_ns for path in store.run_dir(STEM).rglob("*") if path.is_file()}
    folder = pack(store, STEM, tmp_path)
    manifest = json.loads((folder / MANIFEST_NAME).read_text(encoding="utf-8"))
    made = [name for name, entry in manifest["files"].items() if "from" not in entry]
    assert any(name.startswith(f"{RUN_DIR}/plots/") for name in made) and any(name.startswith(f"{RUN_DIR}/analysis/") for name in made)
    assert before == {path: path.stat().st_mtime_ns for path in store.run_dir(STEM).rglob("*") if path.is_file()}
    assert verify(folder) == []


def test_a_pack_is_not_written_over(store, tmp_path):
    pack(store, STEM, tmp_path, derive=False)
    with pytest.raises(FileExistsError):
        pack(store, STEM, tmp_path, derive=False)


def test_a_run_built_from_other_data_is_refused(store, tmp_path):
    results = tmp_path / "results"
    copied = ResultStore(results, TOPIC)
    shutil.copytree(store.run_dir(STEM), copied.run_dir(STEM))
    path = copied.paths(STEM).record
    record = json.loads(path.read_text(encoding="utf-8"))
    record["dataset"]["prompts_sha256"] = "0" * 64
    path.write_text(json.dumps(record, ensure_ascii=False), encoding="utf-8")
    with pytest.raises(ValueError, match="prompts/citizen.csv"):
        pack(copied, STEM, tmp_path / "packs", derive=False)
    assert not (tmp_path / "packs").exists()
