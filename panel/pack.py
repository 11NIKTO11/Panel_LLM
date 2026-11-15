"""
Pack a saved run: one folder holding the run and every file it was built from, each with its hash.

    packs/<topic>/<stem>/
    |-- MANIFEST.json       what the pack is; per file where it came from, its sha256 and its size
    |-- run/                the run's folder as saved; with `derive`, its analysis/ and plots/ made again
    |-- dataset/            dataset.json, forms.json, descriptions/<description>.json, claims/<topic>.json,
    |                       prepared/values.csv, prepared/prompts/<description>.csv
    `-- topic/              topic.json, framings/<framing>.json, questions/<question>.json, the reference
                            results, plot.json, polls.json, positions.json

`dataset/` and `topic/` keep the paths the files have under `datasets/<name>/` and
`config/topics/<topic>/`; only what this run names is taken. The raw source of a dataset and
the code are not packed: the record names the commit.

A run's record carries the sha256 of the values, the prompt table and the forms it was built
from. A pack is refused when one of those files is no longer the one recorded - it would set a
run beside data it was not built from. A file the run names but that does not exist (the
prompts of a run someone else sent) is listed as missing.
"""
import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Union

from panel import paths
from panel.analysis import ANALYSES, NotApplicable, Output, load_run
from panel.dataset import CONFIG_NAME as DATASET_NAME
from panel.dataset import Dataset, load_dataset
from panel.digest import recorded, sha256
from panel.run import ResultStore, RunSpec
from panel.run.record import git_state
from panel.topic import Topic, load_topic, plot, polls, positions
from panel.topic.framing import FRAMINGS_DIR
from panel.topic.question import QUESTIONS_DIR
from panel.topic.vocabulary import CONFIG_NAME as TOPIC_NAME

MANIFEST_NAME = "MANIFEST.json"
RUN_DIR = "run"
DATASET_DIR = "dataset"
TOPIC_DIR = "topic"


def _origin(path: Path) -> str:
    """A path as the manifest names it: relative to the repo when inside it."""
    path = path.resolve()
    return (path.relative_to(paths.ROOT) if path.is_relative_to(paths.ROOT) else path).as_posix()


def _named(dataset: Dataset, topic: Topic, spec: RunSpec) -> Dict[str, Path]:
    """Pack path -> the file of the dataset or the topic that the run names, existing or not."""
    of_dataset = [dataset.dir / DATASET_NAME, dataset.description_path(spec.description), *dataset.built_from(spec.description).values()]
    if topic.name in dataset.claims:
        of_dataset.append(dataset.dir / dataset.claims[topic.name].map)
    of_topic = [topic.dir / TOPIC_NAME, topic.dir / FRAMINGS_DIR / f"{spec.framing}.json"]
    of_topic += [topic.dir / module.CONFIG_NAME for module in (plot, polls, positions)]
    if spec.question:
        of_topic.append(topic.dir / QUESTIONS_DIR / f"{spec.question}.json")
    if topic.reference:
        of_topic.append(topic.dir / topic.reference.path)
    named = {f"{DATASET_DIR}/{path.relative_to(dataset.dir).as_posix()}": path for path in of_dataset}
    named.update({f"{TOPIC_DIR}/{path.relative_to(topic.dir).as_posix()}": path for path in of_topic})
    return named


def pack(
    store: ResultStore, stem: str, to: Union[str, Path] = paths.PACKS_DIR, derive: bool = True,
    datasets_dir: Union[str, Path] = paths.DATASETS_DIR, topics_dir: Union[str, Path] = paths.TOPICS_DIR,
) -> Path:
    """
    Write one run's pack.

    Args:
        store: the topic's results.
        stem: the run.
        to: the folder of packs; the pack is `<to>/<topic>/<stem>/`.
        derive: run every analysis over the run and write its tables and figures into the
            pack's `run/`, over the copies of those saved with the run.

    Returns:
        the pack's folder.

    Raises:
        FileNotFoundError: if the run has no record.
        FileExistsError: if the pack's folder exists; a pack is never written over.
        ValueError: if a file the run was built from is not the one its record gives the hash of.
    """
    record = store.record(stem)
    spec = RunSpec.from_dict(record["spec"])
    dataset, topic = load_dataset(spec.dataset, datasets_dir), load_topic(spec.topic, topics_dir)
    folder = Path(to) / store.topic / stem
    if folder.exists():
        raise FileExistsError(f"{folder} exists; a pack is not written over - remove it or pack elsewhere")

    digests = recorded(record["dataset"])
    files = {name: path for name, path in dataset.built_from(spec.description).items() if digests.get(name) and path.exists()}
    changed = sorted(_origin(path) for name, path in files.items() if sha256(path) != digests[name])
    if changed:
        raise ValueError(f"'{stem}' was built from other files than today's {changed}: its record gives another sha256")

    named = _named(dataset, topic, spec)
    run_dir = store.run_dir(stem)
    copies = {f"{RUN_DIR}/{path.relative_to(run_dir).as_posix()}": path for path in sorted(run_dir.rglob("*")) if path.is_file()}
    copies.update({name: path for name, path in named.items() if path.exists()})
    for name, source in copies.items():
        target = folder / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)

    skipped = {}
    if derive:
        out = Output(folder / RUN_DIR)
        run = load_run(store, stem, datasets_dir=datasets_dir, topics_dir=topics_dir)
        for name, analysis in ANALYSES.items():
            try:
                analysis(run, out)
            except NotApplicable as reason:
                skipped[name] = str(reason)
        for path in out.written:
            copies.pop(path.relative_to(folder).as_posix(), None)

    manifest: Dict[str, Any] = {
        "stem": stem,
        "topic": store.topic,
        "packed_at": datetime.now().isoformat(timespec="seconds"),
        **git_state(),
        "spec": record["spec"],
        "built_from": {name: {"file": f"{DATASET_DIR}/{path.relative_to(dataset.dir).as_posix()}", "sha256": digests[name]} for name, path in files.items()},
        "unverified": sorted(set(dataset.built_from(spec.description)) - set(files)),
        "missing": sorted(name for name, path in named.items() if not path.exists()),
        "derived": derive,
        "not_applicable": skipped,
        "files": {
            name: {"sha256": sha256(path), "bytes": path.stat().st_size, **({"from": _origin(copies[name])} if name in copies else {})}
            for name, path in ((path.relative_to(folder).as_posix(), path) for path in sorted(folder.rglob("*")) if path.is_file())
        },
    }
    (folder / MANIFEST_NAME).write_text(json.dumps(manifest, ensure_ascii=False, indent=4), encoding="utf-8")
    return folder


def verify(folder: Union[str, Path]) -> List[str]:
    """The files of a pack that are not what its manifest says: changed, gone, or not listed."""
    folder = Path(folder)
    listed = json.loads((folder / MANIFEST_NAME).read_text(encoding="utf-8"))["files"]
    found = {path.relative_to(folder).as_posix(): path for path in folder.rglob("*") if path.is_file() and path.name != MANIFEST_NAME}
    wrong = [name for name, entry in listed.items() if name not in found or sha256(found[name]) != entry["sha256"]]
    return sorted(wrong + [name for name in found if name not in listed])


def archive(folder: Union[str, Path]) -> Path:
    """The pack as `<folder>.zip` beside it."""
    folder = Path(folder)
    return Path(shutil.make_archive(str(folder), "zip", folder.parent, folder.name))
