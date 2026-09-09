"""
Write the baseline fixtures from the code as it stands.

    python -m tests.baseline [--overwrite]

Taken once, before the overhaul changed any code (phase 0). Refuses to overwrite an existing
fixture set unless told to: retaking a baseline is a decision to record in the commit that
does it.

Writes under tests/fixtures/:
    manifest.json             when, from which commit, with which library versions
    prompt_tables.json        row count and digest per stored prompt table, keyed
                              <dataset>/<description>, after checking that rendering from
                              the dataset's profile reproduces it
    schemas/<topic>/<framing>.json   the response schemas a run on that topic under that framing
                              is shown; for the legacy topic the frozen copy from its results config.json
    legacy_aggregates.json    country-level aggregation per legacy run, per CSV, for all
                              four weighted x normalized combinations
    legacy_reparse.json       per legacy run: response model, JSON shape and a digest of the
                              frame the raw JSON reparses to, after checking it equals the CSV
"""
import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import pydantic

from panel.dataset import dataset_names, load_dataset, profile_path, read_profile
from panel.describe.render import render
from panel.describe.tables import read_table
from panel.describe.template import load_description
from panel.run import ResultStore
from panel.topic import framing_names, load_framing, load_topic, make_response_models, topic_names

from tests.conftest import FIXTURES, LEGACY_TOPIC, RESULTS_DIR, ROOT, TOPICS_DIR
from tests.snapshots import (
    aggregate_all, frame_digest, json_shape, models_by_name, prompt_table_digest, reparse,
    schema_snapshot,
)

ATOL = 1e-12


def _git_commit() -> str:
    done = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=ROOT)
    return done.stdout.strip()


def _write(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {path.relative_to(ROOT).as_posix()}")


def prompt_tables() -> Dict:
    """
    Digest each stored prompt table of each prepared dataset, after rendering it again.

    Raises:
        RuntimeError: if rendering does not reproduce a stored table exactly.
    """
    out = {}
    for name in dataset_names():
        dataset = load_dataset(name)
        if not dataset.descriptions or not profile_path(dataset).exists():
            continue
        profile = read_profile(dataset)
        for description in dataset.descriptions:
            rendered = render(dataset, load_description(description), profile)
            stored = read_table(dataset.name, description).iloc[:, 0]
            if not rendered.index.equals(stored.index) or int((rendered.values != stored.values).sum()):
                raise RuntimeError(f"{dataset.name}/{description}: rendering differs from the stored table")
            out[f"{dataset.name}/{description}"] = prompt_table_digest(stored)
    return out


def schemas() -> Dict[str, Dict]:
    """Today's generated schemas per topic and framing, keyed <topic>/<framing>; the legacy topic gets what its runs were shown."""
    out = {}
    for name in topic_names(TOPICS_DIR):
        topic = load_topic(name, TOPICS_DIR)
        for framing_name in framing_names(name, TOPICS_DIR):
            framing = load_framing(topic, framing_name, TOPICS_DIR)
            out[f"{name}/{framing_name}"] = schema_snapshot(*make_response_models(topic, framing))
    config = json.loads((RESULTS_DIR / LEGACY_TOPIC / "config.json").read_text(encoding="utf-8"))
    out[f"{LEGACY_TOPIC}/cs"] = config["schemas"]
    return out


def legacy_runs() -> Tuple[Dict, Dict]:
    """
    Aggregates for every legacy run, and a reparse digest for every run that has raw JSON.

    Raises:
        RuntimeError: if a run's raw JSON does not reparse to its stored CSV.
    """
    topic = load_topic(LEGACY_TOPIC, TOPICS_DIR)
    models = models_by_name(topic)
    store = ResultStore(RESULTS_DIR, LEGACY_TOPIC)
    aggregates, reparsed = {}, {}
    for stem in store.runs():
        record = store.record(stem)
        paths = store.paths(stem)
        aggregates[stem] = {"csv": aggregate_all(store.frame(stem), topic)}
        if paths.normalized_csv.exists():
            aggregates[stem]["normalized_csv"] = aggregate_all(store.frame(stem, normalized=True), topic)

        entry = {"rows": int(record["n"]), "response_model": record["spec"]["response_model"], "json": None}
        if paths.results_json.exists():
            raw = json.loads(paths.results_json.read_text(encoding="utf-8"))
            frame = reparse(raw, models[entry["response_model"]], topic.columns)
            stored = store.frame(stem)[topic.columns]
            same_rows = frame.index.equals(stored.index)
            if not same_rows or not np.allclose(frame.to_numpy(dtype=float), stored.to_numpy(dtype=float), rtol=0, atol=ATOL):
                raise RuntimeError(f"{stem}: raw JSON does not reparse to its CSV")
            entry["json"] = {"shape": json_shape(raw), "rows": int(len(frame)), "sha256": frame_digest(frame)}
        reparsed[stem] = entry
    return aggregates, reparsed


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("--overwrite", action="store_true", help="replace an existing fixture set")
    args = parser.parse_args(argv)
    if (FIXTURES / "manifest.json").exists() and not args.overwrite:
        sys.exit(f"{FIXTURES} already holds a baseline; pass --overwrite to retake it")

    _write(FIXTURES / "prompt_tables.json", prompt_tables())
    for name, snapshot in schemas().items():
        _write(FIXTURES / "schemas" / f"{name}.json", snapshot)
    aggregates, reparsed = legacy_runs()
    _write(FIXTURES / "legacy_aggregates.json", aggregates)
    _write(FIXTURES / "legacy_reparse.json", reparsed)
    _write(FIXTURES / "manifest.json", {
        "taken_at": datetime.now().isoformat(timespec="seconds"),
        "git_commit": _git_commit(),
        "python": sys.version.split()[0],
        "pandas": pd.__version__,
        "pydantic": pydantic.__version__,
        "legacy_schema_source": f"results/{LEGACY_TOPIC}/config.json",
    })


if __name__ == "__main__":
    main()
