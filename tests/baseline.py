"""
Write the phase 0 baseline fixtures from the code as it stands.

    python -m tests.baseline [--overwrite]

Run from the repo root, before any code change. Refuses to overwrite an existing fixture
set unless told to: a baseline is taken once, and retaking it is a decision to record in
the commit that does it.

Writes under tests/fixtures/:
    manifest.json             when, from which commit, with which library versions
    prompt_tables.json        row count and digest per stored description table, after
                              checking that rendering from clean_data.csv reproduces it
    schemas/<topic>.json      the response schemas a run on that topic is shown; for the
                              legacy topic the frozen copy from its results config.json
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

from datasets.sod.code.describe import render
from utils.run import ResultStore
from utils.topic import load_topic
from utils.voting import make_response_models

from tests.conftest import FIXTURES, LEGACY_TOPIC, RESULTS_DIR, ROOT, TOPICS_DIR, topic_names
from tests.snapshots import (
    aggregate_all, frame_digest, json_shape, models_by_name, prompt_table_digest, reparse,
    schema_snapshot,
)

VARIANTS = ("self_description_v1", "self_description_v2")
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
    Digest each stored table, after rendering it again from clean_data.csv.

    Raises:
        RuntimeError: if rendering does not reproduce the stored table exactly.
    """
    people = pd.read_csv(ROOT / "data" / "clean_data.csv", index_col="id")
    out = {}
    for variant in VARIANTS:
        rendered = render(people, variant)
        stored = pd.read_csv(ROOT / "data" / "prompts" / f"{variant}.csv", index_col=0).iloc[:, 0]
        if not rendered.index.equals(stored.index):
            raise RuntimeError(f"{variant}: rendered index differs from the stored table")
        differing = int((rendered.values != stored.values).sum())
        if differing:
            raise RuntimeError(f"{variant}: rendering differs from the stored table on {differing} rows")
        out[variant] = prompt_table_digest(stored)
    return out


def schemas() -> Dict[str, Dict]:
    """Today's generated schemas per topic; the legacy topic gets what its runs were actually shown."""
    out = {name: schema_snapshot(*make_response_models(load_topic(name, TOPICS_DIR))) for name in topic_names()}
    config = json.loads((RESULTS_DIR / LEGACY_TOPIC / "config.json").read_text(encoding="utf-8"))
    out[LEGACY_TOPIC] = config["schemas"]
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
