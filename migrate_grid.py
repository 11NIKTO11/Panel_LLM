"""
Bring the grid runs of CoRe/_grid_separate into `results/` in today's shape (`DECISIONS.md` §15).

    python migrate_grid.py <grid_separate>                     lists what would be written
    python migrate_grid.py <grid_separate> --write             writes the runs
    python migrate_grid.py <grid_separate> --write --to <dir>  writes them under another results folder, for a trial

One-off, like the legacy migration: the old folder is only read, every run gets a new stem,
nothing is overwritten.

- ours: each saved repeat is parsed with today's response model, its saved prompt checked
  against today's, and the run saved through `ResultStore.save`; the record then gets the
  saving date, the schema as sent and a `migrated` block. A repeat whose raw JSON is gone is
  rebuilt from its table. The saved table must hold the old numbers exactly.
- the colleague's: delivered as per-profile means, no prompts, no framing, so nothing to
  resolve. Their files are written at the store's paths by hand: one row per profile (repeat
  1 for a run of several repeats), a record with a `delivered` block, and the delivered table
  itself copied to `delivered/`, which keeps what our shape has no column for (`_std`, `w_`).
"""
import argparse
import hashlib
import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import pandas as pd

from panel import paths
from panel.dataset import load_dataset, read_values
from panel.describe.template import load_template
from panel.run import ResultStore, RunSpec, resolve
from panel.run.resolve import REPEAT, SUBJECT, Resolved
from panel.run.record import DELIVERED
from panel.run.store import PROMPT
from panel.topic import load_topic
from panel.topic.normalize import normalize_groups

OLD_RESULTS = Path("results/election_2021")
DELIVERED_DIR = Path("colleague/data")
REPORT = "colleague/notes/EXPERIMENT_REPORT.md"
SENT_SCHEMA = paths.ROOT / "tests/fixtures/grid_schema_sent.json"
TAIL = "_OpenRouter_gpt-4o_VProb_t=0.3"


@dataclass(frozen=True)
class Ours:
    """One run of ours: its spec today, and the saved repeats in order, as old stems relative to the old results folder."""
    spec: RunSpec
    repeats: Tuple[str, ...]


@dataclass(frozen=True)
class Delivered:
    """One of the colleague's delivered tables, and the spec it amounts to."""
    file: str
    spec: RunSpec
    framing_short: str


def grid_spec(dataset: str, repeats: int) -> RunSpec:
    return RunSpec(dataset=dataset, description="default", topic="election_2021", framing="cs_grid", question="legacy",
                   client="OpenRouter", model="gpt-4o", shape="Prob", temperature=0.3, repeats=repeats, random_state=None)


def colleague_spec(topic: str, temperature: float, repeats: int) -> RunSpec:
    return RunSpec(dataset="grid_simple_en", description="delivered", topic=topic, framing="colleague", question=None,
                   client="OpenRouter", model="gpt-4o-2024-11-20", shape="Prob", temperature=temperature, repeats=repeats,
                   random_state=None)


OURS = (
    Ours(grid_spec("grid_sod_retired", 3), (f"archive/grid-sod-r1{TAIL}_2026-09-11-01-33-14", f"grid-sod-r2{TAIL}", f"grid-sod-r3{TAIL}")),
    Ours(grid_spec("grid_sod45_retired", 1), (f"grid-sod-r1{TAIL}",)),
    Ours(grid_spec("grid_simple", 1), (f"grid-simple{TAIL}",)),
)
COLLEAGUE = (
    Delivered("openai__gpt-4o-2024-11-20__income3__r3__t0_3.csv", colleague_spec("election_2021", 0.3, 3), "colleague"),
    Delivered("openai__gpt-4o-2024-11-20__media_fees_2026_income3.csv", colleague_spec("media_fees_2026", 0.0, 1), "colleague"),
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _repeat(old: Path, source: str, resolved: Resolved) -> Tuple[Dict[int, Tuple[str, object]], dict, str]:
    """Subject -> (saved prompt, answer) of one saved repeat, its old record, and where the answers came from."""
    topic, model = resolved.topic, resolved.response_model
    table = pd.read_csv(old / f"{source}.csv", index_col=0, encoding="utf-8")
    record = json.loads((old / f"{source}_run.json").read_text(encoding="utf-8"))
    raw = old / f"{source}.json"
    if raw.exists():
        items = json.loads(raw.read_text(encoding="utf-8"))
        answers = {int(key): (item["prompt"], model.model_validate(item["result"])) for key, item in items.items()}
        if any(answers[subject][0] != table.at[subject, PROMPT] for subject in answers):
            raise ValueError(f"{source}: the raw JSON and the table disagree on a prompt")
        return answers, record, "raw JSON"
    frame = topic.rename(table)
    voted, stayed = topic.turnout
    labels = resolved.framing.labels
    answers = {
        int(subject): (row[PROMPT], model.model_validate({
            "voted_or_not": {"voted": row[voted], "not_voted": row[stayed]},
            "options": [{"name": labels[option], "probability": row[option]} for option in topic.options],
        }))
        for subject, row in frame.iterrows()
    }
    return answers, record, "rebuilt from its table; its raw JSON was overwritten by a later run"


def migrate_ours(old: Path, run: Ours, store: ResultStore) -> str:
    """
    Save one of our runs under its new stem.

    Raises:
        ValueError: if a repeat's subjects, prompts or numbers are not what today's spec gives.
        FileExistsError: if the stem is already taken.
    """
    resolved = resolve(run.spec)
    if store.paths(resolved.stem).record.exists():
        raise FileExistsError(f"'{resolved.stem}' is already in {store.topic_dir}")
    repeats = [_repeat(old, source, resolved) for source in run.repeats]
    results = {}
    for call, row in resolved.calls.iterrows():
        prompt = resolved.prompt(row)
        saved, answer = repeats[row[REPEAT] - 1][0][row[SUBJECT]]
        if saved != prompt:
            raise ValueError(f"{run.repeats[row[REPEAT] - 1]}, subject {row[SUBJECT]}: the saved prompt is not today's")
        results[call] = (prompt, answer)
    if len(results) != sum(len(answers) for answers, _, _ in repeats):
        raise ValueError(f"'{resolved.stem}': the saved repeats hold other subjects than today's spec")
    written = store.save(resolved, results, timestamped=False)

    topic = resolved.topic
    new = store.frame(resolved.stem)
    for number, source in enumerate(run.repeats, start=1):
        was = topic.rename(pd.read_csv(old / f"{source}.csv", index_col=0, encoding="utf-8"))[topic.columns]
        now = (new[new[REPEAT] == number] if run.spec.repeats > 1 else new)[topic.columns]
        if not was.astype(float).equals(now.astype(float)):
            raise ValueError(f"'{resolved.stem}', repeat {number}: the table does not hold the numbers of {source}")

    record = json.loads(written.record.read_text(encoding="utf-8"))
    sent = json.loads(SENT_SCHEMA.read_text(encoding="utf-8"))
    record.update({
        "saved_at": max(old_record["saved_at"] for _, old_record, _ in repeats),
        "git_commit": None, "git_dirty": None, "versions": None, "model_resolved": None,
        "response_schema": sent[repeats[0][1]["spec"]["response_model"]],
        "migrated": {
            "from": "CoRe/_grid_separate",
            "repeats": [
                {"repeat": number, "stem": source, "saved_at": old_record["saved_at"], "git_commit": old_record.get("git_commit"), "answers": origin}
                for number, (source, (_, old_record, origin)) in enumerate(zip(run.repeats, repeats), start=1)
            ],
            "note": "each repeat was saved as a run of its own (the old code had no repeats); written again by migrate_grid.py. The git commits are of that folder's repository.",
        },
    })
    written.record.write_text(json.dumps(record, ensure_ascii=False, indent=4), encoding="utf-8")
    return resolved.stem


def colleague_stem(run: Delivered) -> str:
    """The stem `Resolved.stem` would give, for a run that cannot be resolved."""
    spec = run.spec
    dataset, description = load_dataset(spec.dataset), load_template(load_dataset(spec.dataset).description_path(spec.description))
    parts = [dataset.short, description.short, run.framing_short, spec.client, spec.model, spec.shape, f"t={round(float(spec.temperature), 2)}"]
    return "_".join(parts + ([f"x{spec.repeats}"] if spec.repeats > 1 else []))


def migrate_colleague(old: Path, run: Delivered, store: ResultStore) -> str:
    """
    Save one delivered table as a run: per-profile means, one row per profile.

    Raises:
        ValueError: if its profiles are not the grid's, or a profile has fewer valid repeats than the spec says.
        FileExistsError: if the stem is already taken.
    """
    spec, stem = run.spec, colleague_stem(run)
    written = store.paths(stem)
    if written.record.exists():
        raise FileExistsError(f"'{stem}' is already in {store.topic_dir}")
    topic, dataset = load_topic(spec.topic), load_dataset(spec.dataset)
    values = read_values(dataset)
    source = old / DELIVERED_DIR / run.file
    delivered = pd.read_csv(source, encoding="utf-8")
    ids = pd.Series(values.index, index=values.astype(str).agg("_".join, axis=1))
    if set(delivered["profile_id"]) != set(ids.index) or len(delivered) != len(ids):
        raise ValueError(f"{run.file}: its profiles are not the {len(ids)} of '{dataset.name}'")
    if (delivered["n_valid_repeats"] != spec.repeats).any() or delivered["invalid"].any():
        raise ValueError(f"{run.file}: a profile has other than {spec.repeats} valid repeats")

    frame = topic.rename(delivered.set_index(delivered["profile_id"].map(ids).rename(SUBJECT)).sort_index())
    if topic.turnout:
        voted, stayed = topic.turnout
        frame[stayed] = 1 - frame[voted]
    frame = frame[topic.columns].astype(float)
    if spec.repeats > 1:
        frame.insert(0, REPEAT, 1)
    frame[PROMPT] = None

    def result(row: pd.Series) -> dict:
        answer = {"options": [{"name": topic.names[option], "probability": row[option]} for option in topic.options]}
        if topic.turnout:
            answer = {"voted_or_not": {"voted": row[topic.turnout[0]], "not_voted": row[topic.turnout[1]]}, **answer}
        return answer

    key = (lambda subject: f"{subject}:1") if spec.repeats > 1 else str
    raw = {key(subject): {"prompt": None, "result": result(row)} for subject, row in frame.iterrows()}
    copy = store.run_dir(stem) / store.DELIVERED_DIR / run.file
    record = {
        "stem": stem, "n": len(frame), "saved_at": None, "spec": spec.to_dict(),
        "dataset": {"name": dataset.name, "short": dataset.short, "language": dataset.language,
                    "rows": [int(subject) for subject in frame.index], "values_sha256": _sha256(dataset.values_path)},
        "description": {"name": spec.description, "short": spec.description, "standalone": True, "text": []},
        "framing": None, "question": None, "response_schema": None, "model_resolved": None,
        "git_commit": None, "git_dirty": None, "versions": None,
        DELIVERED: {
            "by": "the colleague", "file": f"delivered/{run.file}", "sha256": _sha256(source), "report": f"{REPORT}, dated 2026-07-22",
            "model": "openai/gpt-4o-2024-11-20 through OpenRouter, as the report says",
            "averaged": spec.repeats,
            "note": (
                f"Per-profile means over {spec.repeats} valid repeats, as delivered: one row per profile, "
                + ("its repeat column 1 although it is the mean of all three; the spread over repeats is only in the delivered table (p_<option>_std). " if spec.repeats > 1 else "")
                + "Their prompts, system prompt and response schema were not delivered (the report names the output as "
                "turnout {voted, not_voted} and party_preferences keyed by option), so description and framing are names "
                "without text. Not Voted is 1 - p_voted; the w_ columns are p_<option> x p_voted and are dropped. "
                "The run's date is unknown; the report is of 2026-07-22."
            ),
        },
    }

    copy.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, copy)
    frame.to_csv(written.results_csv, index=True, encoding="utf-8")
    normalize_groups(frame, topic.normalize_groups).to_csv(written.normalized_csv, index=True, encoding="utf-8")
    written.results_json.write_text(json.dumps(raw, ensure_ascii=False, indent=4), encoding="utf-8")
    written.record.write_text(json.dumps(record, ensure_ascii=False, indent=4), encoding="utf-8")
    return stem


def main() -> None:
    parser = argparse.ArgumentParser(description="Bring the grid runs of CoRe/_grid_separate into results/.")
    parser.add_argument("grid_separate", type=Path, help="the old folder, CoRe/_grid_separate")
    parser.add_argument("--write", action="store_true", help="write the runs; without it only list them")
    parser.add_argument("--to", type=Path, help="another results folder to write under; results/ by default")
    args = parser.parse_args()

    old = args.grid_separate
    results_dir = args.to or paths.RESULTS_DIR
    for run in OURS:
        if not args.write:
            print(f"  {', '.join(run.repeats)}\n    -> {resolve(run.spec).stem}")
            continue
        print(f"  -> {migrate_ours(old / OLD_RESULTS, run, ResultStore(results_dir, run.spec.topic))}")
    for run in COLLEAGUE:
        if not args.write:
            print(f"  {run.file}\n    -> {colleague_stem(run)}")
            continue
        print(f"  -> {migrate_colleague(old, run, ResultStore(results_dir, run.spec.topic))}")
    if not args.write:
        print("--write writes them")


if __name__ == "__main__":
    main()
