"""
Runs saved in the legacy shape, written again as runs of today.

A legacy run is keyed by the subject's row position in the table of its day, its columns
are native names, its record names five fields that are no longer a spec's, and its raw JSON
lists `parties`. Migrating one parses its answers with today's response model and saves them
through `ResultStore.save`, so the files are exactly what a run sent today leaves:
keyed by subject id, option ids for columns, `options` in the JSON, a record a spec reads back.

What a migrated run cannot have is filled from the legacy run, never made up:

- its description is the dataset's stored description: the prompts as they were sent;
- `saved_at` is the legacy run's; the code version and SDK versions are unknown and empty;
- `response_schema` is the schema that was sent, from the legacy folder's frozen `config.json`;
- `migrated` says which legacy run it came from.

The legacy files are read and left as they are.
"""
import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Dict, List, Tuple, Union

import pandas as pd
from pydantic import BaseModel

from panel.dataset import load_dataset, read_values

from .resolve import SUBJECT, Resolved, resolve
from .spec import RunSpec
from .store import PROMPT, ResultStore

CLIENTS = {"OpenAIClient": "OpenAI", "OpenRouterClient": "OpenRouter"}
SHAPES = {"VotingProbabilities": "Prob", "VotingPick": "Pick"}
FROZEN = "config.json"


@dataclass(frozen=True)
class Legacy:
    """
    What the legacy runs of one topic are migrated with.

    Attributes:
        dataset: the dataset whose subjects they asked; its values are in the legacy row order.
        description: the dataset's stored description holding the prompts as sent.
        framing: the topic's framing that words the legacy schema.
        question: the question the legacy prompts ended with.
        random_state: the seed a legacy sample was drawn with.
    """
    dataset: str
    description: str
    framing: str
    question: str
    random_state: int = 42


@dataclass(frozen=True)
class Migrated:
    """
    Attributes:
        source: the legacy stem.
        stem: the stem it was saved under.
        answers: how many answers it holds.
        prompts_checked: answers whose saved prompt was compared with the stored description; 0 when the legacy run saved none.
        normalized_differs: cells of the legacy normalized table that today's differs from, or None when the legacy run had none.
    """
    source: str
    stem: str
    answers: int
    prompts_checked: int
    normalized_differs: Union[int, None]


def legacy_stems(store: ResultStore) -> List[str]:
    """The store's runs whose record is of the legacy shape."""
    return [stem for stem in store.runs() if "description" not in store.record(stem)["spec"]]


def spec_of(record: Dict[str, Any], legacy: Legacy) -> RunSpec:
    """
    The spec a legacy record amounts to.

    Raises:
        KeyError: on a client or response model the legacy runs did not have.
    """
    old = record["spec"]
    return RunSpec(
        dataset=legacy.dataset, description=legacy.description, topic=old["topic"], framing=legacy.framing, question=legacy.question,
        client=CLIENTS[old["client_type"]], model=old["model"], shape=SHAPES[old["response_model"]], temperature=old["temperature"],
        sample=old["sample"], random_state=legacy.random_state if old["sample"] is not None else None,
    )


def write_stored_prompts(source: Path, legacy: Legacy, question: str) -> Path:
    """
    The dataset's stored description, from the legacy prompt table: each prompt without the
    question it ended with, keyed by the subject id at its row position.

    Args:
        source: the legacy prompt table, one prompt per row in the legacy row order.
        legacy: names the dataset and its stored description.
        question: the text every legacy prompt ended with.

    Raises:
        ValueError: if the table is not one prompt per subject, a prompt does not end with
            the question, or a different table is already stored.
    """
    dataset = load_dataset(legacy.dataset)
    ids = read_values(dataset).index
    prompts = pd.read_csv(source, index_col=0, encoding="utf-8").iloc[:, 0]
    ending = f" {question}"
    if len(prompts) != len(ids) or not prompts.str.endswith(ending).all():
        raise ValueError(f"{source} is not one prompt per subject of '{dataset.name}' ending with the question")
    table = pd.Series(prompts.str[: -len(ending)].to_numpy(), index=ids, name=legacy.description).to_frame()
    path = dataset.prompts_path(legacy.description)
    if path.exists():
        if not pd.read_csv(path, index_col=0, encoding="utf-8").equals(table):
            raise ValueError(f"{path} already holds a different table")
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(path, index=True, encoding="utf-8")
    return path


def _answers(store: ResultStore, stem: str, resolved: Resolved) -> Dict[int, Tuple[Union[str, None], BaseModel]]:
    """Row position -> (saved prompt or None, the answer parsed with today's model): from the raw JSON, or from the table for a run that has none."""
    model = resolved.response_model
    if store.paths(stem).results_json.exists():
        return {position: answer for (position, _), answer in store.load(stem, model).items()}
    topic, labels = resolved.topic, resolved.framing.labels
    frame = topic.rename(store.frame(stem))
    voted, stayed = topic.turnout
    chosen = frame[list(topic.options)].idxmax(axis=1)
    return {
        position: (None, model.model_validate({"voted_or_not": {"voted": row[voted], "not_voted": row[stayed]}, "choice": labels[chosen[position]]}))
        for position, row in frame.iterrows()
    }


def migrate(source: ResultStore, stem: str, target: ResultStore, legacy: Legacy) -> Migrated:
    """
    Save one legacy run of `source` into `target` as a run of today.

    Args:
        source: the store holding the legacy run; only read.
        stem: the legacy run.
        target: where the migrated run goes; may be the same folder, the stems differ.
        legacy: what the legacy runs are migrated with.

    Raises:
        ValueError: if the run's subjects are not the ones its spec resolves to, a saved
            prompt is not the stored description with the question, or the saved numbers
            do not come back as they were.
        FileExistsError: if the target already holds the migrated stem.
    """
    record = source.record(stem)
    spec = spec_of(record, legacy)
    resolved = resolve(spec)
    ids = read_values(resolved.dataset).index
    answers = _answers(source, stem, resolved)
    if {ids[position] for position in answers} != set(resolved.rows.index):
        raise ValueError(f"'{stem}': its {len(answers)} subjects are not the ones {spec} resolves to")
    if target.paths(resolved.stem).record.exists():
        raise FileExistsError(f"'{resolved.stem}' is already in {target.topic_dir}")

    calls, position_of = resolved.calls, {subject: position for position, subject in enumerate(ids)}
    results, checked = {}, 0
    for call, row in calls.iterrows():
        saved, answer = answers[position_of[row[SUBJECT]]]
        prompt = resolved.prompt(row)
        if saved is not None:
            if saved != prompt:
                raise ValueError(f"'{stem}', subject {row[SUBJECT]}: the saved prompt is not the stored description with the question")
            checked += 1
        results[call] = (prompt, answer)
    paths = target.save(resolved, results, timestamped=False)

    topic = resolved.topic
    old = topic.rename(source.frame(stem))[topic.columns].set_axis(ids[source.frame(stem).index]).sort_index()
    new = target.frame(resolved.stem)[topic.columns].sort_index()
    if not old.astype(float).equals(new.astype(float)):
        raise ValueError(f"'{stem}': the migrated table does not hold the numbers of the legacy one")
    differs = None
    if source.paths(stem).normalized_csv.exists():
        was = topic.rename(source.frame(stem, normalized=True))[topic.columns].set_axis(ids[source.frame(stem, normalized=True).index]).sort_index()
        now = target.frame(resolved.stem, normalized=True)[topic.columns].sort_index()
        differs = int((~((was - now).abs() < 1e-12) & ~(was.isna() & now.isna())).sum().sum())

    frozen = json.loads(source.path(FROZEN).read_text(encoding="utf-8"))
    saved_record = json.loads(paths.record.read_text(encoding="utf-8"))
    saved_record.update({
        "saved_at": record["saved_at"], "git_commit": None, "git_dirty": None, "versions": None,
        "response_schema": frozen["schemas"][record["spec"]["response_model"]],
        "migrated": {"from": stem, "originally": record.get("migrated_from"), "prompts_checked": checked, "note": "saved in the legacy shape; rewritten by panel.run.legacy"},
    })
    paths.record.write_text(json.dumps(saved_record, ensure_ascii=False, indent=4), encoding="utf-8")
    return Migrated(stem, resolved.stem, len(results), checked, differs)
