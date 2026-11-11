"""
A saved run read back, with everything an analysis sets beside it.

The record names the run's topic and dataset, so a stem is enough. Every analysis takes a
`SavedRun` and reads the same four things from it:

- `results` / `normalized` - one row per call, indexed by subject id, the topic's columns;
- `subjects` - the dataset's values for exactly the subjects the run asked, one row each;
- `claims` - how the dataset's own answer to the topic sorts into the topic's options;
- `reference` - the topic's real-world results, one row per area.

`subjects` is the run's subjects and not the whole values table on purpose: a run over a
sample is compared with what that sample claimed. `values` holds the whole table.

Legacy runs (keyed by row position, built from another table) are not read here.
"""
import hashlib
from dataclasses import dataclass
from typing import Any, Dict, List, Union

import pandas as pd

from panel.dataset import Claims, Dataset, load_claims, load_dataset, read_values
from panel.run import ResultStore, RunSpec
from panel.run.resolve import REPEAT
from panel.topic import PlotStyle, Positions, Topic, load_plot_style, load_positions, load_topic


class NotApplicable(Exception):
    """An analysis has nothing to work on for this run, e.g. no reference to measure against."""


@dataclass(frozen=True)
class SavedRun:
    """
    Attributes:
        stem: the run's name in its topic's results folder.
        spec: what the run was declared as.
        record: the whole run record.
        topic: the topic it answered.
        dataset: the dataset its subjects came from.
        results: one row per call as saved, indexed by subject id: `repeat` if the run
            repeats, then the topic's columns.
        normalized: the same rows with each column group rescaled to sum to 1.
        subjects: the dataset's values for the run's subjects, one row per subject, in the order of `results`.
        values: the dataset's whole values table.
        claims: the dataset's answers to the topic sorted by meaning, or None if it has none.
        reference: the topic's real-world results indexed by area, or None if it has none.
        style: the topic's colours and short labels for figures; empty when it has none or
            none was asked for.
        positions: where experts place the topic's options, or None if it has none.
    """
    stem: str
    spec: RunSpec
    record: Dict[str, Any]
    topic: Topic
    dataset: Dataset
    results: pd.DataFrame
    normalized: pd.DataFrame
    subjects: pd.DataFrame
    values: pd.DataFrame
    claims: Union[Claims, None]
    reference: Union[pd.DataFrame, None]
    style: PlotStyle
    positions: Union[Positions, None] = None

    @property
    def options(self) -> List[str]:
        return list(self.topic.options)

    @property
    def labels(self) -> Dict[str, str]:
        """Topic column -> the name a figure shows: the style's label, else the column itself, an option's id being its short name."""
        return {**{column: column for column in self.topic.columns}, **self.style.labels}

    @property
    def total(self) -> Union[str, None]:
        """The reference row that is the whole, or None without a reference."""
        return self.topic.reference.total if self.topic.reference else None


def find_stem(store: ResultStore, text: str) -> str:
    """
    The one saved run whose stem is `text` or contains it.

    Raises:
        ValueError: if no stem or more than one contains it.
    """
    stems = store.runs()
    if text in stems:
        return text
    found = [stem for stem in stems if text in stem]
    if len(found) != 1:
        raise ValueError(f"'{text}' names {len(found)} runs of '{store.topic}': {found or stems}")
    return found[0]


def load_run(store: ResultStore, stem: str, styled: bool = True) -> SavedRun:
    """
    Read one saved run and what it is compared against.

    Args:
        store: the topic's results.
        stem: the run.
        styled: draw with the topic's `plot.json`; off, figures use their own palette and the
            native names.

    Raises:
        FileNotFoundError: if the run, its normalized table or the dataset's values are missing.
        ValueError: on a legacy run, values rebuilt since the run was sent, saved rows that
            are not the subjects the record lists, or a subject's area missing from the reference.
    """
    record = store.record(stem)
    if "description" not in record["spec"]:
        raise ValueError(f"'{stem}' is a legacy run; it is keyed by row position and is not read here")
    spec = RunSpec.from_dict(record["spec"])
    topic, dataset = load_topic(spec.topic), load_dataset(spec.dataset)

    digest = hashlib.sha256(dataset.values_path.read_bytes()).hexdigest()
    if digest != record["dataset"]["values_sha256"]:
        raise ValueError(f"{dataset.values_path} was rebuilt after '{stem}' was sent; its subjects can no longer be trusted to match")
    values = read_values(dataset)

    columns = ([REPEAT] if spec.repeats > 1 else []) + topic.columns
    results, normalized = topic.rename(store.frame(stem))[columns], topic.rename(store.frame(stem, normalized=True))[columns]
    rows = record["dataset"]["rows"]
    if set(results.index) != set(rows) or not results.index.equals(normalized.index):
        raise ValueError(f"'{stem}': the saved rows are not the {len(rows)} subjects its record lists")
    subjects = values.loc[results.index.drop_duplicates()]

    reference = topic.load_reference().set_index(topic.reference.key) if topic.reference else None
    if reference is not None and dataset.area:
        unknown = sorted(set(subjects[dataset.area]) - set(reference.index))
        if unknown:
            raise ValueError(f"'{stem}': areas of '{dataset.area}' with no reference row: {unknown}")

    return SavedRun(
        stem=stem, spec=spec, record=record, topic=topic, dataset=dataset, results=results,
        normalized=normalized, subjects=subjects, values=values,
        claims=load_claims(dataset, topic.name) if topic.name in dataset.claims else None,
        reference=reference, style=load_plot_style(topic) if styled else PlotStyle(), positions=load_positions(topic),
    )


def describe(run: SavedRun) -> str:
    """What was loaded, in a few lines: the run, its size, and what there is to compare it with."""
    spec = run.spec
    lines = [
        run.stem,
        f"  topic      {run.topic.name}: {len(run.options)} options, {'with' if run.topic.turnout else 'no'} turnout",
        f"  model      {spec.client} {spec.model}, {spec.shape}, saved {run.record['saved_at']}",
        f"  subjects   {len(run.subjects)} of {len(run.values)} in '{run.dataset.name}', {len(run.results)} answers",
        f"  claimed    {'column ' + repr(run.claims.column) if run.claims else 'none'}",
        f"  actual     {f'{len(run.reference) - 1} areas and {run.total!r}' if run.reference is not None else 'none'}",
        f"  area       {run.dataset.area or 'none'}",
        f"  groups     {', '.join(run.dataset.groups) or 'none'}",
    ]
    return "\n".join(lines)
