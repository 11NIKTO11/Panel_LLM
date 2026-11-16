"""
Several saved runs of one topic, read back to be set beside each other.

Each run gets a short label for tables and figures: its model, and when two runs share a
model, the spec fields they differ in. Runs are named one by one - a topic's folder also
holds trial and offline runs, and a comparison should say what it compares.
"""
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Mapping, Sequence, Union

import pandas as pd

from panel import paths
from panel.run import ResultStore
from panel.topic import PlotStyle, Topic, load_polls

from ..output import Output
from ..saved import SavedRun, find_stem, load_run

CLAIMED = "claimed"


@dataclass(frozen=True)
class RunSet:
    """
    Attributes:
        runs: label -> run, in the order they were named.
        polls: the topic's published polls, one row each, a column per option; may be empty.
    """
    topic: Topic
    runs: Dict[str, SavedRun]
    polls: pd.DataFrame
    style: PlotStyle

    @property
    def claimed(self) -> Dict[str, SavedRun]:
        """
        Label -> a run whose subjects' claims stand for it: one entry, `claimed`, when every
        run asked the same subjects, else `claimed (<run>)` per distinct set of subjects.
        Empty when the dataset has no claims for the topic.
        """
        distinct = {}
        for label, run in self.runs.items():
            if run.claims:
                distinct.setdefault(frozenset(run.subjects.index), (label, run))
        if len(distinct) == 1:
            return {CLAIMED: next(iter(distinct.values()))[1]}
        return {f"{CLAIMED} ({label})": run for label, run in distinct.values()}

    @property
    def stands(self) -> Dict[str, str]:
        """Each label of `claimed` -> the label of the run whose subjects' claims stand for it."""
        return {label: next(name for name, other in self.runs.items() if other is run) for label, run in self.claimed.items()}

    @property
    def labels(self) -> Dict[str, str]:
        """Topic column -> the name a figure shows, as the runs' own figures name it."""
        return next(iter(self.runs.values())).labels


def labels(runs: Sequence[SavedRun]) -> Dict[str, SavedRun]:
    specs = [run.spec.to_dict() for run in runs]
    differing = [field for field in specs[0] if field != "model" and len({str(spec[field]) for spec in specs}) > 1]
    models = [spec["model"] for spec in specs]
    return {
        spec["model"] if models.count(spec["model"]) == 1 else f"{spec['model']} ({', '.join(f'{field}={spec[field]}' for field in differing)})": run
        for spec, run in zip(specs, runs)
    }


def load_runs(
    store: ResultStore, texts: Union[Sequence[str], Mapping[str, str]], styled: bool = True, weighted: bool = True,
    datasets_dir: Union[str, Path] = paths.DATASETS_DIR, topics_dir: Union[str, Path] = paths.TOPICS_DIR,
) -> RunSet:
    """
    `texts` is per run its stem or any text only one stem contains; or label -> such a text,
    and the runs are called by those labels instead of by their models.
    """
    given = list(texts) if isinstance(texts, Mapping) else None
    stems = [find_stem(store, text) for text in (texts.values() if given else texts)]
    if not stems or len(set(stems)) != len(stems):
        raise ValueError(f"name each run to compare once; got {stems}")
    runs = [load_run(store, stem, styled, weighted, datasets_dir, topics_dir) for stem in stems]
    return RunSet(topic=runs[0].topic, runs=dict(zip(given, runs)) if given else labels(runs), polls=load_polls(runs[0].topic), style=runs[0].style)


def comparison_output(store: ResultStore, name: str, runs: RunSet, save: bool = True, suffix: str = "") -> Output:
    """
    Where a named comparison of `runs` leaves what it makes: `results/<topic>/comparisons/<name>/`.
    When saving, the folder also says which runs it was made from (`runs.json`, label -> stem).
    """
    if save:
        store.save_comparison(name, {label: run.stem for label, run in runs.runs.items()})
    return Output(store.comparison_dir(name), save=save, suffix=suffix)


def describe(runs: RunSet) -> str:
    width = max(len(label) for label in runs.runs)
    lines = [f"  {label:<{width}}  {len(run.subjects)} subjects, {len(run.results)} answers  {run.stem}" for label, run in runs.runs.items()]
    return "\n".join([f"{runs.topic.name}: {len(runs.runs)} runs", *lines, f"  claimed: {', '.join(runs.claimed) or 'none'}", f"  polls: {', '.join(runs.polls.index) or 'none'}"])
