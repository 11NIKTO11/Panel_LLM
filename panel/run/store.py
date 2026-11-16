"""
ResultStore: owner of every path under one topic's results folder. Nothing else builds a results path.

A saved run is a folder named by its stem, holding four files - the raw results as JSON, the
same as a wide CSV, a row-normalized CSV, and the run record - an archived copy of the CSV and
the record per save, and whatever was derived from the run. The stem identifies a run at a
glance; the record carries what a name cannot, and having one is what makes a run a run:
`runs()` lists records. What was derived from several runs sits in a folder per named
comparison, with the runs it was made from.

    results/election_2021/
    |-- runs/<stem>/
    |   |-- results.json            raw results: {"<id>" or "<id>:<repeat>": {"prompt", "result"}}
    |   |-- results.csv             one row per call: id, [repeat], the topic's columns, Prompt
    |   |-- normalized.csv          column groups rescaled to sum to 1 per row
    |   |-- run.json                the run record (panel.run.record)
    |   |-- archive/                <timestamp>.csv + <timestamp>_run.json, one pair per save
    |   |-- analysis/               tables derived from the run: <statistic>.csv
    |   |-- plots/                  figures derived from the run: <name>.png
    |   `-- delivered/              for a run kept only as a table: the table as delivered
    `-- comparisons/<name>/
        |-- runs.json               label -> stem: the runs this comparison was made from
        |-- analysis/               <statistic>.csv
        `-- plots/                  <name>.png

One folder per topic: runs on different topics answer with different vocabularies and cannot
be compared or concatenated. The dataset stays in the stem rather than becoming a folder - two
datasets on one topic are meant to be compared. Files inside a run's folder carry no stem: the
folder does, and a stem twice in one path does not fit a Windows path.
"""
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Mapping, Tuple, Type, Union

import pandas as pd
from pydantic import BaseModel

from panel.digest import recorded
from panel.llm import BaseLLMClient
from panel.topic.normalize import normalize_groups

from .record import KEPT, build_record
from .resolve import REPEAT, SUBJECT, Resolved

PROMPT = "Prompt"


@dataclass(frozen=True)
class RunPaths:
    """The four files of one run, plain or archived."""
    stem: str
    results_json: Path
    results_csv: Path
    normalized_csv: Path
    record: Path


def digests_in_use(results_dir: Union[str, Path]) -> Dict[str, List[str]]:
    """The sha256 of every file a saved run of any topic was built from -> the stems of those runs."""
    found: Dict[str, List[str]] = {}
    for path in sorted(Path(results_dir).glob(f"*/{ResultStore.RUNS_DIR}/*/{ResultStore.RECORD_NAME}")):
        for digest in recorded(json.loads(path.read_text(encoding="utf-8")).get("dataset", {})).values():
            found.setdefault(digest, []).append(path.parent.name)
    return found


class ResultStore:
    """One topic's results folder; see the module docstring for the layout."""

    RUNS_DIR = "runs"
    COMPARISONS_DIR = "comparisons"
    RECORD_NAME = "run.json"
    MEMBERS_NAME = "runs.json"
    ANALYSIS_DIR = "analysis"
    PLOTS_DIR = "plots"
    ARCHIVE_DIR = "archive"
    DELIVERED_DIR = "delivered"

    def __init__(self, results_dir: Union[str, Path], topic: str):
        self.results_dir = Path(results_dir)
        self.topic = topic

    @property
    def topic_dir(self) -> Path:
        return self.results_dir / self.topic

    def run_dir(self, stem: str) -> Path:
        return self.topic_dir / self.RUNS_DIR / stem

    def comparison_dir(self, name: str) -> Path:
        return self.topic_dir / self.COMPARISONS_DIR / name

    def paths(self, stem: str, timestamp: Union[str, None] = None) -> RunPaths:
        """The four files of one run, or with a timestamp their archived copy."""
        directory = self.run_dir(stem)
        if timestamp:
            archive = directory / self.ARCHIVE_DIR
            return RunPaths(
                stem=stem,
                results_json=archive / f"{timestamp}.json",
                results_csv=archive / f"{timestamp}.csv",
                normalized_csv=archive / f"{timestamp}_normalized.csv",
                record=archive / f"{timestamp}_{self.RECORD_NAME}",
            )
        return RunPaths(
            stem=stem,
            results_json=directory / "results.json",
            results_csv=directory / "results.csv",
            normalized_csv=directory / "normalized.csv",
            record=directory / self.RECORD_NAME,
        )

    def runs(self) -> List[str]:
        return sorted(path.parent.name for path in (self.topic_dir / self.RUNS_DIR).glob(f"*/{self.RECORD_NAME}"))

    def save_comparison(self, name: str, runs: Mapping[str, str]) -> Path:
        path = self.comparison_dir(name) / self.MEMBERS_NAME
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(dict(runs), ensure_ascii=False, indent=4), encoding="utf-8")
        return path

    def record(self, stem: str) -> Dict[str, Any]:
        path = self.paths(stem).record
        if not path.exists():
            raise FileNotFoundError(f"no run record for '{stem}' at {path}; saved runs: {self.runs()}")
        return json.loads(path.read_text(encoding="utf-8"))

    def frame(self, stem: str, normalized: bool = False) -> pd.DataFrame:
        """As stored, under the option ids of the day it was saved; `Topic.rename` turns them into today's."""
        paths = self.paths(stem)
        path = paths.normalized_csv if normalized else paths.results_csv
        if not path.exists():
            raise FileNotFoundError(f"{path} not found; saved runs: {self.runs()}")
        return pd.read_csv(path, index_col=0, encoding="utf-8")

    def raw(self, stem: str) -> Dict[str, Any]:
        path = self.paths(stem).results_json
        if not path.exists():
            raise FileNotFoundError(f"{path} not found; saved runs: {self.runs()}")
        return json.loads(path.read_text(encoding="utf-8"))

    def load(self, stem: str, response_model: Type[BaseModel]) -> Dict[Tuple[int, int], Tuple[str, BaseModel]]:
        """
        Keyed (subject id, repeat), each response with its prompt; a saved response that no longer fits the
        model raises `pydantic.ValidationError`.
        """
        parsed = {}
        for key, item in self.raw(stem).items():
            subject, _, repeat = key.partition(":")
            parsed[(int(subject), int(repeat or 1))] = (item["prompt"], response_model.model_validate(item["result"]))
        return parsed

    def refuse(self, resolved: Resolved, overwrite: bool = False) -> None:
        """
        Raise if `resolved` may not be saved here; `execute` asks before any call is made.
        A run brought in rather than sent - its record has a `migrated` or a `delivered` block -
        has no archived copy and is never saved over, whatever `overwrite` says.
        """
        if resolved.topic.name != self.topic:
            raise ValueError(f"run topic '{resolved.topic.name}' does not match store topic '{self.topic}'")
        stem = resolved.stem
        record = self.paths(stem).record
        if not record.exists():
            return
        existing = json.loads(record.read_text(encoding="utf-8"))
        kept = [block for block in KEPT if block in existing]
        if kept:
            raise FileExistsError(f"'{stem}' holds a run that was brought in ({', '.join(kept)}), which is never saved over")
        if not overwrite and existing.get("spec") != resolved.spec.to_dict():
            raise FileExistsError(f"'{stem}' already holds a run of a different spec; pass overwrite=True to replace it")

    def save(
        self,
        resolved: Resolved,
        results: Dict[int, Tuple[str, BaseModel]],
        client: Union[BaseLLMClient, None] = None,
        overwrite: bool = False,
    ) -> RunPaths:
        """
        `results` is {call: (prompt, parsed)} from `run_llm_batch` over `resolved.calls`; returns the plain paths.

        The archived copy is what keeps an overwritten run recoverable; it holds the CSV and the record
        only, because the raw JSON and the normalized CSV are reconstructions of the CSV. The plain set is
        the handle a stem is loaded by and is overwritten freely by the same spec.
        """
        self.refuse(resolved, overwrite)
        stem = resolved.stem
        plain = self.paths(stem)

        now = datetime.now()
        frame = _results_frame(results, resolved)
        record = json.dumps(build_record(resolved, len(results), now, client), ensure_ascii=False, indent=4)
        for paths in (self.paths(stem, now.strftime("%Y-%m-%d-%H-%M-%S")), plain):
            paths.results_csv.parent.mkdir(parents=True, exist_ok=True)
            frame.to_csv(paths.results_csv, index=True, encoding="utf-8")
            paths.record.write_text(record, encoding="utf-8")
        raw = _raw_results(results, resolved)
        plain.results_json.write_text(json.dumps(raw, ensure_ascii=False, indent=4), encoding="utf-8")
        normalize_groups(frame, resolved.topic.normalize_groups).to_csv(plain.normalized_csv, index=True, encoding="utf-8")
        return plain


def _results_frame(results: Dict[int, Tuple[str, BaseModel]], resolved: Resolved) -> pd.DataFrame:
    """One row per call, indexed by subject id: [repeat,] the topic's columns, Prompt."""
    rows = []
    for call, (prompt, parsed) in results.items():
        row = parsed.to_series()
        row[PROMPT] = prompt
        rows.append(row.rename(call))
    calls = resolved.calls[[SUBJECT, REPEAT]]
    frame = calls.join(pd.DataFrame(rows), how="inner").sort_values([SUBJECT, REPEAT]).set_index(SUBJECT)
    return frame.drop(columns=REPEAT) if resolved.spec.repeats == 1 else frame


def _raw_results(results: Dict[int, Tuple[str, BaseModel]], resolved: Resolved) -> Dict[str, Any]:
    """{"<id>" or "<id>:<repeat>": {"prompt", "result"}}, in subject order."""
    calls = resolved.calls
    ordered = sorted(results, key=lambda call: (calls.at[call, SUBJECT], calls.at[call, REPEAT]))
    raw = {}
    for call in ordered:
        subject, repeat = int(calls.at[call, SUBJECT]), int(calls.at[call, REPEAT])
        key = str(subject) if resolved.spec.repeats == 1 else f"{subject}:{repeat}"
        prompt, parsed = results[call]
        raw[key] = {"prompt": prompt, "result": parsed.model_dump()}
    return raw
