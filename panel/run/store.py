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
    |-- comparisons/<name>/
    |   |-- runs.json               label -> stem: the runs this comparison was made from
    |   |-- analysis/               <statistic>.csv
    |   `-- plots/                  <name>.png
    |-- preferences/                pre-RunSpec tables that map to no run, with their figures
    `-- <name>_claimed_votes.csv    topic-level reference series (legacy)

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

from panel.llm import BaseLLMClient
from panel.topic.normalize import normalize_groups

from .record import build_record
from .resolve import REPEAT, SUBJECT, Resolved

PROMPT = "Prompt"
DIGEST_SUFFIX = "_sha256"


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
        dataset = json.loads(path.read_text(encoding="utf-8")).get("dataset", {})
        for key, digest in dataset.items():
            if key.endswith(DIGEST_SUFFIX):
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
        """One run's folder: its four files, its archive, and what was derived from it."""
        return self.topic_dir / self.RUNS_DIR / stem

    def comparison_dir(self, name: str) -> Path:
        """One named comparison's folder: what was derived from several runs, and which."""
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
        """Stems of this topic's saved runs: every run folder with a run record."""
        return sorted(path.parent.name for path in (self.topic_dir / self.RUNS_DIR).glob(f"*/{self.RECORD_NAME}"))

    def save_comparison(self, name: str, runs: Mapping[str, str]) -> Path:
        """Write which runs a named comparison was made from, `label -> stem`, as its `runs.json`."""
        path = self.comparison_dir(name) / self.MEMBERS_NAME
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(dict(runs), ensure_ascii=False, indent=4), encoding="utf-8")
        return path

    def record(self, stem: str) -> Dict[str, Any]:
        """
        Raises:
            FileNotFoundError: if the run has no record file.
        """
        path = self.paths(stem).record
        if not path.exists():
            raise FileNotFoundError(f"no run record for '{stem}' at {path}; saved runs: {self.runs()}")
        return json.loads(path.read_text(encoding="utf-8"))

    def frame(self, stem: str, normalized: bool = False) -> pd.DataFrame:
        """
        A run's CSV as stored: the ids of its day for a run saved here, native names for a legacy
        one. `Topic.rename` turns either into today's ids.

        Raises:
            FileNotFoundError: if that run has no such CSV.
        """
        paths = self.paths(stem)
        path = paths.normalized_csv if normalized else paths.results_csv
        if not path.exists():
            raise FileNotFoundError(f"{path} not found; saved runs: {self.runs()}")
        return pd.read_csv(path, index_col=0, encoding="utf-8")

    def raw(self, stem: str) -> Dict[str, Any]:
        """
        A run's raw results JSON, as written.

        Raises:
            FileNotFoundError: if the run has no JSON (the legacy VPick runs have none).
        """
        path = self.paths(stem).results_json
        if not path.exists():
            raise FileNotFoundError(f"{path} not found; saved runs: {self.runs()}")
        return json.loads(path.read_text(encoding="utf-8"))

    def load(self, stem: str, response_model: Type[BaseModel]) -> Dict[Tuple[int, int], Tuple[Union[str, None], BaseModel]]:
        """
        Every saved response of a run, parsed again, keyed (subject id, repeat).

        Reads both shapes of raw JSON: items that are `{prompt, result}` pairs, and legacy items
        that are the result itself (then the prompt is None). A legacy key is a subject id; a
        repeated run's key is `<id>:<repeat>`.

        Raises:
            FileNotFoundError: if the run has no JSON.
            pydantic.ValidationError: if a saved response no longer fits the model.
        """
        parsed = {}
        for key, item in self.raw(stem).items():
            subject, _, repeat = key.partition(":")
            wrapped = isinstance(item, dict) and set(item) == {"prompt", "result"}
            prompt, payload = (item["prompt"], item["result"]) if wrapped else (None, item)
            parsed[(int(subject), int(repeat or 1))] = (prompt, response_model.model_validate(payload))
        return parsed

    def save(
        self,
        resolved: Resolved,
        results: Dict[int, Tuple[str, BaseModel]],
        client: Union[BaseLLMClient, None] = None,
        timestamped: bool = True,
        overwrite: bool = False,
    ) -> RunPaths:
        """
        Write one result set under the run's stem.

        Called twice per run - once as it defaults, once with `timestamped=False`. The
        timestamped set lands in the run's `archive/`, unique per save, and is what keeps an overwritten
        run recoverable; it holds the CSV and the record only, because the raw JSON and the
        normalized CSV are reconstructions of the CSV. The plain set is the handle a stem is
        loaded by and is overwritten freely by the same spec.

        Args:
            resolved: the run that produced `results`.
            results: {call: (prompt, parsed)} from `run_llm_batch` over `resolved.calls`.
            client: the client that made the calls, for the record's `model_resolved`.
            timestamped: write the archived copy instead of the plain one.
            overwrite: replace a run of a different spec that already holds the stem.

        Raises:
            ValueError: if the run's topic is not this store's topic.
            FileExistsError: if the stem holds a run of a different spec and `overwrite` is False.
        """
        if resolved.topic.name != self.topic:
            raise ValueError(f"run topic '{resolved.topic.name}' does not match store topic '{self.topic}'")
        stem = resolved.stem
        plain = self.paths(stem)
        if plain.record.exists() and not overwrite:
            existing = json.loads(plain.record.read_text(encoding="utf-8")).get("spec")
            if existing != resolved.spec.to_dict():
                raise FileExistsError(f"'{stem}' already holds a run of a different spec; pass overwrite=True to replace it")

        now = datetime.now()
        paths = self.paths(stem, now.strftime("%Y-%m-%d-%H-%M-%S") if timestamped else None)
        paths.results_csv.parent.mkdir(parents=True, exist_ok=True)
        frame = _results_frame(results, resolved)
        record = build_record(resolved, len(results), now, client)

        frame.to_csv(paths.results_csv, index=True, encoding="utf-8")
        paths.record.write_text(json.dumps(record, ensure_ascii=False, indent=4), encoding="utf-8")
        if not timestamped:
            raw = _raw_results(results, resolved)
            paths.results_json.write_text(json.dumps(raw, ensure_ascii=False, indent=4), encoding="utf-8")
            normalized = normalize_groups(frame, resolved.topic.normalize_groups)
            normalized.to_csv(paths.normalized_csv, index=True, encoding="utf-8")
        return paths


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
