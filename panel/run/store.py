import json
import subprocess
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Sequence, Union

import pandas as pd

from panel.analysis.aggregate import normalize_groups
from .io_results import results_to_dataframe, save_results_to_json
from .spec import RunSpec


@dataclass(frozen=True)
class RunPaths:
    stem: str
    results_json: Path
    results_csv: Path
    normalized_csv: Path
    record: Path


def _git_commit() -> Union[str, None]:
    """Current HEAD, or None when git is unavailable."""
    try:
        done = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, timeout=5)
        return done.stdout.strip() or None
    except Exception:
        return None


class ResultStore:
    """
    Owns every path under one topic's results folder. Nothing else builds a results path.

    A saved run is four files sharing one stem: the raw results as JSON, the same as a
    wide CSV, a row-normalized CSV, and a run record holding the RunSpec that produced
    it. The stem identifies a run at a glance; the record carries the parts a filename
    cannot, and having one is what makes a run a run - `runs()` lists records.

        results/election_2021/
        |-- <stem>.json                 raw results
        |-- <stem>.csv                  the same, wide
        |-- <stem>_normalized.csv       column groups rescaled to sum to 1 per row
        |-- <stem>_run.json             the run record
        |-- <name>_claimed_votes.csv    topic-level reference series
        |-- analysis/                   tables derived from runs: <statistic>_<stem>.csv
        |-- archive/                    <stem>_<timestamp>.csv + _run.json, one per save
        `-- preferences/                pre-RunSpec tables that map to no single run

    One folder per topic: runs on different topics answer with different vocabularies
    and cannot be compared or concatenated. The dataset stays in the stem rather than
    becoming a folder - two datasets on one topic are meant to be compared.
    """

    RECORD_SUFFIX = "_run.json"
    NORMALIZED_SUFFIX = "_normalized"
    ANALYSIS_DIR = "analysis"
    ARCHIVE_DIR = "archive"

    def __init__(self, results_dir: Union[str, Path], topic: str):
        self.results_dir = Path(results_dir)
        self.topic = topic

    @property
    def topic_dir(self) -> Path:
        return self.results_dir / self.topic

    def path(self, name: str) -> Path:
        """A path in this topic's folder, for files belonging to no single run."""
        return self.topic_dir / name

    def analysis(self, name: str) -> Path:
        """A path in `analysis/`, for tables derived from runs. Creates the folder."""
        directory = self.topic_dir / self.ANALYSIS_DIR
        directory.mkdir(parents=True, exist_ok=True)
        return directory / name

    def paths(self, stem: str, timestamp: Union[str, None] = None) -> RunPaths:
        """The four files of one run, or with a timestamp their archived copy."""
        directory = self.topic_dir / self.ARCHIVE_DIR if timestamp else self.topic_dir
        base = f"{stem}_{timestamp}" if timestamp else stem
        return RunPaths(
            stem=stem,
            results_json=directory / f"{base}.json",
            results_csv=directory / f"{base}.csv",
            normalized_csv=directory / f"{base}{self.NORMALIZED_SUFFIX}.csv",
            record=directory / f"{base}{self.RECORD_SUFFIX}",
        )

    def runs(self) -> List[str]:
        """Stems of this topic's saved runs: every stem with a run record."""
        return sorted(
            path.name[: -len(self.RECORD_SUFFIX)]
            for path in self.topic_dir.glob(f"*{self.RECORD_SUFFIX}")
        )

    def save(
        self,
        spec: RunSpec,
        results: Dict[int, Any],
        normalize: Union[Sequence[Sequence[str]], None] = None,
        timestamped: bool = True,
    ) -> RunPaths:
        """
        Write one result set: the results as JSON, as a wide CSV, optionally a
        normalized CSV, and a record holding the spec that produced them.

        Called twice per run - once as it defaults, once with `timestamped=False`. The
        timestamped set lands in `archive/`: unique per save, and what keeps an
        overwritten run recoverable. The plain set is the handle you load by stem, and
        is overwritten freely. Saving only the plain set makes an overwrite
        unrecoverable, which is why the archive is the default.

        An archived save writes the CSV and the record only. The raw JSON and the
        normalized CSV are reconstructions of that CSV, so archiving them would double
        the folder for nothing - which means `normalize` does nothing when `timestamped`.

        Args:
            spec: the run that produced `results`.
            results: {row id: (prompt, parsed)} from `run_llm_batch`.
            normalize: column groups each rescaled to sum to 1 per row, written as the
                normalized CSV. Omit to skip that file; ignored for an archived save.
            timestamped: write the archived copy instead of the plain one.

        Returns:
            The paths written.

        Raises:
            ValueError: if the spec's topic is not this store's topic.
        """
        if spec.topic != self.topic:
            raise ValueError(f"spec topic '{spec.topic}' does not match store topic '{self.topic}'")

        now = datetime.now()
        paths = self.paths(spec.stem, now.strftime("%Y-%m-%d-%H-%M-%S") if timestamped else None)
        paths.results_csv.parent.mkdir(parents=True, exist_ok=True)

        frame = results_to_dataframe(results).fillna(0)
        record = {
            "stem": spec.stem,
            "n": len(results),
            "saved_at": now.isoformat(timespec="seconds"),
            "git_commit": _git_commit(),
            "spec": spec.to_dict(),
        }

        frame.to_csv(paths.results_csv, index=True)
        paths.record.write_text(json.dumps(record, ensure_ascii=False, indent=4), encoding="utf-8")
        if not timestamped:
            save_results_to_json(results, str(paths.results_json))
            if normalize:
                normalize_groups(frame, normalize).to_csv(paths.normalized_csv, index=True)

        return paths

    def frame(self, stem: str, normalized: bool = False) -> pd.DataFrame:
        """
        Raises:
            FileNotFoundError: if that run has no such CSV.
        """
        paths = self.paths(stem)
        path = paths.normalized_csv if normalized else paths.results_csv
        if not path.exists():
            raise FileNotFoundError(f"{path} not found; saved runs: {self.runs()}")
        return pd.read_csv(path, index_col=0)

    def record(self, stem: str) -> Dict[str, Any]:
        """
        Raises:
            FileNotFoundError: if the run has no record file.
        """
        path = self.paths(stem).record
        if not path.exists():
            raise FileNotFoundError(
                f"no run record for '{stem}' at {path}; migrate it with the cell in update_script.ipynb"
            )
        return json.loads(path.read_text(encoding="utf-8"))
