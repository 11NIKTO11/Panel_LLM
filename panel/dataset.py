"""
What a dataset declares, and its values table.

A dataset is a folder under `datasets/`. Everything that knows its vocabulary lives there:

    datasets/<name>/
        dataset.json          this declaration
        forms.json            how each value is worded inside a sentence (panel.describe.forms)
        descriptions/*.json   its description templates (panel.describe.template)
        claims/<topic>.json   its answers to a topic, sorted by meaning

Its prepare step writes, under `data/<name>/`:

    values.csv                one row per subject, the dataset's own values - read by analysis
    prompts/<description>.csv one rendered text per subject - read by a run

Declaration shape:

    {
        "name": "sod",                    the folder name, and the dataset's name in a run
        "short": "sod",                   stem part: ASCII, no underscore
        "title": "...",
        "language": "cs",                 the language its values and descriptions are written in
        "source": "SoD-CZ/....sav",       what prepare reads, relative to the folder
        "columns": ["gender", "age"],     every column of the values table, in order
        "optional": ["employment"],       columns a subject may lack; the rest must be filled
        "groups": ["gender", "region"],   columns analysis may group subjects by
        "area": "region",                 the column whose values name a row of reference results
        "claims": {"election_2021": {"column": "voted_party", "map": "claims/election_2021.json"}},
        "note": "free text, ignored by code"
    }
"""
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Union

import pandas as pd

from panel import paths

CONFIG_NAME = "dataset.json"
FORMS_NAME = "forms.json"
DESCRIPTIONS_DIR = "descriptions"
REQUIRED_KEYS = {"name", "short", "title", "language", "source", "columns", "claims"}
OPTIONAL_KEYS = {"optional", "groups", "area", "note"}
CLAIM_KEYS = {"column", "map"}
MAP_KEYS = {"options", "abstained", "excluded"}
MAP_OPTIONAL_KEYS = {"note"}
INDEX = "id"


@dataclass(frozen=True)
class ClaimRef:
    """Where a dataset keeps its answer to one topic: the values column, and the map file."""
    column: str
    map: str


@dataclass(frozen=True)
class Dataset:
    """
    Attributes:
        name: folder under `datasets/` and under `data/`, and what a run records.
        short: the dataset's part of a results stem.
        title: human label.
        language: language code of its values and descriptions.
        source: what prepare reads, relative to `dir`.
        columns: every column of the values table.
        optional: columns a subject may lack.
        groups: columns analysis may group by.
        area: the column naming a row of a topic's reference results, or None.
        claims: topic name -> where the dataset's answer to it lives.
        dir: the dataset's folder.
        note: free text; not used by code.
    """
    name: str
    short: str
    title: str
    language: str
    source: str
    columns: Tuple[str, ...]
    optional: Tuple[str, ...]
    groups: Tuple[str, ...]
    area: Union[str, None]
    claims: Dict[str, ClaimRef]
    dir: Path
    note: str = ""

    @property
    def descriptions(self) -> Tuple[str, ...]:
        """The description templates in the dataset's folder, by name."""
        return tuple(sorted(path.stem for path in (self.dir / DESCRIPTIONS_DIR).glob("*.json")))

    @property
    def forms_path(self) -> Path:
        return self.dir / FORMS_NAME

    @property
    def data_dir(self) -> Path:
        """Where everything prepare writes for this dataset lives."""
        return paths.DATA_DIR / self.name

    @property
    def values_path(self) -> Path:
        return self.data_dir / "values.csv"

    def description_path(self, name: str) -> Path:
        return self.dir / DESCRIPTIONS_DIR / f"{name}.json"

    def prompts_path(self, description: str) -> Path:
        return self.data_dir / "prompts" / f"{description}.csv"


@dataclass(frozen=True)
class Claims:
    """One dataset's answers to one topic, sorted by meaning."""
    dataset: str
    topic: str
    column: str
    options: Dict[str, str]
    abstained: Tuple[str, ...]
    excluded: Tuple[str, ...]


def dataset_names(datasets_dir: Union[str, Path] = paths.DATASETS_DIR) -> List[str]:
    """Every folder under `datasets_dir` that declares a dataset, sorted."""
    return sorted(path.parent.name for path in Path(datasets_dir).glob(f"*/{CONFIG_NAME}"))


def load_dataset(name: str, datasets_dir: Union[str, Path] = paths.DATASETS_DIR) -> Dataset:
    """
    Read one dataset declaration.

    Raises:
        FileNotFoundError: if there is no declaration of that name.
        ValueError: on unknown or missing keys, a name that disagrees with the folder, a short
            that could not sit in a stem, or a role naming a column the dataset does not have.
    """
    directory = Path(datasets_dir) / name
    path = directory / CONFIG_NAME
    if not path.exists():
        raise FileNotFoundError(f"no dataset declaration at {path}")
    config = json.loads(path.read_text(encoding="utf-8"))
    unknown = sorted(set(config) - REQUIRED_KEYS - OPTIONAL_KEYS)
    missing = sorted(REQUIRED_KEYS - set(config))
    if unknown or missing:
        raise ValueError(f"{path}: unknown keys {unknown}, missing keys {missing}")
    if config["name"] != name:
        raise ValueError(f"{path} declares name '{config['name']}'")
    short = config["short"]
    if not short or not short.isascii() or "_" in short:
        raise ValueError(f"{path}: short '{short}' must be non-empty ASCII without underscores")

    columns = tuple(config["columns"])
    if not columns or len(set(columns)) != len(columns):
        raise ValueError(f"{path}: 'columns' is empty or repeats a column")
    claims = {}
    for topic, entry in config["claims"].items():
        if set(entry) != CLAIM_KEYS:
            raise ValueError(f"{path}: claims for '{topic}' need exactly the keys {sorted(CLAIM_KEYS)}")
        claims[topic] = ClaimRef(column=entry["column"], map=entry["map"])
    optional, groups, area = tuple(config.get("optional", ())), tuple(config.get("groups", ())), config.get("area")
    named = set(optional) | set(groups) | {ref.column for ref in claims.values()} | ({area} if area else set())
    stray = sorted(named - set(columns))
    if stray:
        raise ValueError(f"{path}: roles name columns the dataset does not have: {stray}")

    return Dataset(
        name=config["name"], short=short, title=config["title"], language=config["language"],
        source=config["source"], columns=columns, optional=optional, groups=groups, area=area,
        claims=claims, dir=directory, note=config.get("note", ""),
    )


def load_claims(dataset: Dataset, topic: str) -> Claims:
    """
    The dataset's claims map for one topic.

    Raises:
        KeyError: if the dataset declares no answer for that topic.
        ValueError: on unknown or missing keys, or an answer sorted into two groups.
    """
    if topic not in dataset.claims:
        raise KeyError(f"dataset '{dataset.name}' declares no claims for topic '{topic}'")
    ref = dataset.claims[topic]
    path = dataset.dir / ref.map
    config = json.loads(path.read_text(encoding="utf-8"))
    unknown = sorted(set(config) - MAP_KEYS - MAP_OPTIONAL_KEYS)
    missing = sorted(MAP_KEYS - set(config))
    if unknown or missing:
        raise ValueError(f"{path}: unknown keys {unknown}, missing keys {missing}")
    answers = list(config["options"]) + list(config["abstained"]) + list(config["excluded"])
    if len(set(answers)) != len(answers):
        raise ValueError(f"{path}: an answer is sorted into more than one group")
    return Claims(
        dataset=dataset.name, topic=topic, column=ref.column, options=dict(config["options"]),
        abstained=tuple(config["abstained"]), excluded=tuple(config["excluded"]),
    )


def validate_values(dataset: Dataset, frame: pd.DataFrame) -> None:
    """
    Refuse a values table that is not what the dataset declares.

    Raises:
        ValueError: if the index is not a unique `id`, the columns are not exactly the declared
            ones, or a column that is not optional has gaps.
    """
    problems = []
    if frame.index.name != INDEX:
        problems.append(f"index is named {frame.index.name!r}, not '{INDEX}'")
    if not frame.index.is_unique or frame.index.hasnans:
        problems.append("index has duplicates or gaps")
    missing = [column for column in dataset.columns if column not in frame.columns]
    undeclared = [column for column in frame.columns if column not in dataset.columns]
    if missing or undeclared:
        problems.append(f"columns missing: {missing}, undeclared: {undeclared}")
    gaps = {column: int(frame[column].isna().sum())
            for column in dataset.columns if column in frame.columns and column not in dataset.optional}
    gaps = {column: count for column, count in gaps.items() if count}
    if gaps:
        problems.append(f"columns with missing values: {gaps}")
    if problems:
        raise ValueError(f"values of '{dataset.name}': " + "; ".join(problems))


def write_values(dataset: Dataset, frame: pd.DataFrame) -> Path:
    """
    Validate and store a dataset's values table at `data/<name>/values.csv`.

    Raises:
        ValueError: see `validate_values`.
    """
    validate_values(dataset, frame)
    dataset.values_path.parent.mkdir(parents=True, exist_ok=True)
    frame[list(dataset.columns)].to_csv(dataset.values_path, index=True, encoding="utf-8")
    return dataset.values_path


def read_values(dataset: Dataset) -> pd.DataFrame:
    """
    A dataset's stored values table, validated on the way in.

    Raises:
        FileNotFoundError: if the dataset has not been prepared.
        ValueError: see `validate_values`.
    """
    if not dataset.values_path.exists():
        raise FileNotFoundError(f"no values table at {dataset.values_path}; run the dataset's prepare step first")
    frame = pd.read_csv(dataset.values_path, index_col=INDEX, encoding="utf-8")
    validate_values(dataset, frame)
    return frame
