"""
What a dataset declares, and its values table.

A dataset is a folder under `datasets/`. Everything that knows its vocabulary lives there:

    datasets/<name>/
        dataset.json          this declaration
        forms.json            how each value is worded inside a sentence (panel.describe.forms)
        descriptions/*.json   its description templates (panel.describe.template)
        claims/<topic>.json   its answers to a topic, sorted by meaning
        prepared/             what its prepare step writes
            values.csv                one row per subject, the dataset's own values - read by analysis
            prompts/<description>.csv one rendered text per subject - read by a run

What the keys of a declaration mean is on `Dataset`.
"""
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple, Union

import pandas as pd

from panel import paths
from panel.config import check_keys, check_short

CONFIG_NAME = "dataset.json"
FORMS_NAME = "forms.json"
DESCRIPTIONS_DIR = "descriptions"
PREPARED_DIR = "prepared"
REQUIRED_KEYS = {"name", "short", "title", "language", "source", "columns", "claims"}
OPTIONAL_KEYS = {"optional", "groups", "area", "levels", "shared", "together", "note"}
LEVEL_KEYS = {"bands", "order", "several"}
CLAIM_KEYS = {"column", "map"}
SHARED_KEYS = {"column", "map"}
MAP_KEYS = {"options", "abstained", "excluded"}
MAP_OPTIONAL_KEYS = {"note"}
INDEX = "id"


@dataclass(frozen=True)
class ClaimRef:
    """Where a dataset keeps its answer to one topic: the values column, and the map file relative to its folder."""
    column: str
    map: str


@dataclass(frozen=True)
class SharedRef:
    """
    One column read in a vocabulary shared beyond the dataset: the values column, and its
    answers -> the shared keys. What the keys mean is between the dataset and whoever asks
    for them by name, e.g. a question worded `by` gender.
    """
    column: str
    map: Dict[str, str]


@dataclass(frozen=True)
class Levels:
    """
    How analysis reads one grouping column as levels; exactly one of the three is set.

    - bands: label -> the highest value it holds, for a numeric column, lowest band first.
    - order: the column's values along their scale, low to high, non-answers last.
    - several: the separator of a cell that holds several answers; a subject then belongs to each.
    """
    bands: Dict[str, float]
    order: Tuple[str, ...]
    several: Union[str, None]


@dataclass(frozen=True)
class Dataset:
    """
    - name: the folder under `datasets/`, and what a run records.
    - language: of its values and descriptions.
    - source: what prepare reads, relative to `dir`; None for prepared data that is never rebuilt;
      `dataset.json` for a grid, built from its levels (`panel.grid`).
    - columns: every column of the values table, in order.
    - optional: columns a subject may lack; the rest must be filled.
    - groups: columns analysis may group subjects by.
    - area: the column whose values name a row of a topic's reference results, or None.
    - levels: column -> how analysis reads it; a column with no entry is its values, largest group first.
    - claims: topic name -> where the dataset's answer to it lives.
    - shared: shared vocabulary name -> the column carrying it and its map.
    - together: name -> columns that share credit as one factor in an attribution (kernel's Shapley effects).
    - note: not used by code.
    """
    name: str
    short: str
    title: str
    language: str
    source: Union[str, None]
    columns: Tuple[str, ...]
    optional: Tuple[str, ...]
    groups: Tuple[str, ...]
    area: Union[str, None]
    levels: Dict[str, "Levels"]
    claims: Dict[str, ClaimRef]
    shared: Dict[str, SharedRef]
    dir: Path
    together: Dict[str, Tuple[str, ...]] = field(default_factory=dict)
    note: str = ""

    @property
    def descriptions(self) -> Tuple[str, ...]:
        return tuple(sorted(path.stem for path in (self.dir / DESCRIPTIONS_DIR).glob("*.json")))

    @property
    def forms_path(self) -> Path:
        return self.dir / FORMS_NAME

    @property
    def data_dir(self) -> Path:
        return self.dir / PREPARED_DIR

    @property
    def values_path(self) -> Path:
        return self.data_dir / "values.csv"

    def description_path(self, name: str) -> Path:
        return self.dir / DESCRIPTIONS_DIR / f"{name}.json"

    def prompts_path(self, description: str) -> Path:
        return self.data_dir / "prompts" / f"{description}.csv"

    def built_from(self, description: str) -> Dict[str, Path]:
        """The files a run of one description is built from, by the name its record gives their digests."""
        return {"values": self.values_path, "prompts": self.prompts_path(description), "forms": self.forms_path}


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
    return sorted(path.parent.name for path in Path(datasets_dir).glob(f"*/{CONFIG_NAME}"))


def load_dataset(name: str, datasets_dir: Union[str, Path] = paths.DATASETS_DIR) -> Dataset:
    directory = Path(datasets_dir) / name
    path = directory / CONFIG_NAME
    if not path.exists():
        raise FileNotFoundError(f"no dataset declaration at {path}")
    config = json.loads(path.read_text(encoding="utf-8"))
    check_keys(config, REQUIRED_KEYS, OPTIONAL_KEYS, str(path))
    if config["name"] != name:
        raise ValueError(f"{path} declares name '{config['name']}'")
    short = check_short(config["short"], str(path))

    columns = tuple(config["columns"])
    if not columns or len(set(columns)) != len(columns):
        raise ValueError(f"{path}: 'columns' is empty or repeats a column")
    claims = {}
    for topic, entry in config["claims"].items():
        if set(entry) != CLAIM_KEYS:
            raise ValueError(f"{path}: claims for '{topic}' need exactly the keys {sorted(CLAIM_KEYS)}")
        claims[topic] = ClaimRef(column=entry["column"], map=entry["map"])
    shared = {}
    for vocabulary, entry in config.get("shared", {}).items():
        if set(entry) != SHARED_KEYS:
            raise ValueError(f"{path}: shared '{vocabulary}' needs exactly the keys {sorted(SHARED_KEYS)}")
        shared[vocabulary] = SharedRef(column=entry["column"], map=dict(entry["map"]))
    optional, groups, area = tuple(config.get("optional", ())), tuple(config.get("groups", ())), config.get("area")
    named = set(optional) | set(groups) | ({area} if area else set())
    named |= {ref.column for ref in claims.values()} | {ref.column for ref in shared.values()}
    levels = {}
    for column, entry in config.get("levels", {}).items():
        if len(entry) != 1 or not set(entry) <= LEVEL_KEYS:
            raise ValueError(f"{path}: levels of '{column}' need exactly one of {sorted(LEVEL_KEYS)}")
        levels[column] = Levels(bands=dict(entry.get("bands", {})), order=tuple(entry.get("order", ())), several=entry.get("several"))
    named |= set(levels)
    together = {name: tuple(group) for name, group in config.get("together", {}).items()}
    grouped = [column for group in together.values() for column in group]
    if len(grouped) != len(set(grouped)) or any(len(group) < 2 for group in together.values()):
        raise ValueError(f"{path}: 'together' needs groups of two columns or more, none in two groups")
    named |= set(grouped)
    stray = sorted(named - set(columns))
    if stray:
        raise ValueError(f"{path}: roles name columns the dataset does not have: {stray}")

    return Dataset(
        name=config["name"], short=short, title=config["title"], language=config["language"],
        source=config["source"], columns=columns, optional=optional, groups=groups, area=area, levels=levels,
        claims=claims, shared=shared, dir=directory, note=config.get("note", ""), together=together,
    )


def load_claims(dataset: Dataset, topic: str) -> Claims:
    if topic not in dataset.claims:
        raise KeyError(f"dataset '{dataset.name}' declares no claims for topic '{topic}'")
    ref = dataset.claims[topic]
    path = dataset.dir / ref.map
    config = json.loads(path.read_text(encoding="utf-8"))
    check_keys(config, MAP_KEYS, MAP_OPTIONAL_KEYS, str(path))
    answers = list(config["options"]) + list(config["abstained"]) + list(config["excluded"])
    if len(set(answers)) != len(answers):
        raise ValueError(f"{path}: an answer is sorted into more than one group")
    return Claims(
        dataset=dataset.name, topic=topic, column=ref.column, options=dict(config["options"]),
        abstained=tuple(config["abstained"]), excluded=tuple(config["excluded"]),
    )


def shared_keys(dataset: Dataset, vocabulary: str, values: pd.DataFrame) -> pd.Series:
    """Each subject's key in a shared vocabulary; a subject without an answer has none."""
    if vocabulary not in dataset.shared:
        raise KeyError(f"dataset '{dataset.name}' declares no shared '{vocabulary}'")
    ref = dataset.shared[vocabulary]
    answers = values[ref.column]
    unmapped = sorted(set(answers.dropna()) - set(ref.map))
    if unmapped:
        raise ValueError(
            f"'{dataset.name}' shared '{vocabulary}': answers of '{ref.column}' not in the map: {unmapped}"
        )
    return answers.map(ref.map)


def validate_values(dataset: Dataset, frame: pd.DataFrame) -> None:
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
    validate_values(dataset, frame)
    dataset.values_path.parent.mkdir(parents=True, exist_ok=True)
    frame[list(dataset.columns)].to_csv(dataset.values_path, index=True, encoding="utf-8")
    return dataset.values_path


def read_values(dataset: Dataset) -> pd.DataFrame:
    if not dataset.values_path.exists():
        raise FileNotFoundError(f"no values table at {dataset.values_path}; run the dataset's prepare step first")
    frame = pd.read_csv(dataset.values_path, index_col=INDEX, encoding="utf-8")
    validate_values(dataset, frame)
    return frame
