"""
The shared profile schema: concepts, the columns that represent them, and their common scales.

`datasets/schema.json` is the one vocabulary every profile table is checked against, so a
column means the same thing in every dataset that carries it: `age` is always an exact age,
`age_range` always a band. A dataset emits whichever representations it has; a template
phrases whichever it knows. For grouping across datasets a concept may declare `levels`, a
common scale, and each of its columns how it maps onto them: `bins` for a number, `map` for a
band or phrase (`panel.analysis.groups.to_levels`).

File shape:

    {
        "note": "...",
        "concepts": {
            "age": {
                "levels": ["18-29", "30-44", "45-59", "60+"],
                "columns": {
                    "age": {"description": "exact age in years", "bins": [30, 45, 60]},
                    "age_range": {"description": "a band, as written", "map": {"18-29": "18-29", ...}}
                }
            },
            "gender": {"columns": {"gender": "the dataset's own label"}},
            ...
        }
    }

A column entry is a description string, or an object with `description` and optionally one of
`bins` (the lower edges of every level but the first) or `map` (value -> level).
"""
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple, Union

from panel import paths

SCHEMA_PATH = paths.DATASETS_DIR / "schema.json"
FILE_KEYS = {"concepts"}
OPTIONAL_FILE_KEYS = {"note"}
CONCEPT_KEYS = {"columns"}
OPTIONAL_CONCEPT_KEYS = {"levels"}
COLUMN_KEYS = {"description"}
OPTIONAL_COLUMN_KEYS = {"bins", "map"}


@dataclass(frozen=True)
class Column:
    """One representation of a concept, and how it maps onto the concept's levels, if it does."""
    name: str
    description: str
    bins: Union[Tuple[float, ...], None] = None
    map: Union[Dict[str, str], None] = None


@dataclass(frozen=True)
class Concept:
    """An attribute, its representation columns in file order, and its common scale if declared."""
    name: str
    columns: Tuple[Column, ...]
    levels: Union[Tuple[str, ...], None] = None


@dataclass(frozen=True)
class Schema:
    """
    Attributes:
        concepts: concept name -> the concept, in file order.
    """
    concepts: Dict[str, Concept]

    @property
    def columns(self) -> Dict[str, str]:
        """Column -> concept name, over every concept."""
        return {column.name: concept.name for concept in self.concepts.values() for column in concept.columns}

    @property
    def descriptions(self) -> Dict[str, str]:
        """Column -> what it holds."""
        return {column.name: column.description for concept in self.concepts.values() for column in concept.columns}


def _column(name: str, entry, levels: Union[Tuple[str, ...], None], concept: str, path: Path) -> Column:
    if isinstance(entry, str):
        return Column(name=name, description=entry)
    unknown = sorted(set(entry) - COLUMN_KEYS - OPTIONAL_COLUMN_KEYS)
    missing = sorted(COLUMN_KEYS - set(entry))
    if unknown or missing or ("bins" in entry and "map" in entry):
        raise ValueError(f"{path}: column '{name}' has unknown keys {unknown}, missing keys {missing}, or both bins and map")
    bins, mapping = entry.get("bins"), entry.get("map")
    if (bins is not None or mapping is not None) and not levels:
        raise ValueError(f"{path}: column '{name}' maps onto levels but concept '{concept}' declares none")
    if bins is not None and len(bins) != len(levels) - 1:
        raise ValueError(f"{path}: column '{name}' needs {len(levels) - 1} bin edges for {len(levels)} levels")
    if mapping is not None and (stray := sorted(set(mapping.values()) - set(levels))):
        raise ValueError(f"{path}: column '{name}' maps onto levels the concept lacks: {stray}")
    return Column(
        name=name, description=entry["description"],
        bins=tuple(float(edge) for edge in bins) if bins is not None else None,
        map=dict(mapping) if mapping is not None else None,
    )


def load_schema(path: Union[str, Path] = SCHEMA_PATH) -> Schema:
    """
    Read the profile schema.

    Raises:
        FileNotFoundError: if there is no schema file.
        ValueError: on unknown keys, a concept without columns, a column claimed twice, or a
            level mapping that does not fit its concept's levels.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"no profile schema at {path}")
    config = json.loads(path.read_text(encoding="utf-8"))
    unknown = sorted(set(config) - FILE_KEYS - OPTIONAL_FILE_KEYS)
    missing = sorted(FILE_KEYS - set(config))
    if unknown or missing:
        raise ValueError(f"{path}: unknown keys {unknown}, missing keys {missing}")

    concepts: Dict[str, Concept] = {}
    seen: Dict[str, str] = {}
    for name, entry in config["concepts"].items():
        stray = sorted(set(entry) - CONCEPT_KEYS - OPTIONAL_CONCEPT_KEYS)
        if stray or not entry.get("columns"):
            raise ValueError(f"{path}: concept '{name}' has unknown keys {stray} or no columns")
        levels = tuple(entry["levels"]) if entry.get("levels") else None
        columns = []
        for column_name, column_entry in entry["columns"].items():
            if column_name in seen:
                raise ValueError(f"{path}: column '{column_name}' is declared by both '{seen[column_name]}' and '{name}'")
            seen[column_name] = name
            columns.append(_column(column_name, column_entry, levels, name, path))
        concepts[name] = Concept(name=name, columns=tuple(columns), levels=levels)
    return Schema(concepts=concepts)
