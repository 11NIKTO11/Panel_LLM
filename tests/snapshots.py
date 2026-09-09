"""
How each phase 0 baseline is computed, so the builder and the tests share one definition.

Every function takes loaded objects and returns something JSON-serialisable or a digest;
nothing here reads the repo. `baseline.py` writes what these return to `tests/fixtures/`,
and the tests recompute them and compare.
"""
import hashlib
from typing import Dict, Sequence, Tuple, Type

import pandas as pd

from panel.analysis.aggregate import aggregate_results_from_df
from panel.topic import make_response_models

COMBINATIONS: Tuple[Tuple[bool, bool], ...] = tuple(
    (weighted, normalized) for weighted in (True, False) for normalized in (True, False)
)


def combination_key(weighted: bool, normalized: bool) -> str:
    return f"weighted={weighted} normalized={normalized}"


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def prompt_table_digest(descriptions: pd.Series) -> Dict:
    """Row count and a digest over `id<TAB>text` lines, independent of CSV formatting."""
    lines = "\n".join(f"{index}\t{text}" for index, text in descriptions.items())
    return {"rows": int(len(descriptions)), "sha256": _sha256(lines)}


def schema_snapshot(probabilities: Type, pick: Type) -> Dict:
    """Both response models' JSON schemas, keyed by class name."""
    return {model.__name__: model.model_json_schema() for model in (probabilities, pick)}


def models_by_name(topic) -> Dict[str, Type]:
    """A topic's response models keyed by class name, as a run record names them."""
    return {model.__name__: model for model in make_response_models(topic)}


def aggregate_all(frame: pd.DataFrame, topic) -> Dict[str, Dict[str, float]]:
    """Country-level aggregation of one results frame for all four weighted x normalized combinations."""
    return {
        combination_key(weighted, normalized): {
            column: float(value)
            for column, value in aggregate_results_from_df(frame, topic, weighted, normalized).items()
        }
        for weighted, normalized in COMBINATIONS
    }


def json_shape(raw: Dict) -> str:
    """'wrapped' when items are {prompt, result} pairs, 'bare' when each item is the result itself."""
    first = next(iter(raw.values()))
    return "wrapped" if isinstance(first, dict) and set(first) == {"prompt", "result"} else "bare"


def reparse(raw: Dict, model: Type, columns: Sequence[str]) -> pd.DataFrame:
    """
    Every saved response parsed through `model`, as a frame sorted by id in `columns` order.

    Raises:
        pydantic.ValidationError: if a saved response no longer fits the model.
        KeyError: if the model's rows lack one of `columns`.
    """
    wrapped = json_shape(raw) == "wrapped"
    rows = {
        int(key): model(**(item["result"] if wrapped else item)).to_series()
        for key, item in raw.items()
    }
    return pd.DataFrame(rows).T.sort_index().loc[:, list(columns)]


def frame_digest(frame: pd.DataFrame) -> str:
    """A digest over `id:v1,v2,...` lines with full float repr, sorted by id."""
    ordered = frame.sort_index()
    lines = "\n".join(
        f"{index}:" + ",".join(repr(float(value)) for value in row)
        for index, row in zip(ordered.index, ordered.to_numpy())
    )
    return _sha256(lines)
