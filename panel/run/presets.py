"""
Presets: a base run, and the data and the models worth a name, kept as data in `config/presets.json`.

The base is a complete run: every field of `DATA` and of `MODEL`. A data preset says what is
asked, of whom and in what form, and may state fields of `DATA` only; a model preset says which
model answers and how, and may state fields of `MODEL` only. A run is the base, then a data
preset, then a model preset, then single fields - each replacing what came before. The two
kinds share no field, so their order decides nothing. How much of the data a run takes
(`sample`, `random_state`) is in none of them: a trial is the same run, only smaller. A preset
name is a handle for people; a saved run is identified by its stem and its record, which come
from the spec.
"""
import json
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any, Dict, Union, get_args

from .spec import RunSpec

KEYS = {"base", "data", "models"}
FIELDS = tuple(field.name for field in fields(RunSpec))
DATA = ("dataset", "description", "topic", "framing", "question", "shape", "repeats", "sum_low", "sum_high")
MODEL = ("client", "model", "temperature", "effort")
NUMBERS = {field.name: kind for field in fields(RunSpec) for kind in (int, float) if kind in (field.type, *get_args(field.type))}
NONE = "none"


@dataclass(frozen=True)
class Presets:
    base: Dict[str, Any]
    data: Dict[str, Dict[str, Any]]
    models: Dict[str, Dict[str, Any]]


def load_presets(path: Union[str, Path]) -> Presets:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"no presets at {path}")
    config = json.loads(path.read_text(encoding="utf-8"))
    if set(config) != KEYS:
        raise ValueError(f"{path} needs exactly the keys {sorted(KEYS)}")
    if set(config["base"]) != set(DATA + MODEL):
        raise ValueError(f"{path}: base must state exactly {list(DATA + MODEL)}; it states {sorted(config['base'])}")
    for kind, allowed in (("data", DATA), ("models", MODEL)):
        for name, preset in config[kind].items():
            unknown = sorted(set(preset) - set(allowed))
            if unknown or not preset:
                raise ValueError(f"{path}: {kind} '{name}' must state some of {list(allowed)}; unknown {unknown}")
    return Presets(config["base"], config["data"], config["models"])


def parse(field: str, text: str) -> Union[str, int, float, None]:
    if text.lower() == NONE:
        return None
    return NUMBERS.get(field, str)(text)


def build_spec(base: Dict[str, Any], data: Union[Dict[str, Any], None], model: Union[Dict[str, Any], None], changes: Dict[str, Any]) -> RunSpec:
    return RunSpec(**{**base, **(data or {}), **(model or {}), **changes})
