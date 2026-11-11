"""
Presets: a base run, and the data and the models worth a name, kept as data.

`config/presets.json`:

    {
        "base": {"dataset": "sod", "description": "default", "topic": "election_2021", "framing": "cs",
                 "question": "default", "shape": "Prob", "repeats": 1,
                 "client": "Fake", "model": "fake-model", "temperature": 0.0, "effort": null},
        "data": {
            "sod-extended": {"dataset": "sod", "description": "extended"},
            "direct":       {"dataset": "regions", "description": "direct", "question": null, "repeats": 10}
        },
        "models": {
            "nano":      {"client": "OpenRouter", "model": "gpt-4.1-nano", "temperature": 0.0},
            "haiku-5.5": {"client": "Anthropic", "model": "claude-haiku-5-5", "temperature": null, "effort": "low"}
        }
    }

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
from typing import Any, Dict, Union

from .spec import RunSpec

KEYS = {"base", "data", "models"}
FIELDS = tuple(field.name for field in fields(RunSpec))
DATA = ("dataset", "description", "topic", "framing", "question", "shape", "repeats")
MODEL = ("client", "model", "temperature", "effort")
NUMBERS = {"temperature": float, "repeats": int, "sample": int, "random_state": int}
NONE = "none"


@dataclass(frozen=True)
class Presets:
    """
    Attributes:
        base: every field of `DATA` and `MODEL`.
        data: name -> the `DATA` fields it replaces, in file order.
        models: name -> the `MODEL` fields it replaces, in file order.
    """
    base: Dict[str, Any]
    data: Dict[str, Dict[str, Any]]
    models: Dict[str, Dict[str, Any]]


def load_presets(path: Union[str, Path]) -> Presets:
    """
    Raises:
        FileNotFoundError: if there is no such file.
        ValueError: on keys other than `base`, `data` and `models`, a base that does not state
            exactly the fields of both kinds, or a preset that states nothing or a field that
            is not of its kind.
    """
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
    """
    A field's value from the text a person typed; `none` is None.

    Raises:
        ValueError: if a number field gets something else.
    """
    if text.lower() == NONE:
        return None
    return NUMBERS.get(field, str)(text)


def build_spec(base: Dict[str, Any], data: Union[Dict[str, Any], None], model: Union[Dict[str, Any], None], changes: Dict[str, Any]) -> RunSpec:
    """
    A spec from the base, with a data preset, a model preset and single fields laid over it.

    Args:
        base: the base run's fields.
        data: a data preset's fields, or None.
        model: a model preset's fields, or None.
        changes: fields that replace all of those.
    """
    return RunSpec(**{**base, **(data or {}), **(model or {}), **changes})
