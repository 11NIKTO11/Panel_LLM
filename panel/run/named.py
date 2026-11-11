"""
Named runs: the specs worth a name, kept as data so the runner and the analysis share one list.

`config/runs.json`:

    {
        "base": {"dataset": "sod", "description": "default", "topic": ..., "random_state": 42},
        "runs": {
            "nano":    {"client": "OpenRouter", "model": "gpt-4.1-nano", "temperature": 0.0, "effort": null, "sample": null},
            "neutral": {"client": ..., "sample": null, "question": "neutral"}
        }
    }

The base says what is asked, of whom and in what form; a run says which model answers and how
(`OWN`: client, model, temperature, effort, sample). Every run states all of `OWN` and the
base none of it, so changing the base moves the experiment and never a model's settings. A
run may also replace a base field. Base and run together state every field of a spec, the
ones with a default included. One level: a run does not build on another run. The name is a
handle for people; a saved run is identified by its stem and its record, which come from the
spec, so renaming a run renames nothing on disk.
"""
import json
from dataclasses import fields
from pathlib import Path
from typing import Any, Dict, Union

from .spec import RunSpec

FIELDS = tuple(field.name for field in fields(RunSpec))
KEYS = {"base", "runs"}
OWN = ("client", "model", "temperature", "effort", "sample")
NUMBERS = {"temperature": float, "repeats": int, "sample": int, "random_state": int}
NONE = "none"


def load_runs(path: Union[str, Path]) -> Dict[str, Dict[str, Any]]:
    """
    Every named run as its fields, the base laid under it; name -> fields, in file order.

    Raises:
        FileNotFoundError: if there is no such file.
        ValueError: on keys other than `base` and `runs`, a base that sets a field of `OWN` or
            lacks any other, a run that lacks a field of `OWN`, or a field a spec does not have.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"no named runs at {path}")
    config = json.loads(path.read_text(encoding="utf-8"))
    if set(config) != KEYS:
        raise ValueError(f"{path} needs exactly the keys {sorted(KEYS)}")
    base = config["base"]
    if set(base) != set(FIELDS) - set(OWN):
        raise ValueError(f"{path}: the base must state exactly {sorted(set(FIELDS) - set(OWN))}, not {sorted(base)}")
    for name, named in config["runs"].items():
        missing, unknown = sorted(set(OWN) - set(named)), sorted(set(named) - set(FIELDS))
        if missing or unknown:
            raise ValueError(f"{path}: '{name}' must state {list(OWN)}: missing {missing}, unknown {unknown}")
    return {name: {**base, **named} for name, named in config["runs"].items()}


def parse(field: str, text: str) -> Union[str, int, float, None]:
    """
    A field's value from the text a person typed; `none` is None.

    Raises:
        ValueError: if a number field gets something else.
    """
    if text.lower() == NONE:
        return None
    return NUMBERS.get(field, str)(text)


def build_spec(named: Union[Dict[str, Any], None], changes: Dict[str, Any]) -> RunSpec:
    """
    A spec from a named run, from fields given one by one, or from a named run with some replaced.

    Args:
        named: a named run's fields, or None to start from nothing.
        changes: fields that replace the named run's.

    Raises:
        ValueError: if a field without a default is given by neither.
    """
    given = {**(named or {}), **changes}
    try:
        return RunSpec(**given)
    except TypeError as error:
        raise ValueError(f"not a complete run: {error}; given {sorted(given)}") from None
