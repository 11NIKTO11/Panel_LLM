"""
A framing: everything the model reads that is not the description, in one language.

`config/topics/<topic>/<framing>.json` carries the system prompt, the labels the options are shown under, and the wording of the response
schema - class titles and field descriptions - which the provider sees in the JSON schema and
which therefore are prompt content. A topic has one framing per language (`cs`, `en`) and may
have more (`cs-terse`); the framing's `short` goes into the results stem.

Config shape:

    {
        "name": "cs",                          the file stem
        "topic": "election_2021",              the folder
        "language": "cs",
        "short": "cs",                         stem part: ASCII, no underscore
        "title": "...",                        human label, for figures and reports
        "instructions": "<system prompt>",     null for a framing whose runs send none
        "labels": {"SPOLU": "SPOLU coalition"},   option id -> label; the native name by default
        "schema": {
            "turnout": {"title": "TurnoutProbability", "description": "...",      only for a topic
                        "voted": "...", "not_voted": "...", "field": "..."},     with turnout
            "option": {"title": "OptionProbability", "description": "...", "name": "...", "probability": "..."},
            "probabilities": {"description": "...", "options": "..."},
            "pick": {"description": "...", "choice": "..."}
        },
        "note": "free text, ignored by code"
    }

The system prompt can depend on what else a run names, and nothing links them: a prompt that
lists the profile's attributes fits one description, one that speaks of probabilities fits
one response shape. Nothing checks this; a framing that makes such an assumption says so in
its `note`.

The loader raises on a label for an id the topic lacks, on labels that are not unique after
defaults, and on schema text that disagrees with the topic about turnout.
"""
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Union

from panel import paths

from .plot import CONFIG_NAME as PLOT_CONFIG_NAME
from .polls import CONFIG_NAME as POLLS_CONFIG_NAME
from .positions import CONFIG_NAME as POSITIONS_CONFIG_NAME
from .vocabulary import CONFIG_NAME as TOPIC_CONFIG_NAME, Topic

REQUIRED_KEYS = {"name", "topic", "language", "short", "title", "instructions", "labels", "schema"}
OPTIONAL_KEYS = {"note"}
TEXT_KEYS = {
    "turnout": {"title", "description", "voted", "not_voted", "field"},
    "option": {"title", "description", "name", "probability"},
    "probabilities": {"description", "options"},
    "pick": {"description", "choice"},
}


@dataclass(frozen=True)
class TurnoutText:
    """The turnout model's title and descriptions; `field` describes `voted_or_not` on both models."""
    title: str
    description: str
    voted: str
    not_voted: str
    field: str


@dataclass(frozen=True)
class OptionText:
    """The per-option probability model's title and descriptions."""
    title: str
    description: str
    name: str
    probability: str


@dataclass(frozen=True)
class ProbabilitiesText:
    """`Probabilities`: its description and that of `options`."""
    description: str
    options: str


@dataclass(frozen=True)
class PickText:
    """`Pick`: its description and that of `choice`."""
    description: str
    choice: str


@dataclass(frozen=True)
class SchemaText:
    """Every piece of the response schema the provider reads; `turnout` is None for a topic without one."""
    option: OptionText
    probabilities: ProbabilitiesText
    pick: PickText
    turnout: Union[TurnoutText, None]


@dataclass(frozen=True)
class Framing:
    """
    Attributes:
        name: the file stem under the topic folder.
        topic: the topic it words.
        language: language code of every text here.
        short: the framing's part of a results stem.
        title: human label, for figures and reports.
        instructions: system prompt; None for a framing whose runs send none.
        labels: option id -> the label the model sees, complete over the topic's options.
        schema: the response schema wording.
        note: free text; not used by code.
    """
    name: str
    topic: str
    language: str
    short: str
    title: str
    instructions: Union[str, None]
    labels: Dict[str, str]
    schema: SchemaText
    note: str = ""


def framing_names(topic: str, topics_dir: Union[str, Path] = paths.TOPICS_DIR) -> List[str]:
    """Every framing file beside a topic's `topic.json`, sorted."""
    folder = Path(topics_dir) / topic
    return sorted(path.stem for path in folder.glob("*.json") if path.name not in (TOPIC_CONFIG_NAME, PLOT_CONFIG_NAME, POLLS_CONFIG_NAME, POSITIONS_CONFIG_NAME))


def _text(config: dict, block: str, path: Path) -> dict:
    """One block of schema text with exactly its keys."""
    entry = config.get(block)
    if not isinstance(entry, dict) or set(entry) != TEXT_KEYS[block]:
        raise ValueError(f"{path}: schema.{block} needs exactly the keys {sorted(TEXT_KEYS[block])}")
    return entry


def load_framing(topic: Topic, name: str, topics_dir: Union[str, Path] = paths.TOPICS_DIR) -> Framing:
    """
    Read one framing of a topic.

    Raises:
        FileNotFoundError: if the topic folder holds no framing of that name.
        ValueError: on unknown or missing keys, a name or topic that disagrees with the file, a
            short that could not sit in a stem, a label for an unknown option, labels that are
            not unique after defaults, or schema text that disagrees with the topic on turnout.
    """
    path = Path(topics_dir) / topic.name / f"{name}.json"
    if not path.exists():
        raise FileNotFoundError(f"no framing '{name}' for topic '{topic.name}' at {path}")
    config = json.loads(path.read_text(encoding="utf-8"))
    unknown = sorted(set(config) - REQUIRED_KEYS - OPTIONAL_KEYS)
    missing = sorted(REQUIRED_KEYS - set(config))
    if unknown or missing:
        raise ValueError(f"{path}: unknown keys {unknown}, missing keys {missing}")
    if config["name"] != name or config["topic"] != topic.name:
        raise ValueError(f"{path} declares name '{config['name']}' and topic '{config['topic']}'")
    short = config["short"]
    if not short or not short.isascii() or "_" in short:
        raise ValueError(f"{path}: short '{short}' must be non-empty ASCII without underscores")

    foreign = sorted(set(config["labels"]) - set(topic.options))
    if foreign:
        raise ValueError(f"{path}: labels for options the topic does not have: {foreign}")
    labels = {option: config["labels"].get(option, topic.names[option]) for option in topic.options}
    if len(set(labels.values())) != len(labels):
        raise ValueError(f"{path}: option labels are not unique after defaults")

    schema = config["schema"]
    blocks = sorted(set(schema) - set(TEXT_KEYS))
    if blocks:
        raise ValueError(f"{path}: unknown schema blocks {blocks}")
    if ("turnout" in schema) != (topic.turnout is not None):
        raise ValueError(f"{path}: schema.turnout must be present exactly when the topic has turnout")
    text = SchemaText(
        option=OptionText(**_text(schema, "option", path)),
        probabilities=ProbabilitiesText(**_text(schema, "probabilities", path)),
        pick=PickText(**_text(schema, "pick", path)),
        turnout=TurnoutText(**_text(schema, "turnout", path)) if topic.turnout else None,
    )
    return Framing(
        name=name, topic=topic.name, language=config["language"], short=short, title=config["title"],
        instructions=config["instructions"], labels=labels, schema=text,
        note=config.get("note", ""),
    )
