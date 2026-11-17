"""
A framing: everything the model reads that is not the description or the question, in one language.

`config/topics/<topic>/framings/<name>.json` carries the system prompt, the labels the options are shown under, and the wording of the response
schema - class titles and field descriptions - which the provider sees in the JSON schema and
which therefore are prompt content. A topic has one framing per language (`cs`, `en`) and may
have more (`cs-terse`); the framing's `short` goes into the results stem.

The system prompt can depend on what else a run names, and nothing links them: a prompt that
lists the profile's attributes fits one description, one that speaks of probabilities fits
one response shape. Nothing checks this; a framing that makes such an assumption says so in
its `note`.

A framing states the answer in one of two ways, and a run sends what it states: `schema`, the
wording of a structured answer, or `answer`, a menu of one-token answers whose probabilities
are read off the model's next-token distribution (`DECISIONS.md` §40). The menu is the last
thing in the prompt.
"""
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Union

from panel import paths
from panel.config import check_keys, check_short

from .vocabulary import Topic

FRAMINGS_DIR = "framings"
REQUIRED_KEYS = {"name", "topic", "language", "short", "title", "instructions", "labels"}
ANSWER_FORMS = {"schema", "answer"}
OPTIONAL_KEYS = {"note"} | ANSWER_FORMS
ANSWER_KEYS = {"tokens", "line"}
ABSTAIN_KEYS = {"token", "label"}
TEXT_KEYS = {
    "turnout": {"title", "description", "voted", "not_voted", "field"},
    "option": {"title", "description", "name", "probability"},
    "probabilities": {"description", "options"},
    "pick": {"description", "choice"},
}


@dataclass(frozen=True)
class TurnoutText:
    """The turnout model's title and descriptions; `field` describes `voted_or_not` and is sent after `description`, in one text."""
    title: str
    description: str
    voted: str
    not_voted: str
    field: str


@dataclass(frozen=True)
class OptionText:
    title: str
    description: str
    name: str
    probability: str


@dataclass(frozen=True)
class ProbabilitiesText:
    description: str
    options: str


@dataclass(frozen=True)
class PickText:
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
class Abstain:
    token: str
    label: str


@dataclass(frozen=True)
class AnswerMenu:
    """
    tokens: option id -> the token that stands for it, in the order the menu lists them.
    abstain: the entry for not having voted, listed last; None for a topic without turnout.
    line: one menu line, with `{token}` and `{label}`.
    """
    tokens: Dict[str, str]
    abstain: Union[Abstain, None]
    line: str

    def text(self, labels: Dict[str, str]) -> str:
        entries = [(token, labels[option]) for option, token in self.tokens.items()]
        if self.abstain:
            entries.append((self.abstain.token, self.abstain.label))
        return "\n".join(self.line.format(token=token, label=label) for token, label in entries)


@dataclass(frozen=True)
class Framing:
    """
    short: the framing's part of a results stem.
    title: human label, for figures and reports.
    instructions: the system prompt; None for a framing whose runs send none.
    labels: option id -> the label the model sees, complete over the topic's options (the native name by default).
    schema, answer: exactly one is set.
    note: free text; not used by code.
    """
    name: str
    topic: str
    language: str
    short: str
    title: str
    instructions: Union[str, None]
    labels: Dict[str, str]
    schema: Union[SchemaText, None]
    answer: Union[AnswerMenu, None] = None
    note: str = ""

    @property
    def menu(self) -> Union[str, None]:
        return self.answer.text(self.labels) if self.answer else None


def framing_names(topic: str, topics_dir: Union[str, Path] = paths.TOPICS_DIR) -> List[str]:
    folder = Path(topics_dir) / topic / FRAMINGS_DIR
    return sorted(path.stem for path in folder.glob("*.json"))


def _text(config: dict, block: str, path: Path) -> dict:
    entry = config.get(block)
    if not isinstance(entry, dict) or set(entry) != TEXT_KEYS[block]:
        raise ValueError(f"{path}: schema.{block} needs exactly the keys {sorted(TEXT_KEYS[block])}")
    return entry


def _schema(schema: dict, topic: Topic, path: Path) -> SchemaText:
    blocks = sorted(set(schema) - set(TEXT_KEYS))
    if blocks:
        raise ValueError(f"{path}: unknown schema blocks {blocks}")
    if ("turnout" in schema) != (topic.turnout is not None):
        raise ValueError(f"{path}: schema.turnout must be present exactly when the topic has turnout")
    return SchemaText(
        option=OptionText(**_text(schema, "option", path)),
        probabilities=ProbabilitiesText(**_text(schema, "probabilities", path)),
        pick=PickText(**_text(schema, "pick", path)),
        turnout=TurnoutText(**_text(schema, "turnout", path)) if topic.turnout else None,
    )


def _answer(answer: dict, topic: Topic, path: Path) -> AnswerMenu:
    """Tokens are compared as a client matches them: case and surrounding space do not tell two apart."""
    check_keys(answer, ANSWER_KEYS | ({"abstain"} if topic.turnout else set()), set(), f"{path}: answer")
    tokens = answer["tokens"]
    if set(tokens) != set(topic.options):
        raise ValueError(f"{path}: answer.tokens must name exactly the topic's options")
    abstain = None
    if topic.turnout:
        check_keys(answer["abstain"], ABSTAIN_KEYS, set(), f"{path}: answer.abstain")
        abstain = Abstain(**answer["abstain"])
    every = [token.strip().casefold() for token in (*tokens.values(), *([abstain.token] if abstain else []))]
    if not all(every) or len(set(every)) != len(every):
        raise ValueError(f"{path}: answer tokens must be non-empty and differ by more than case")
    return AnswerMenu(tokens=dict(tokens), abstain=abstain, line=answer["line"])


def load_framing(topic: Topic, name: str) -> Framing:
    path = topic.dir / FRAMINGS_DIR / f"{name}.json"
    if not path.exists():
        raise FileNotFoundError(f"no framing '{name}' for topic '{topic.name}' at {path}")
    config = json.loads(path.read_text(encoding="utf-8"))
    check_keys(config, REQUIRED_KEYS, OPTIONAL_KEYS, str(path))
    if config["name"] != name or config["topic"] != topic.name:
        raise ValueError(f"{path} declares name '{config['name']}' and topic '{config['topic']}'")
    short = check_short(config["short"], str(path))

    foreign = sorted(set(config["labels"]) - set(topic.options))
    if foreign:
        raise ValueError(f"{path}: labels for options the topic does not have: {foreign}")
    labels = {option: config["labels"].get(option, topic.names[option]) for option in topic.options}
    if len(set(labels.values())) != len(labels):
        raise ValueError(f"{path}: option labels are not unique after defaults")

    if len(ANSWER_FORMS & set(config)) != 1:
        raise ValueError(f"{path}: a framing states exactly one of {sorted(ANSWER_FORMS)}")
    return Framing(
        name=name, topic=topic.name, language=config["language"], short=short, title=config["title"],
        instructions=config["instructions"], labels=labels,
        schema=_schema(config["schema"], topic, path) if "schema" in config else None,
        answer=_answer(config["answer"], topic, path) if "answer" in config else None,
        note=config.get("note", ""),
    )
