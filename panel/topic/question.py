"""
A question: what follows a description, in one or several forms, in each language it exists in.

`config/topics/<topic>/questions/<name>.json`. A question belongs to a topic, which it names,
and is separate from the framing so that a run can change it and nothing else. One file holds
the question in every language; a run takes the one its framing is written in.

- a language's `text` is one wording for everyone, or key -> wording when `by` says what the keys are.
- `by` names a shared vocabulary (`panel.dataset.SharedRef`); each language decides for itself
  whether it needs one.
- `fallback` is optional. Without it a dataset that does not share the vocabulary, or a
  subject without an answer, cannot be asked this question.
- a variant is additive: new wording is a new file, never an edit of a text a run has recorded.
  Adding a language to a file changes no recorded text.
"""
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Union

import pandas as pd

from panel import paths
from panel.config import check_keys, check_short

from .vocabulary import Topic

QUESTIONS_DIR = "questions"
KEYS = {"name", "short", "languages"}
REQUIRED_KEYS = {"text"}
OPTIONAL_KEYS = {"by", "fallback", "note"}


@dataclass(frozen=True)
class Question:
    """
    One question in one language.

    text: the one wording, or None when the question is worded `by` a vocabulary.
    forms: key -> wording; empty for a question with one wording.
    fallback: the key used for a subject without one, or None.
    """
    name: str
    short: str
    language: str
    text: Union[str, None]
    by: Union[str, None]
    forms: Dict[str, str]
    fallback: Union[str, None]
    note: str = ""

    def wording(self, key: Union[str, None] = None) -> str:
        """`key` is the subject's key in the `by` vocabulary; None or missing for a subject without one."""
        if self.by is None:
            return self.text
        if key is None or pd.isna(key):
            if self.fallback is None:
                raise ValueError(f"question '{self.name}' is worded by '{self.by}' and has no fallback")
            return self.forms[self.fallback]
        if key not in self.forms:
            raise ValueError(f"question '{self.name}' has no form for {self.by} '{key}'; it has {sorted(self.forms)}")
        return self.forms[key]


def _read(path: Path) -> dict:
    config = json.loads(path.read_text(encoding="utf-8"))
    if set(config) != KEYS:
        raise ValueError(f"{path} needs exactly the keys {sorted(KEYS)}")
    return config


def question_names(topic: str, language: str, topics_dir: Union[str, Path] = paths.TOPICS_DIR) -> List[str]:
    folder = Path(topics_dir) / topic / QUESTIONS_DIR
    return sorted(path.stem for path in folder.glob("*.json") if language in _read(path)["languages"])


def load_question(topic: Topic, language: str, name: str) -> Question:
    path = topic.dir / QUESTIONS_DIR / f"{name}.json"
    if not path.exists():
        raise FileNotFoundError(f"no question '{name}' for topic '{topic.name}' at {path}")
    config = _read(path)
    if config["name"] != name:
        raise ValueError(f"{path} declares name '{config['name']}'")
    short = check_short(config["short"], str(path))
    if language not in config["languages"]:
        raise ValueError(f"{path}: no '{language}' wording; it has {sorted(config['languages'])}")

    entry = config["languages"][language]
    check_keys(entry, REQUIRED_KEYS, OPTIONAL_KEYS, f"{path}: '{language}'")
    text, by, fallback = entry["text"], entry.get("by"), entry.get("fallback")
    several = isinstance(text, dict)
    if several != (by is not None):
        raise ValueError(f"{path}: '{language}' text holds several forms exactly when 'by' names what they are keyed by")
    if several and not text:
        raise ValueError(f"{path}: '{language}' text holds no form")
    if fallback is not None and (not several or fallback not in text):
        raise ValueError(f"{path}: '{language}' fallback '{fallback}' is not one of the forms")
    return Question(
        name=name, short=short, language=language, text=None if several else text, by=by,
        forms=dict(text) if several else {}, fallback=fallback, note=entry.get("note", ""),
    )
