"""
Language packs: the grammar a template cannot express, one folder per language code.

A pack is `panel/describe/lang/<code>/`: `grammar.py` with the primitives (age clause,
negation, gender agreement), `filters.py` naming the filters templates may call and the
`usable` predicate that drops non-answers, and `forms/` tables for closed sets such as region
names in their cases. A description declares its language and pulls that pack; a dataset may
add filters for its own quirks at render time.
"""
import importlib
from dataclasses import dataclass
from typing import Any, Callable, Mapping


@dataclass(frozen=True)
class Language:
    """One pack, loaded: its code, the filters templates may name, and what counts as an answer."""
    code: str
    filters: Mapping[str, Callable]
    usable: Callable[[Any], bool]


def load_language(code: str) -> Language:
    """
    The pack for a language code.

    Raises:
        ValueError: if no pack exists for the code.
    """
    module_name = f"{__name__}.{code}.filters"
    try:
        module = importlib.import_module(module_name)
    except ModuleNotFoundError as error:
        if error.name and module_name.startswith(error.name):
            raise ValueError(f"no language pack for '{code}' under {__name__}") from error
        raise
    return Language(code=code, filters=dict(module.FILTERS), usable=module.usable)
