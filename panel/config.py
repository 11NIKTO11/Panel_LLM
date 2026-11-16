"""
The checks every declaration file gets: exactly the keys its loader knows, and a `short` that
can sit in a results stem.

A loader reads its own file and knows its own keys; what is the same in all of them is how a
wrong one is refused, and that lives here.
"""
from typing import AbstractSet, Mapping


def check_keys(config: Mapping, required: AbstractSet[str], optional: AbstractSet[str], where: str) -> None:
    """`where` is what the error names: the file, and the block inside it if any."""
    unknown = sorted(set(config) - required - optional)
    missing = sorted(required - set(config))
    if unknown or missing:
        raise ValueError(f"{where}: unknown keys {unknown}, missing keys {missing}")


def check_short(short: str, where: str) -> str:
    """A `short` is the part of a results stem a declaration stands for; an underscore separates the parts."""
    if not short or not short.isascii() or "_" in short:
        raise ValueError(f"{where}: short '{short}' must be non-empty ASCII without underscores")
    return short
