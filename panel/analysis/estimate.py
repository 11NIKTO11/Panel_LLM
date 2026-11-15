"""
How a table names a value that comes with an interval: the value under its own name, the
interval's ends beside it as `<name>_low` and `<name>_high`.

The one place the two suffixes are spelled, so that a table's writer and its readers agree.
"""
from typing import Tuple

ENDS = ("_low", "_high")


def low(name: str) -> str:
    return f"{name}{ENDS[0]}"


def high(name: str) -> str:
    return f"{name}{ENDS[1]}"


def bounds(name: str) -> Tuple[str, str, str]:
    """The three columns of one estimate: the value, the interval's low end, its high end."""
    return name, low(name), high(name)


def is_end(column: str) -> bool:
    """Whether a column holds an end of an interval and not a value."""
    return column.endswith(ENDS)
