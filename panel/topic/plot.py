"""
How a topic likes to be drawn: a colour and a short label per option, as data.

`config/topics/<name>/plot.json`, optional:

    {
        "colors": {"ANO": "#261060", ...},     column -> hex colour; every option or none,
                                               and both turnout columns or neither
        "labels": {"ANO": "ANO", ...},         column -> the label a figure shows; any subset
        "last": ["Other"],                     options a ranking puts at the end, whatever their size
        "note": "free text, ignored by code"
    }

A topic without the file, or a caller that asks for none, gets the empty style: figures then
use their own palette and name an option by its id.
"""
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Tuple, Union

from panel import paths

from .vocabulary import Topic

CONFIG_NAME = "plot.json"
KEYS = {"colors", "labels", "last", "note"}
HEX = re.compile(r"#[0-9a-fA-F]{6}")


@dataclass(frozen=True)
class PlotStyle:
    """
    Attributes:
        colors: topic column -> hex colour; all the options or none of them, and both
            turnout columns or neither.
        labels: topic column -> display label, for the columns that have one.
        last: options a figure that ranks the options draws after the others (a catch-all).
    """
    colors: Dict[str, str] = field(default_factory=dict)
    labels: Dict[str, str] = field(default_factory=dict)
    last: Tuple[str, ...] = ()


def load_plot_style(topic: Topic, topics_dir: Union[str, Path] = paths.TOPICS_DIR) -> PlotStyle:
    """
    The topic's plot style, or the empty one if it has no `plot.json`.

    Raises:
        ValueError: on unknown keys, an id the topic does not have, colours for only some of
            the options or one turnout column, or a colour that is not `#rrggbb`.
    """
    path = Path(topics_dir) / topic.name / CONFIG_NAME
    if not path.exists():
        return PlotStyle()
    config = json.loads(path.read_text(encoding="utf-8"))
    colors, labels = dict(config.get("colors", {})), dict(config.get("labels", {}))
    unknown = sorted(set(config) - KEYS)
    last = tuple(config.get("last", ()))
    foreign = sorted((set(colors) | set(labels)) - set(topic.columns)) + sorted(set(last) - set(topic.options))
    partial = [
        column for group in (topic.options, topic.turnout or ())
        if set(group) & set(colors) for column in group if column not in colors
    ]
    malformed = sorted(option for option, color in colors.items() if not HEX.fullmatch(color))
    if unknown or foreign or partial or malformed:
        raise ValueError(
            f"{path}: unknown keys {unknown}, ids the topic does not have {foreign}, "
            f"columns without a colour {partial}, colours that are not #rrggbb {malformed}"
        )
    return PlotStyle(colors=colors, labels=labels, last=last)
