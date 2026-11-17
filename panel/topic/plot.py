"""
How a topic likes to be drawn: a colour and a short label per option, as data.

`config/topics/<name>/plot.json`, optional: `colors` is column -> hex colour, for every option or
none and both turnout columns or neither; `labels` is column -> the label a figure shows, for any
subset; `last` names the options a ranking puts at the end, whatever their size (a catch-all).

A topic without the file, or a caller that asks for none, gets the empty style: figures then
use their own palette and name an option by its id.
"""
import json
import re
from dataclasses import dataclass, field
from typing import Dict, Tuple

from .vocabulary import Topic

CONFIG_NAME = "plot.json"
KEYS = {"colors", "labels", "last", "note"}
HEX = re.compile(r"#[0-9a-fA-F]{6}")


@dataclass(frozen=True)
class PlotStyle:
    colors: Dict[str, str] = field(default_factory=dict)
    labels: Dict[str, str] = field(default_factory=dict)
    last: Tuple[str, ...] = ()


def load_plot_style(topic: Topic) -> PlotStyle:
    path = topic.dir / CONFIG_NAME
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
