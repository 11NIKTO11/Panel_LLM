"""
Where experts place a topic's options: positions on rated items, as data.

`config/topics/<name>/positions.json`, optional:

    {
        "scale": [0, 10],
        "axes": {
            "x": {"item": "lrecon", "label": "economic left (0) - right (10)"},
            "y": {"item": "galtan", "label": "GAL (0) - TAN (10)"}
        },
        "positions": {"ANO": {"lrecon": 4.5, "galtan": 5.73, ...}, ...},
        "spread": {"lrecon": {"mean": 4.6, "std": 2.1}, ...},
        "note": "free text, ignored by code"
    }

`positions` rates some of the topic's options, every one on the same items; `scale` is the
lowest and the highest rating an item can get, which a map's axes span; `axes` names the
two items a map is drawn in; `spread` gives each item's mean and standard deviation in the
population the ratings come from, to put the items on one scale. No model sees this file.
"""
import json
from dataclasses import dataclass
from typing import Dict, Tuple, Union

import pandas as pd

from .vocabulary import Topic

CONFIG_NAME = "positions.json"
KEYS = {"scale", "axes", "positions", "spread", "note"}


@dataclass(frozen=True)
class Positions:
    """
    Attributes:
        table: one row per rated option, in the topic's order, a column per item.
        axes: the two items of the map, x then y.
        labels: item -> axis label, for the two axes.
        spread: indexed by item, the columns `mean` and `std`.
        scale: the lowest and the highest rating an item can get.
    """
    table: pd.DataFrame
    axes: Tuple[str, str]
    labels: Dict[str, str]
    spread: pd.DataFrame
    scale: Tuple[float, float]

    @property
    def plane(self) -> pd.DataFrame:
        """The rated options in the two map items."""
        return self.table[list(self.axes)]

    @property
    def standardized(self) -> pd.DataFrame:
        """Every item standardised by `spread`."""
        return (self.table - self.spread["mean"]) / self.spread["std"]


def load_positions(topic: Topic) -> Union[Positions, None]:
    """
    The topic's expert positions, or None if it has no `positions.json`.

    Raises:
        ValueError: on unknown or missing keys, an option the topic does not have, options
            rated on different items, or an axis or spread that does not match the items.
    """
    path = topic.dir / CONFIG_NAME
    if not path.exists():
        return None
    config = json.loads(path.read_text(encoding="utf-8"))
    if set(config) - KEYS or not KEYS - {"note"} <= set(config):
        raise ValueError(f"{path}: needs the keys {sorted(KEYS - {'note'})}, has {sorted(config)}")
    table = pd.DataFrame(config["positions"], dtype=float).T
    foreign = sorted(set(table.index) - set(topic.options))
    axes = tuple(config["axes"][side]["item"] for side in ("x", "y"))
    spread = pd.DataFrame(config["spread"], dtype=float).T
    if foreign or table.isna().any().any() or not set(axes) <= set(table.columns) or set(spread.index) != set(table.columns):
        raise ValueError(f"{path}: options the topic does not have {foreign}, or options, axes and spread that do not share one set of items")
    return Positions(
        table=table.loc[[option for option in topic.options if option in table.index]], axes=axes,
        labels={config["axes"][side]["item"]: config["axes"][side]["label"] for side in ("x", "y")}, spread=spread.loc[table.columns],
        scale=(float(config["scale"][0]), float(config["scale"][1])),
    )
