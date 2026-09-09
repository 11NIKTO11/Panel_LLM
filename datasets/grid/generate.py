"""
Build the grid's profile table: every combination of the factor levels in its config.

    python -m datasets.grid.generate

Composition root for the grid slice. `dataset.json` names the config (`grid_income3.json`),
which declares factor columns and their ordered levels in the survey's own vocabulary, so a
profile built here reads like a respondent and shares the survey's column schema. Rows are
numbered 1..n in `itertools.product` order, so a regenerated grid keeps its ids.

Which factors to vary and at which levels is a research choice, and lives only in the config.
"""
import itertools
import json

import pandas as pd

from panel.dataset import load_dataset, write_profile
from panel.schema import load_schema

CONFIG_KEYS = {"factors"}
OPTIONAL_CONFIG_KEYS = {"note"}


def build(factors: dict) -> pd.DataFrame:
    """The full factorial of `factors`, one profile per row, ids 1..n."""
    names = list(factors)
    rows = [dict(zip(names, combination)) for combination in itertools.product(*factors.values())]
    grid = pd.DataFrame(rows)
    grid.index = pd.RangeIndex(1, len(grid) + 1, name="id")
    return grid


def main() -> None:
    dataset = load_dataset("grid")
    path = dataset.dir / dataset.source
    config = json.loads(path.read_text(encoding="utf-8"))
    unknown = sorted(set(config) - CONFIG_KEYS - OPTIONAL_CONFIG_KEYS)
    missing = sorted(CONFIG_KEYS - set(config))
    if unknown or missing:
        raise ValueError(f"{path}: unknown keys {unknown}, missing keys {missing}")

    profile = build(config["factors"])
    written = write_profile(dataset, profile, load_schema())
    sizes = " x ".join(str(len(levels)) for levels in config["factors"].values())
    print(f"{sizes} = {len(profile)} profiles -> {written}")


if __name__ == "__main__":
    main()
