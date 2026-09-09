"""Generate the full-factorial sociodemographic grid for the artificial-matrix experiment.

Composition root: reads a grid config (factor names and their ordered levels)
and writes the grid as CSV. Factor vocabulary lives entirely in the config; this
script only combines whatever factors it declares.

Factor names are column names of the original survey data (data/clean_data.csv), so
every grid variant shares one column schema. Factor *levels* are the variant: each
config declares its own value vocabulary.

    grid_config_sod.json     levels taken verbatim from clean_data.csv, exact ages
    grid_config_simple.json  levels as used in the colleague's experiment, age bands

A factor holding bands rather than exact values must be named for what it holds:
`age_range`, not `age`. The description template's first two alternatives are the age
priority: exact `age` if the config varies it, else the band ("je mi mezi 30 a 44 lety").

A config that declares `description_template` is a prompt source, and its grid is written
with the rendered description as a column. One that does not is for reading someone else's
results, not for generating ours.

Rows are identified by an integer id in an unnamed leading column, matching the row
convention of clean_data.csv.

Paths are the config's. The template it names and the grid it writes both resolve beside
it, so a variant's config, its wording and its profiles travel together and no variant
overwrites another; `--output` puts the CSV anywhere else. Both paths may be absolute or
relative to wherever the script is called from.

Usage:
    python -m datasets.grid.generate_grid <config.json> [--output PATH]

Run it from the repo root - that is what puts `utils` on the import path.
"""
import argparse
import itertools
import json
from pathlib import Path

import pandas as pd

from utils import czech_grammar as cz
from utils.prompts import load_template, render_frame

FILTERS = {"years": cz.age_clause}      # grid levels are survey vocabulary; only age needs grammar


def build_grid(factors: dict) -> pd.DataFrame:
    names = list(factors)
    rows = [dict(zip(names, combo)) for combo in itertools.product(*factors.values())]
    df = pd.DataFrame(rows)
    df.index = pd.RangeIndex(1, len(df) + 1)
    return df


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("config", type=Path, help="grid config JSON")
    parser.add_argument("--output", type=Path, default=None,
                        help="output CSV (default: grid_<grid_name>.csv beside the config)")
    args = parser.parse_args()

    config = json.loads(args.config.read_text(encoding="utf-8"))
    variant_dir = args.config.parent
    grid = build_grid(config["factors"])
    output = args.output or variant_dir / f"grid_{config['grid_name']}.csv"

    # A config that names a template is a prompt source, and its grid carries the rendered
    # description as a column - no separate table, because a grid is regenerated anyway.
    template_name = config.get("description_template")
    if template_name:
        template = load_template(variant_dir / f"{template_name}.json")
        grid[template.name] = render_frame(template, grid, FILTERS)

    output.parent.mkdir(parents=True, exist_ok=True)
    grid.to_csv(output, index=True)

    sizes = " x ".join(str(len(levels)) for levels in config["factors"].values())
    print(f"Grid '{config['grid_name']}': {sizes} = {len(grid)} profiles -> {output}")
    if template_name:
        print(f"described as '{template_name}':\n{grid[template_name].iloc[0]}")
    else:
        print("no description_template: this config is not a prompt source")


if __name__ == "__main__":
    main()
