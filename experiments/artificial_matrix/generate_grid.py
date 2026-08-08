"""Generate the full-factorial sociodemographic grid for the artificial-matrix experiment.

Composition root: reads a grid config (factor names and their ordered levels)
and writes the grid as CSV. Factor vocabulary lives entirely in the config; this
script only combines whatever factors it declares.

Factor names are column names of the original survey data (data/clean_data.csv), so
every grid variant shares one column schema. Factor *levels* are the variant: each
config declares its own value vocabulary.

    grid_config_sod.json        levels taken verbatim from clean_data.csv
    grid_config_colleague.json  levels as used in the colleague's experiment

Rows are identified by an integer id in an unnamed leading column, matching the row
convention of clean_data.csv. Output defaults to data/grid_<grid_name>.csv, so
variants never overwrite each other.

Usage:
    python experiments/artificial_matrix/generate_grid.py --config <PATH> [--output PATH]
"""
import argparse
import itertools
import json
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).parents[2]
DEFAULT_CONFIG = Path(__file__).parent / "grid_config_sod.json"
OUTPUT_DIR = PROJECT_ROOT / "data"


def build_grid(factors: dict) -> pd.DataFrame:
    names = list(factors)
    rows = [dict(zip(names, combo)) for combo in itertools.product(*factors.values())]
    df = pd.DataFrame(rows)
    df.index = pd.RangeIndex(1, len(df) + 1)
    return df


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG, help="grid config JSON")
    parser.add_argument("--output", type=Path, default=None,
                        help="output CSV path (default: data/grid_<grid_name>.csv)")
    args = parser.parse_args()

    config = json.loads(args.config.read_text(encoding="utf-8"))
    grid = build_grid(config["factors"])
    output = args.output or OUTPUT_DIR / f"grid_{config['grid_name']}.csv"

    output.parent.mkdir(parents=True, exist_ok=True)
    grid.to_csv(output, index=True)

    sizes = " x ".join(str(len(levels)) for levels in config["factors"].values())
    print(f"Grid '{config['grid_name']}': {sizes} = {len(grid)} profiles -> {output}")


if __name__ == "__main__":
    main()
