"""
Build the survey's profile table: the raw `.sav` -> `data/profiles/sod.csv`.

    python -m datasets.sod.prepare [--sav PATH]

Composition root for the survey slice: reads the raw file, maps every row through `clean`,
and hands the table to `panel.dataset.write_profile`, which refuses a table missing a
required answer or carrying a column the schema does not know. Then it renders every
description the dataset declares into `data/prompts/sod/`, in the same pass, so a prompt table
never lags the rows it describes.
"""
import argparse
from pathlib import Path
from typing import Union

import pandas as pd

from panel.dataset import load_dataset, write_profile
from panel.describe.render import render_all
from panel.schema import load_schema

from .clean import process_respondent


def build(raw: pd.DataFrame) -> pd.DataFrame:
    """One profile row per respondent, keyed by the survey's own id."""
    return pd.DataFrame([process_respondent(raw.loc[i]) for i in raw.index]).set_index("id")


def main(argv: Union[list, None] = None) -> None:
    dataset = load_dataset("sod")
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("--sav", type=Path, default=dataset.dir / dataset.source, help="the raw survey file")
    args = parser.parse_args(argv)

    profile = build(pd.read_spss(args.sav))
    path = write_profile(dataset, profile, load_schema())
    print(f"{len(profile)} respondents -> {path}")
    for name, table in render_all(dataset, profile).items():
        print(f"{name} -> {table}")


if __name__ == "__main__":
    main()
