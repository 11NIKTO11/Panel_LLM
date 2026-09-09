"""
Build the regions' profile table: the fifteen area names -> `data/profiles/regions.csv`.

    python -m datasets.regions.prepare

Composition root for the regions slice: reads `regions.csv`, keeps the `region` column and
numbers the rows 1..15 in file order.
"""
import pandas as pd

from panel.dataset import load_dataset, write_profile
from panel.schema import load_schema


def build(source: pd.DataFrame) -> pd.DataFrame:
    """One profile row per area, in the source's order."""
    profile = source[["region"]].copy()
    profile.index = pd.RangeIndex(1, len(profile) + 1, name="id")
    return profile


def main() -> None:
    dataset = load_dataset("regions")
    profile = build(pd.read_csv(dataset.dir / dataset.source, encoding="utf-8"))
    path = write_profile(dataset, profile, load_schema())
    print(f"{len(profile)} areas -> {path}")


if __name__ == "__main__":
    main()
