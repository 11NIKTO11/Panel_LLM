"""Description tables: the stored tables are the phase 0 ones, and rendering reproduces them."""
import pandas as pd
import pytest

from datasets.sod.describe import render

from tests.conftest import ROOT, load_fixture
from tests.snapshots import prompt_table_digest

BASELINE = load_fixture("prompt_tables")


def stored(variant: str) -> pd.Series:
    return pd.read_csv(ROOT / "data" / "prompts" / f"{variant}.csv", index_col=0).iloc[:, 0]


def assert_renders_identical(people: pd.DataFrame, variant: str) -> None:
    rendered = render(people, variant)
    table = stored(variant)
    assert rendered.index.equals(table.index)
    identical = int((rendered.values == table.values).sum())
    assert identical == len(table), f"{variant}: {identical} of {len(table)} rows identical"


@pytest.mark.parametrize("variant", sorted(BASELINE))
def test_prompt_table_unchanged(variant):
    assert prompt_table_digest(stored(variant)) == BASELINE[variant]


def test_prompts_v1_identical(people):
    assert_renders_identical(people, "self_description_v1")


def test_prompts_v2_identical(people):
    assert_renders_identical(people, "self_description_v2")
