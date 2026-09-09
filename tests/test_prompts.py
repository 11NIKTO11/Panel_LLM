"""Description tables: the stored tables are the baseline ones, and rendering reproduces them."""
import pandas as pd
import pytest

from panel.dataset import load_dataset, profile_path, read_profile
from panel.describe.render import render
from panel.describe.tables import read_table
from panel.describe.template import load_description

from tests.conftest import load_fixture, require
from tests.snapshots import prompt_table_digest

BASELINE = load_fixture("prompt_tables")


def stored(dataset: str, description: str) -> pd.Series:
    return read_table(dataset, description).iloc[:, 0]


def assert_renders_identical(people: pd.DataFrame, dataset: str, description: str) -> None:
    rendered = render(load_dataset(dataset), load_description(description), people)
    table = stored(dataset, description)
    assert rendered.index.equals(table.index)
    identical = int((rendered.values == table.values).sum())
    assert identical == len(table), f"{dataset}/{description}: {identical} of {len(table)} rows identical"


@pytest.mark.parametrize("key", sorted(BASELINE))
def test_prompt_table_unchanged(key):
    assert prompt_table_digest(stored(*key.split("/"))) == BASELINE[key]


def test_prompts_v1_identical(people):
    assert_renders_identical(people, "sod", "self_v1")


def test_prompts_v2_identical(people):
    assert_renders_identical(people, "sod", "self_v2")


def test_shared_template_grid():
    grid = load_dataset("grid")
    require(profile_path(grid))
    rendered = render(grid, load_description("self_v1"), read_profile(grid))
    assert (rendered.str.count(r"\. ") == 2).all() and rendered.str.endswith(".").all()
    assert prompt_table_digest(rendered) == BASELINE["grid/self_v1"]
