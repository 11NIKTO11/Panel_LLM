"""
Shared fixtures: where things are, and loaders for the phase 0 baselines under `fixtures/`.

An input that is not in git (`data/clean_data.csv`, `results/`) makes a test skip naming
the path it needed, rather than fail.
"""
import json
from pathlib import Path
from typing import Any, List

import pandas as pd
import pytest

from panel.run import ResultStore
from panel.topic import Topic, load_topic

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = Path(__file__).resolve().parent / "fixtures"
TOPICS_DIR = ROOT / "config" / "topics"
RESULTS_DIR = ROOT / "results"
LEGACY_TOPIC = "election_2021_legacy"


def load_fixture(name: str) -> Any:
    """A JSON fixture by path under `tests/fixtures/` without extension, e.g. `schemas/election_2021`."""
    return json.loads((FIXTURES / f"{name}.json").read_text(encoding="utf-8"))


def topic_names() -> List[str]:
    return sorted(path.stem for path in TOPICS_DIR.glob("*.json"))


def require(path: Path) -> Path:
    """Skip the calling test when a gitignored input is not on this machine."""
    if not path.exists():
        pytest.skip(f"{path.relative_to(ROOT)} is not on this machine")
    return path


@pytest.fixture(scope="session")
def people() -> pd.DataFrame:
    return pd.read_csv(require(ROOT / "data" / "clean_data.csv"), index_col="id")


@pytest.fixture(scope="session")
def legacy_topic() -> Topic:
    return load_topic(LEGACY_TOPIC, TOPICS_DIR)


@pytest.fixture(scope="session")
def legacy_store() -> ResultStore:
    require(RESULTS_DIR / LEGACY_TOPIC)
    return ResultStore(RESULTS_DIR, LEGACY_TOPIC)
