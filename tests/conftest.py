"""
Shared fixtures: where things are, and loaders for the phase 0 baselines under `fixtures/`.

An input that is not in git (`data/profiles/sod.csv`, `results/`) makes a test skip naming
the path it needed, rather than fail.
"""
import json
from pathlib import Path
from typing import Any, List

import pandas as pd
import pytest

from panel import paths
from panel.dataset import load_dataset, profile_path, read_profile
from panel.run import ResultStore
from panel.topic import Topic, load_topic

ROOT = paths.ROOT
FIXTURES = Path(__file__).resolve().parent / "fixtures"
TOPICS_DIR = paths.TOPICS_DIR
RESULTS_DIR = paths.RESULTS_DIR
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
    dataset = load_dataset("sod")
    require(profile_path(dataset))
    return read_profile(dataset)


@pytest.fixture(scope="session")
def legacy_topic() -> Topic:
    return load_topic(LEGACY_TOPIC, TOPICS_DIR)


@pytest.fixture(scope="session")
def legacy_store() -> ResultStore:
    require(RESULTS_DIR / LEGACY_TOPIC)
    return ResultStore(RESULTS_DIR, LEGACY_TOPIC)
