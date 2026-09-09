"""
Where the repo's data lives, anchored on this package rather than on the working directory.

A Jupyter kernel starts in the folder holding its notebook, so a cwd-relative "data/..."
means different things in different notebooks; anchoring on `__file__` makes cwd matter for
imports only. Run records store no filesystem paths, so these never reach saved results.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = ROOT / "config"
TOPICS_DIR = CONFIG_DIR / "topics"
DESCRIPTIONS_DIR = CONFIG_DIR / "descriptions"
DATASETS_DIR = ROOT / "datasets"
DATA_DIR = ROOT / "data"
PROFILES_DIR = DATA_DIR / "profiles"
PROMPTS_DIR = DATA_DIR / "prompts"
RESULTS_DIR = ROOT / "results"
GRAPHS_DIR = ROOT / "graphs"
