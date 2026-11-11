"""
Analysis of saved runs: a run read back with what it is compared against (`saved`), where its
tables and figures go (`output`), and one module per analysis.

Every analysis is a function `(run, out) -> (table, figures)` that raises `NotApplicable` for a
run it has nothing to work on; `ANALYSES` names them in the order a full pass runs them.
"""
from .answers import answers
from .area import area
from .compass import compass
from .correlation import correlation
from .country import country
from .distance import distance
from .effects import effects
from .groups import groups
from .output import Output
from .spread import spread
from .saved import NotApplicable, SavedRun, describe, find_stem, load_run

ANALYSES = {"answers": answers, "country": country, "distance": distance, "area": area, "spread": spread, "correlation": correlation, "groups": groups, "effects": effects, "compass": compass}
