"""
Several runs of one topic set beside each other: the runs read back (`loaded`) and one module
per comparison.

Every comparison is a function `(runs, out) -> (table, figures)`, like an analysis of one
run, and builds on those analyses; `COMPARISONS` names them in the order a full pass runs them.
"""
from .areas import areas
from .compass import compass
from .distances import distances
from .effects import effects
from .loaded import RunSet, describe, load_runs
from .sets import Comparison, load_comparisons
from .shares import shares
from .spreads import spreads

COMPARISONS = {"shares": shares, "distances": distances, "areas": areas, "spread": spreads, "effects": effects, "compass": compass}
