"""
Runs held against a model trained on the subjects' own answers: what both stand on
(`ground`) and one module per benchmark.

Every benchmark is a function `(ground, out) -> (table, figures)`; `BENCHMARKS` names them in
the order a full pass runs them. The comparison is subject by subject, where the analyses of
`panel.analysis` and `panel.analysis.comparison` compare whole distributions.
"""
from .agreement import agreement
from .calibration import calibration
from .ground import Ground, describe, load_ground
from .scores import knowledge, scores

BENCHMARKS = {"scores": scores, "knowledge": knowledge, "agreement": agreement, "calibration": calibration}
