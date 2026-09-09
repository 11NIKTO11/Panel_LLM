"""
MANOVA of per-subject distributions on profile columns, via statsmodels.

One results frame at a time: the caller joins several runs into a dict of reports and
`effects` tabulates them. Dependent columns are the option probabilities, ILR-transformed by
default so the compositional constraint does not enter the model.
"""
import re
from typing import Dict, Sequence

import pandas as pd
from statsmodels.multivariate.manova import MANOVA

from .compositional import ilr_frame


def _formula_name(name: str) -> str:
    """A column name statsmodels' formula parser accepts: anything not alphanumeric or `_` becomes `_`."""
    return re.sub(r"\W+", "_", name.strip())


def manova(profile: pd.DataFrame, results: pd.DataFrame, predictors: Sequence[str], columns: Sequence[str], ilr: bool = True):
    """
    Fit `columns ~ predictors` over the profile joined with the results and return the multivariate tests.

    Args:
        profile: subjects with the predictor columns, indexed by id.
        results: one row per subject with the dependent columns, indexed by id.
        predictors: profile columns; object-typed ones become categorical.
        columns: dependent columns of `results`.
        ilr: model the ILR coordinates of `columns` instead of the raw probabilities.

    Returns:
        statsmodels `MultivariateTestResults`.

    Raises:
        ValueError: if no rows are left after dropping incomplete ones.
    """
    merged = profile.join(results, how="inner").copy()
    dependent = list(columns)
    safe = {column: _formula_name(column) for column in dependent}
    for original, alias in safe.items():
        merged[alias] = pd.to_numeric(merged[original], errors="coerce")
    for predictor in predictors:
        if merged[predictor].dtype == "object":
            merged[predictor] = merged[predictor].astype("category")
    merged = merged.dropna(subset=list(safe.values()) + list(predictors))
    if merged.empty:
        raise ValueError("no complete rows to fit")

    left = list(safe.values())
    if ilr:
        merged = ilr_frame(merged, left)
        left = [f"ilr_{i}" for i in range(1, len(left))]
    formula = f"{' + '.join(left)} ~ {' + '.join(predictors)}"
    return MANOVA.from_formula(formula, data=merged).mv_test()


def effects(reports: Dict[str, object], stats: Sequence[str] = ("Wilks' lambda", "Pillai's trace")) -> pd.DataFrame:
    """
    One row per (report, predictor) with the chosen multivariate statistics, the intercept skipped.

    Args:
        reports: name -> `MultivariateTestResults`, as `manova` returns.
        stats: which test statistics to extract.
    """
    records = []
    for name, tests in reports.items():
        for term, result in tests.results.items():
            if term == "Intercept":
                continue
            table = result["stat"]
            row = {"model": name, "predictor": term}
            for stat in stats:
                if stat in table.index:
                    row[stat] = table.loc[stat, "Value"]
            records.append(row)
    summary = pd.DataFrame.from_records(records)
    return summary.set_index(["model", "predictor"]).sort_index() if not summary.empty else summary
