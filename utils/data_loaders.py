"""
Data loading and sanitization utilities for LLM panel analysis.

All public loaders implement two options:
- drop_voted: drop the 'Voted' column and weight remaining party columns by the 'Voted' value (per row).
               The 'Not Voted' column (if present) is kept unchanged. Row sums remain ~1 if inputs were valid.
- drop_both:  drop both 'Voted' and 'Not Voted' columns, then re-normalize remaining party columns row-wise to sum to 1.

Additionally, each loader validates row sums and returns a diagnostics report with counts of rows that don't sum to 1
(after the chosen transformation) within a tolerance.

Note: If both drop_voted and drop_both are True, a ValueError is raised.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple
import os
import glob
import warnings

import numpy as np
import pandas as pd

from .constants import VOTED, NOT_VOTED, COUNTRY


@dataclass
class RowSumReport:
    total_rows: int
    n_not_sum_one: int
    tol: float
    note: str = ""

    def as_dict(self) -> Dict[str, object]:
        return {
            "total_rows": self.total_rows,
            "n_not_sum_one": self.n_not_sum_one,
            "tol": self.tol,
            "note": self.note,
        }


def _validate_flags(drop_voted: bool, drop_both: bool) -> None:
    if drop_voted and drop_both:
        raise ValueError("drop_voted and drop_both are mutually exclusive; set only one of them to True.")


def _infer_party_columns(columns: Sequence[str]) -> List[str]:
    """Infer party columns by excluding the two special columns if present."""
    return [c for c in columns if c not in (VOTED, NOT_VOTED)]


def _row_sum_report(df: pd.DataFrame, cols: Optional[Sequence[str]] = None, tol: float = 1e-8, note: str = "") -> RowSumReport:
    if cols is None:
        cols = list(df.columns)
    if len(cols) == 0:
        return RowSumReport(total_rows=len(df), n_not_sum_one=0, tol=tol, note=note)
    sums = df[cols].sum(axis=1).astype(float)
    n_bad = int((np.abs(sums - 1.0) > tol).sum())
    return RowSumReport(total_rows=len(df), n_not_sum_one=n_bad, tol=tol, note=note)


def _drop_unnamed_first_column(df: pd.DataFrame) -> pd.DataFrame:
    if len(df.columns) > 0 and str(df.columns[0]).lower().startswith("unnamed"):
        return df.drop(columns=[df.columns[0]])
    return df


def _ensure_numeric(df: pd.DataFrame, exclude: Optional[Sequence[str]] = None) -> pd.DataFrame:
    exclude = set(exclude or [])
    out = df.copy()
    for c in out.columns:
        if c not in exclude:
            out[c] = pd.to_numeric(out[c], errors="coerce")
    return out


def _apply_drop_logic_rowwise(
    df: pd.DataFrame,
    drop_voted: bool = False,
    drop_both: bool = False,
    tol: float = 1e-8,
) -> Tuple[pd.DataFrame, Dict[str, object]]:
    """Apply drop logic to a row-wise probability table.

    - If drop_both: drop Voted and Not Voted, renormalize remaining party columns to sum to 1.
    - If drop_voted: drop Voted, multiply all party columns by original Voted; keep Not Voted unchanged.
                      This yields unconditional party shares; row sums should remain ~1.
    - Else: leave as is.

    Returns transformed DataFrame and a diagnostics dict.
    """
    _validate_flags(drop_voted, drop_both)

    cols = list(df.columns)
    has_voted = VOTED in cols
    party_cols = _infer_party_columns(cols)

    out = df.copy()

    diagnostics: Dict[str, object] = {}
    diagnostics["pre_row_sum_report"] = _row_sum_report(out, cols=cols, tol=tol, note="before_transform").as_dict()

    if drop_both:
        to_drop = [c for c in (VOTED, NOT_VOTED) if c in out.columns]
        if to_drop:
            out = out.drop(columns=to_drop)
        party_cols2 = list(out.columns)
        sums = out[party_cols2].sum(axis=1)
        nonzero = sums.replace(0, np.nan)
        out[party_cols2] = out[party_cols2].div(nonzero, axis=0).fillna(0.0)
        diagnostics["post_row_sum_report"] = _row_sum_report(out, cols=party_cols2, tol=tol, note="drop_both_after").as_dict()
        diagnostics["mode"] = "drop_both"
        return out, diagnostics

    if drop_voted and has_voted:
        voted_vals = out[VOTED].astype(float)
        if party_cols:
            out[party_cols] = out[party_cols].mul(voted_vals, axis=0)
        out = out.drop(columns=[VOTED])
        cols_after = [c for c in out.columns]
        diagnostics["post_row_sum_report"] = _row_sum_report(out, cols=cols_after, tol=tol, note="drop_voted_after").as_dict()
        diagnostics["mode"] = "drop_voted"
        return out, diagnostics

    diagnostics["post_row_sum_report"] = _row_sum_report(out, cols=cols, tol=tol, note="no_drop_after").as_dict()
    diagnostics["mode"] = "none"
    return out, diagnostics


def load_voting_results(
    csv_path: str,
    drop_voted: bool = False,
    drop_both: bool = False,
    tol: float = 1e-8,
) -> Tuple[pd.DataFrame, Dict[str, object]]:
    """Load a voting results CSV (respondent-level probabilities) and apply drop logic.

    Returns (df, diagnostics).
    """
    _validate_flags(drop_voted, drop_both)

    df = pd.read_csv(csv_path)
    df = _drop_unnamed_first_column(df)
    df = _ensure_numeric(df)

    out, diagnostics = _apply_drop_logic_rowwise(df, drop_voted=drop_voted, drop_both=drop_both, tol=tol)
    diagnostics["source"] = os.path.basename(csv_path)
    return out, diagnostics


def load_voting_results_dict(
    csv_dir: str,
    drop_voted: bool = False,
    drop_both: bool = False,
    tol: float = 1e-8,
) -> Tuple[Dict[str, pd.DataFrame], Dict[str, Dict[str, object]]]:
    """Load all voting_results_*.csv files in a directory into a dict keyed by model name.

    Model name is derived from the filename suffix after the last underscore (e.g., gpt-4o, gpt-4.1, etc.).
    Returns (dict_of_dataframes, dict_of_diagnostics_per_model).
    """
    _validate_flags(drop_voted, drop_both)

    pattern = os.path.join(csv_dir, "voting_results_*.csv")
    files = sorted(glob.glob(pattern))
    if not files:
        warnings.warn(f"No voting results found with pattern: {pattern}")
    results: Dict[str, pd.DataFrame] = {}
    reports: Dict[str, Dict[str, object]] = {}

    for path in files:
        base = os.path.basename(path)
        model_key = base.split("_")[-1].replace(".csv", "")
        df, diag = load_voting_results(path, drop_voted=drop_voted, drop_both=drop_both, tol=tol)
        results[model_key] = df
        reports[model_key] = diag
    return results, reports


def load_claimed_votes(
    csv_dir: str,
    drop_voted: bool = False,
    drop_both: bool = False,
    index_name: str = "party",
    value_name: str = "share",
    tol: float = 1e-8,
) -> Tuple[pd.DataFrame, Dict[str, object]]:
    """Load claimed_votes.csv and apply drop logic.

    Returns a DataFrame with index named `index_name` and a single column named `value_name`.
    The drop logic here is applied to a single-row distribution.
    """
    _validate_flags(drop_voted, drop_both)

    path = os.path.join(csv_dir, "claimed_votes.csv")
    s = pd.read_csv(path, index_col=0).squeeze("columns")
    df = s.to_frame().T
    df.columns = df.columns.astype(str)

    out, diagnostics = _apply_drop_logic_rowwise(df, drop_voted=drop_voted, drop_both=drop_both, tol=tol)

    tidy = out.T
    tidy.index.name = index_name
    tidy.columns = [value_name]

    if drop_both:
        report = _row_sum_report(out, cols=list(out.columns), tol=tol, note="claimed_drop_both").as_dict()
    elif drop_voted:
        report = _row_sum_report(out, cols=list(out.columns), tol=tol, note="claimed_drop_voted").as_dict()
    else:
        report = _row_sum_report(out, cols=list(out.columns), tol=tol, note="claimed_no_drop").as_dict()
    diagnostics["post_tidy_report"] = report
    diagnostics["source"] = os.path.basename(path)

    return tidy, diagnostics


def load_election_averages(
    csv_dir: str,
    region: str = COUNTRY,
    drop_voted: bool = False,
    drop_both: bool = False,
    index_name: str = "party",
    value_name: str = "average",
    tol: float = 1e-8,
) -> Tuple[pd.DataFrame, Dict[str, object]]:
    """Load election_data.csv, select a region row, and return a tidy one-column DataFrame.

    - If drop_both: drop Voted + Not Voted and renormalize parties to sum 1.
    - If drop_voted: drop Voted, multiply party columns by Voted; keep Not Voted unchanged.
    """
    _validate_flags(drop_voted, drop_both)

    path = os.path.join(csv_dir, "election_data.csv")
    df = pd.read_csv(path)

    numeric_cols = [c for c in df.columns if c != "region"]
    df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors="coerce")

    row = df[df["region"] == region]
    if row.empty:
        raise ValueError(f"Region '{region}' not found in election_data.csv")

    vals = row.drop(columns=["region"]).reset_index(drop=True)

    out, diagnostics = _apply_drop_logic_rowwise(vals, drop_voted=drop_voted, drop_both=drop_both, tol=tol)

    tidy = out.T
    tidy.index.name = index_name
    tidy.columns = [value_name]

    if drop_both:
        report = _row_sum_report(out, cols=list(out.columns), tol=tol, note="election_drop_both").as_dict()
    elif drop_voted:
        report = _row_sum_report(out, cols=list(out.columns), tol=tol, note="election_drop_voted").as_dict()
    else:
        report = _row_sum_report(out, cols=list(out.columns), tol=tol, note="election_no_drop").as_dict()
    diagnostics["post_tidy_report"] = report
    diagnostics["source"] = os.path.basename(path)

    return tidy, diagnostics


def average_distribution(df: pd.DataFrame) -> pd.Series:
    """Compute the column-wise mean distribution for a respondent-level DataFrame.

    Returns a Series indexed by category name.
    """
    return df.mean(axis=0)


def build_voting_averages_dict(
    voting_results: Dict[str, pd.DataFrame],
) -> Dict[str, pd.Series]:
    """Take a dict of respondent-level DataFrames and return their average distributions as Series."""
    return {model: average_distribution(df) for model, df in voting_results.items()}
