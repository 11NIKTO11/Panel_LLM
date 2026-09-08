from typing import Sequence

import numpy as np
import pandas as pd

def generate_prob_vectors_df(n, m):
    """Generate n random probability vectors of length m (rows sum to 1), e.g. as mock model output."""
    random_vectors = np.random.rand(n, m)
    probability_vectors = random_vectors / random_vectors.sum(axis=1, keepdims=True)
    df = pd.DataFrame(probability_vectors)
    return df

def explore_dataset_values(data:pd.DataFrame, columns:list[str]):
    print("--- Starting Data Exploration: Unique Column Values ---\n")

    for column_name in columns:
        # First, check if the column actually exists in the DataFrame
        if column_name in data.columns:
            print(f"--- Column: '{column_name}' ---")
            try:
                # Get unique values and convert to a list for clean printing
                unique_values = data[column_name].unique().tolist()
                unique_values.sort()
                print(unique_values)
            except Exception as e:
                print(f"Could not retrieve unique values. Error: {e}")
            print("\n") # Add a newline for better readability
        else:
            # Print a warning if a column is not found
            print(f"--- Column: '{column_name}' ---")
            print("WARNING: This column was not found in the DataFrame.\n")

    print("--- Data Exploration Complete ---")

def aggregate_results_from_df(df: pd.DataFrame, topic, weighted: bool = True, normalized: bool = True) -> pd.Series:
    """
    Aggregate per-respondent results into one Series: attendance, then the options.

    Args:
        df: one row per respondent, carrying the topic's columns.
        topic: `utils.topic.Topic` naming the option columns and the attendance pair. A
            topic without an attendance dimension yields no attendance rows, and
            `weighted` then has nothing to weight by.
        weighted: If True, weight option probabilities by the attendance probability, so a
            respondent who likely abstained contributes less.
        normalized: If True, normalize option probabilities within each row before aggregation.

    Returns:
        pd.Series: attendance means (if the topic has them), then the options, normalized
            to sum to 1 across the options.
    """
    party_cols = list(topic.options)

    attendance_agg = pd.Series(dtype=float)
    weights = 1.0
    if topic.turnout:
        voted, not_voted = topic.turnout
        attendance_agg = pd.Series({
            voted: float(df[voted].mean()),
            not_voted: float(df[not_voted].mean()),
        })
        if weighted:
            weights = df[voted]

    # Aggregate option probabilities
    denom = df[party_cols].sum(axis=1) if normalized else 1.0
    denom = denom.replace(0, 1) if normalized else denom

    # Apply normalization and weighting
    weighted_parties = (df[party_cols].div(denom, axis=0).fillna(0)).multiply(weights, axis=0)

    # Mean across all respondents
    party_sums = weighted_parties.mean(axis=0)

    # Normalize party distribution to sum to 1 across all parties
    total = party_sums.sum()
    normalized_parties = party_sums / total if total > 0 else party_sums * 0
    result_series = pd.concat([attendance_agg, normalized_parties])

    return result_series

def normalize_groups(df: pd.DataFrame, groups: Sequence[Sequence[str]]) -> pd.DataFrame:
    """
    Rescale each group of columns to sum to 1 per row.

    Args:
        df: wide results, one row per respondent.
        groups: column groups, e.g. `topic.normalize_groups`.
    """
    out = df.copy()
    for columns in groups:
        columns = list(columns)
        out[columns] = out[columns].div(out[columns].sum(axis=1), axis=0)
    return out
