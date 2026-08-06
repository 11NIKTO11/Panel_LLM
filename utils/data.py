import numpy as np
import pandas as pd
from .constants import PARTY_COLUMNS_2021, VOTED, NOT_VOTED

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

def aggregate_results_from_df(df: pd.DataFrame, weighted: bool = True, normalized: bool = True) -> pd.Series:
    """
    Aggregate voting results from a DataFrame into a single Series with attendance and party probabilities.

    Args:
        df: DataFrame with columns [VOTED, NOT_VOTED, ...party columns...].
            Each row represents a respondent's voting probabilities.
        weighted: If True, weight party probabilities by VOTED probability (respondents who didn't vote contribute less).
        normalized: If True, normalize party probabilities within each row before aggregation.

    Returns:
        pd.Series: Aggregated results with:
            - VOTED: mean probability of voting
            - NOT_VOTED: mean probability of not voting
            - Party columns: aggregated party probabilities (normalized to sum to 1 across all parties)
    """
    party_cols = PARTY_COLUMNS_2021

    # Aggregate attendance probabilities (simple mean)
    attendance_agg = pd.Series({
        VOTED: float(df[VOTED].mean()),
        NOT_VOTED: float(df[NOT_VOTED].mean()),
    })

    # Aggregate party probabilities
    denom = df[party_cols].sum(axis=1) if normalized else 1.0
    denom = denom.replace(0, 1)
    weights = df[VOTED] if weighted else 1.0

    # Apply normalization and weighting
    weighted_parties = (df[party_cols].div(denom, axis=0).fillna(0)).multiply(weights, axis=0)

    # Mean across all respondents
    party_sums = weighted_parties.mean(axis=0)

    # Normalize party distribution to sum to 1 across all parties
    total = party_sums.sum()
    normalized_parties = party_sums / total if total > 0 else party_sums * 0
    result_series = pd.concat([attendance_agg, normalized_parties])

    return result_series
