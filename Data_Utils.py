# Canonical names used across VotingResult and data mappings
# Always import and use these instead of string literals.
import math

VOTED = "Voted"
NOT_VOTED = "Not Voted"

ANO = "ANO 2011"
SPOLU = "Koalice Spolu (ODS, TOP 09, KDU-ČSL)"
PIRSTAN = "Koalice PIRÁTI a STAROSTOVÉ"
KSCM = "Komunistická strana Čech a Moravy (KSČM)"
SPD = "Svoboda a přímá demokracie – Tomio Okamura (SPD)"
CSSD = "Česká strana sociálně demokratická (ČSSD)"
TSS = "Trikolóra, Svobodní a Soukromníci"
PRISAHA = "Přísaha Roberta Šlachty"
JINA_STRANA = "Jiná strana"

# Attendance columns
ATTENDANCE_COLUMNS = [VOTED, NOT_VOTED]

# Explicit party column order used in CSVs/Series
PARTY_COLUMNS_2021 = [
    ANO, SPOLU, PIRSTAN, SPD, PRISAHA, CSSD, KSCM, TSS, JINA_STRANA
]

# All columns for complete schema
ALL_COLUMNS = ATTENDANCE_COLUMNS + PARTY_COLUMNS_2021

# Convenience loader for actual election results CSV
import os
import pandas as pd
from typing import Optional

def load_actual_results() -> pd.DataFrame:
    """
    Load actual election results from CSV.

    Returns:
        pd.DataFrame: DataFrame with columns [region, VOTED, NOT_VOTED, ...party columns...]
    """
    csv_path = os.path.join("data", "election_data.csv")
    return pd.read_csv(csv_path, encoding="utf-8-sig")

# TODO: partie col, weight col/ weights
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
    # if not math.isclose(total,1):
    #     print(f"WARNING total mean before normalization is:{total}")

    normalized_parties = party_sums / total if total > 0 else party_sums * 0
    result_series = pd.concat([attendance_agg, normalized_parties])

    return result_series