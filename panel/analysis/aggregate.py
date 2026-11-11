"""
Aggregate per-subject rows into one distribution - predicted from a results frame, claimed from
a profile's answers.
"""
import pandas as pd

from panel.topic.normalize import normalize_groups


def aggregate_results_from_df(df: pd.DataFrame, topic, weighted: bool = True, normalized: bool = True) -> pd.Series:
    """
    Aggregate per-respondent results into one Series: attendance, then the options.

    Args:
        df: one row per respondent, carrying the topic's columns.
        topic: `panel.topic.Topic` naming the option columns and the attendance pair. A
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


def claimed_distribution(frame: pd.DataFrame, claims, topic, count_excluded_as_abstained: bool = False) -> pd.Series:
    """
    What subjects said they chose - the *claimed* reference series, in the topic's columns.

    Args:
        frame: profile rows carrying the claims' answer column.
        claims: `panel.dataset.Claims` sorting the dataset's answer strings into option ids,
            abstained and excluded.
        topic: `panel.topic.Topic` whose columns key the series.
        count_excluded_as_abstained: if True, non-substantive answers stay in the turnout
            denominator as non-voters instead of dropping out of it.

    Returns:
        pd.Series: attendance (if the topic has it), then each option's share of the
            substantive choices, summing to 1.

    Raises:
        ValueError: if an answer string is not in the claims map, or the map names an option
            id the topic does not have.
    """
    answers = frame[claims.column].astype(str)
    unknown = sorted(set(answers) - set(claims.answers))
    if unknown:
        raise ValueError(f"claims map {claims.dataset}/{claims.topic} does not cover answers {unknown}")
    foreign = sorted(set(claims.options.values()) - set(topic.options))
    if foreign:
        raise ValueError(f"claims map {claims.dataset}/{claims.topic} names options unknown to topic '{topic.name}': {foreign}")

    substantive = answers[~answers.isin(claims.excluded)]
    chosen = substantive[~substantive.isin(claims.abstained)]
    denominator = len(answers) if count_excluded_as_abstained else len(substantive)
    turnout = len(chosen) / denominator if denominator else 0.0
    counts = chosen.map(claims.options).value_counts()

    result = {}
    if topic.turnout:
        yes, no = topic.turnout
        result[yes], result[no] = turnout, 1 - turnout
    for option in topic.options:
        result[option] = counts.get(option, 0) / len(chosen) if len(chosen) else 0.0
    return pd.Series(result, dtype=float)
