"""
Per-subject rows into one distribution: predicted from a run's answers, claimed from the
dataset's own answer to the topic. And a run's calls into one row per subject.

An outcome is one of the topic's options or, on a topic with turnout, not taking part; an
answer as outcomes is each option's share times the probability of taking part, and the
probability of not taking part, summing to 1.
"""
import pandas as pd


def subject_answers(normalized: pd.DataFrame, topic) -> pd.DataFrame:
    """
    One row per subject with the topic's columns: the mean of the subject's answers over
    its repeats, without the subjects that have no preference in any of them.

    Args:
        normalized: one row per call, indexed by subject, each column group summing to 1;
            a call whose options summed to zero is empty there.
        topic: `panel.topic.Topic`.
    """
    return normalized[topic.columns].groupby(level=0, sort=False).mean().dropna(subset=list(topic.options))


def predicted_distribution(normalized: pd.DataFrame, topic, weighted: bool = True) -> pd.Series:
    """
    What a run predicts for a group of subjects: turnout, then each option's share.

    Args:
        normalized: one row per call with the topic's columns, each group summing to 1; a
            row whose options summed to zero is empty there and counts for no option.
        topic: `panel.topic.Topic`.
        weighted: weight each row's option shares by its probability of voting, so a likely
            abstainer counts for less. Nothing to weight by on a topic without turnout.

    Returns:
        The mean of each turnout column (if the topic has turnout), then the options,
        summing to 1.
    """
    options = list(topic.options)
    turnout = pd.Series(dtype=float)
    shares = normalized[options].fillna(0.0)
    if topic.turnout:
        turnout = normalized[list(topic.turnout)].mean()
        if weighted:
            shares = shares.mul(normalized[topic.turnout[0]], axis=0)
    totals = shares.sum()
    return pd.concat([turnout, totals / totals.sum() if totals.sum() > 0 else totals])


def claimed_distribution(subjects: pd.DataFrame, claims, topic, count_excluded_as_abstained: bool = False) -> pd.Series:
    """
    What a group of subjects said they chose, in the topic's columns.

    Args:
        subjects: values rows carrying the claims' answer column.
        claims: `panel.dataset.Claims`: answers sorted into option ids, abstained, excluded.
        topic: `panel.topic.Topic`.
        count_excluded_as_abstained: keep answers that say nothing ("don't know") in the
            turnout denominator as non-voters instead of dropping them.

    Returns:
        Turnout (if the topic has it), then each option's share of those who named one.

    Raises:
        ValueError: on an answer the claims map does not sort, or an option id in the map
            that the topic does not have.
    """
    answers = subjects[claims.column]
    unknown = sorted(set(answers) - set(claims.options) - set(claims.abstained) - set(claims.excluded))
    if unknown:
        raise ValueError(f"claims map {claims.dataset}/{claims.topic} does not sort the answers {unknown}")
    foreign = sorted(set(claims.options.values()) - set(topic.options))
    if foreign:
        raise ValueError(f"claims map {claims.dataset}/{claims.topic} names options '{topic.name}' does not have: {foreign}")

    said = answers[~answers.isin(claims.excluded)]
    chose = said[~said.isin(claims.abstained)]
    asked = len(answers) if count_excluded_as_abstained else len(said)
    counts = chose.map(claims.options).value_counts()

    result = {}
    if topic.turnout:
        voted = len(chose) / asked if asked else 0.0
        result[topic.turnout[0]], result[topic.turnout[1]] = voted, 1 - voted
    for option in topic.options:
        result[option] = counts.get(option, 0) / len(chose) if len(chose) else 0.0
    return pd.Series(result, dtype=float)


def outcome_rows(normalized: pd.DataFrame, topic) -> pd.DataFrame:
    """
    One row per call, a column per outcome, summing to 1; a call is left out if it says no
    option while it might take part.

    Args:
        normalized: one row per call, indexed by subject, each column group summing to 1.
        topic: `panel.topic.Topic`.
    """
    options = list(topic.options)
    if not topic.turnout:
        return normalized[options].dropna()
    taking, staying = topic.turnout
    defined = normalized[options].notna().all(axis=1) | (normalized[taking] == 0)
    joint = normalized[options].fillna(0.0).mul(normalized[taking], axis=0).assign(**{staying: normalized[staying]})
    return joint[defined].dropna()


def outcome_shares(normalized: pd.DataFrame, topic) -> pd.DataFrame:
    """One row per subject, a column per outcome: `outcome_rows` averaged over the subject's repeats."""
    return outcome_rows(normalized, topic).groupby(level=0, sort=False).mean()
