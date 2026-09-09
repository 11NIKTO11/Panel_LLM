"""
What respondents said they voted - the *claimed* reference series.

One of the three reference points a run is read against (predicted / claimed / actual);
this module owns the claimed one. The actual result comes from the topic's
`reference_results` CSV, the predicted one from the run's own output.
"""
from collections import Counter

import pandas as pd

from .clean import is_non_substantive_responses


def get_claimed_results(respondents: pd.DataFrame, topic, count_non_substantive_as_not_voted: bool = False) -> pd.Series:
    """
    Aggregate what respondents said they voted - the *claimed* reference series.

    Args:
        respondents: DataFrame with a 'voted_party' column of survey responses
        topic: `panel.topic.Topic` naming the option columns and the attendance pair
        count_non_substantive_as_not_voted: If True, count "Nevím"/"Nechci uvést" as not voted

    Returns:
        pd.Series: attendance (if the topic has it), then each option's share of the
            claimed votes, normalized to sum to 1
    """
    # Filter out totally non-substantive answers unless counting them as not voted in the denominator.
    votes_all = respondents["voted_party"].astype(str).tolist()
    votes_subst = [vote for vote in votes_all if not is_non_substantive_responses(vote)]
    party_votes = [vote for vote in votes_subst if "Nebyl" not in vote]

    denom = len(respondents) if count_non_substantive_as_not_voted else len(votes_subst)
    voted = (len(party_votes) / denom) if denom > 0 else 0.0

    # Count party votes
    party_counts = Counter(party_votes)
    total_party_votes = len(party_votes)

    # Build result series
    result_dict = {}
    if topic.turnout:
        attended, abstained = topic.turnout
        result_dict[attended] = voted
        result_dict[abstained] = 1 - voted

    # Add all options with their share (0 if nobody claimed it)
    for party in topic.options:
        if total_party_votes > 0:
            result_dict[party] = party_counts.get(party, 0) / total_party_votes
        else:
            result_dict[party] = 0.0

    return pd.Series(result_dict)
