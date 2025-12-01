from collections import Counter
import math
import pandas as pd

from .constants import VOTED, NOT_VOTED, PARTY_COLUMNS_2021
from .respondent_description import is_non_substantive_responses
from .voting import VotingProbabilities


def get_actual_results(respondents: pd.DataFrame, count_non_substantive_as_not_voted: bool = False) -> pd.Series:
    """
    Aggregate actual voting results from survey respondents into a Series.

    Args:
        respondents: DataFrame with 'voted_party' column containing survey responses
        count_non_substantive_as_not_voted: If True, count "Nevím"/"Nechci uvést" as "Not Voted"

    Returns:
        pd.Series: Aggregated results with attendance and party probabilities
            - VOTED: probability of voting
            - NOT_VOTED: probability of not voting
            - Party columns: vote share for each party (normalized to sum to 1)
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
    result_dict = {
        VOTED: voted,
        NOT_VOTED: 1 - voted
    }

    # Add all parties with their probabilities (0 if not in data)
    for party in PARTY_COLUMNS_2021:
        result_dict[party] = (party_counts.get(party, 0) / total_party_votes) if total_party_votes > 0 else 0.0

    return pd.Series(result_dict)


def evaluate_result(respondent_id: int, res: 'VotingProbabilities', tol: float = 1e-2):
    """
    Pure evaluation: returns a dict with detected issues; does not mutate outer scope.
    Returns keys:
      - bad_voted_sum: tuple (rid, voted, not_voted, sum)
      - dup_parties: tuple (rid, [duplicate_names])
      - bad_party_probs_sum: tuple (rid, sum, n_parties)
    Missing keys mean no issue of that type.
    """
    issues = {}
    # Check voted + not_voted ~= 1
    v = res.voted_or_not.voted
    nv = res.voted_or_not.not_voted
    s_voted = v + nv
    if not math.isclose(s_voted, 1.0, rel_tol=0, abs_tol=tol):
        issues['bad_voted_sum'] = (respondent_id, v, nv, s_voted)

    # Check duplicate party names
    names = [p.name for p in res.parties]
    from collections import Counter as _Counter
    counts = _Counter(names)
    dups = [n for n, c in counts.items() if n is not None and c > 1]
    if dups:
        issues['dup_parties'] = (respondent_id, dups)

    # Check that party probabilities sum to 1 (if any parties present and respondent voted)
    probs = [p.probability for p in res.parties]
    if probs and v > 0.0:  # only check if list is non-empty
        s_parties = sum(probs)
        if not math.isclose(s_parties, 1.0, rel_tol=0, abs_tol=tol):
            issues['bad_party_probs_sum'] = (respondent_id, s_parties, v)

    return issues


def evaluate_voting_results(voting_results, tol: float = 1e-2):
    bad_voted_sum, bad_party_probs_sum, duplicate_parties = [], [], []
    for rid, result in voting_results.items():
        try:
            issues = evaluate_result(rid, result, tol)
            if 'bad_voted_sum' in issues:
                bad_voted_sum.append(issues['bad_voted_sum'])
            if 'bad_party_probs_sum' in issues:
                bad_party_probs_sum.append(issues['bad_party_probs_sum'])
            if 'dup_parties' in issues:
                duplicate_parties.append(issues['dup_parties'])
        except Exception as e:
            print(f"Error checking respondent {rid}: {e}")
    return bad_voted_sum, bad_party_probs_sum, duplicate_parties
