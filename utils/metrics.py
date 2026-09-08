import math
from collections import Counter


def evaluate_result(respondent_id: int, res, tol: float = 1e-2):
    """
    Pure evaluation: returns a dict with detected issues; does not mutate outer scope.

    Works for any topic's response model - the attendance check runs only where the
    topic has an attendance dimension, and the option checks only where the model
    carries a list of options.

    Returns keys:
      - bad_voted_sum: tuple (rid, voted, not_voted, sum)
      - dup_parties: tuple (rid, [duplicate_names])
      - bad_party_probs_sum: tuple (rid, sum, n_parties)
    Missing keys mean no issue of that type.
    """
    issues = {}
    voted = None

    if getattr(type(res), "TURNOUT", None):
        voted = res.voted_or_not.voted
        not_voted = res.voted_or_not.not_voted
        total = voted + not_voted
        if not math.isclose(total, 1.0, rel_tol=0, abs_tol=tol):
            issues['bad_voted_sum'] = (respondent_id, voted, not_voted, total)

    options = getattr(res, "parties", None)
    if options is None:
        return issues

    counts = Counter(option.name for option in options)
    dups = [name for name, count in counts.items() if name is not None and count > 1]
    if dups:
        issues['dup_parties'] = (respondent_id, dups)

    # A respondent who certainly abstained is allowed an empty distribution.
    probs = [option.probability for option in options]
    if probs and (voted is None or voted > 0.0):
        total = sum(probs)
        if not math.isclose(total, 1.0, rel_tol=0, abs_tol=tol):
            issues['bad_party_probs_sum'] = (respondent_id, total, voted)

    return issues


def evaluate_voting_results(voting_results, tol: float = 1e-2):
    bad_voted_sum, bad_party_probs_sum, duplicate_parties = [], [], []
    for rid, (prompt, result) in voting_results.items():
        issues = evaluate_result(rid, result, tol)
        if 'bad_voted_sum' in issues:
            bad_voted_sum.append(issues['bad_voted_sum'])
        if 'bad_party_probs_sum' in issues:
            bad_party_probs_sum.append(issues['bad_party_probs_sum'])
        if 'dup_parties' in issues:
            duplicate_parties.append(issues['dup_parties'])
    return bad_voted_sum, bad_party_probs_sum, duplicate_parties
