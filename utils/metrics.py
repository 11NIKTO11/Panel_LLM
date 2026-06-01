import math
from .voting import VotingProbabilities

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
    for rid, (prompt, result) in voting_results.items():
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
