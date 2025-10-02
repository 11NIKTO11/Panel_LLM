import math
from pandas import Series  # for type hints only
from pydantic import BaseModel, Field,  field_validator
from typing import List, Literal, TYPE_CHECKING
from collections import Counter

class PartyProbability(BaseModel):
    """
    Strukturovaná reprezentace pravděpodobnosti hlasování pro konkrétní stranu.
    """
    name: Literal[
        "ANO 2011",
        "Koalice Spolu (ODS, TOP 09, KDU-ČSL)",
        "Koalice PIRÁTI a STAROSTOVÉ",
        "Komunistická strana Čech a Moravy (KSČM)",
        "Svoboda a přímá demokracie – Tomio Okamura (SPD)",
        "Česká strana sociálně demokratická (ČSSD)",
        "Trikolóra, Svobodní a Soukromníci",
        "Přísaha Roberta Šlachty",
        "Jiná strana"
    ] = Field(description="Název politické strany.")
    probability: float = Field(
        ge=0, le=1, description="Pravděpodobnost, že respondent hlasoval pro tuto stranu."
    )

class VotedProbability(BaseModel):
    """
    Pravděpodobnost, zda respondent volil, nebo nevolil.
    """
    voted: float = Field(ge=0, le=1, description="Pravděpodobnost, že respondent volil.")
    not_voted: float = Field(ge=0, le=1, description="Pravděpodobnost, že respondent nevolil.")

    # @field_validator('*', pre=True, always=True)
    # def check_sum(cls, v, values):
    #     # Tento validátor zajistí, že součet se bude blížit 1.0
    #     # Můžete si pohrát s tolerancí (např. 0.01)
    #     if 'voted' in values and 'not_voted' in values:
    #         if abs(values['voted'] + values['not_voted'] - 1.0) > 0.001:
    #             raise ValueError("Součet pravděpodobností pro 'voted' a 'not_voted' se musí rovnat 1.0")
    #     return v

class VotingResult(BaseModel):
    """
    Strukturovaný výstup pro volební chování respondenta ve volbách do poslanecké sněmovny 2021.
    """
    # Změna zde: použití nové třídy místo složitého Tuple
    voted_or_not: VotedProbability = Field(
        description="Pravděpodobnost, zda respondent volil, nebo nevolil. Součet musí být 1.0."
    )
    parties: List[PartyProbability] = Field(
        description="Seznam možných stran s pravděpodobností volby. Součet pravděpodobností musí být 1.0."
    )

    def to_series(self) -> 'Series':
        """
        Transform this VotingResult into a single pandas.Series (one row) with a flat schema.

        Columns produced:
          - voted_or_not.voted
          - voted_or_not.not_voted
          - party.<party_name>

        If multiple PartyProbability entries share the same name, later ones overwrite earlier ones.
        Missing parties will simply be absent (NaN when combined into a DataFrame).
        """
        try:
            import pandas as pd  # local import to avoid hard dependency for non-analytics workflows
        except Exception as e:
            raise RuntimeError("pandas is required to create a Series from VotingResult") from e

        row = {
            "voted": self.voted_or_not.voted,
            "not_voted": self.voted_or_not.not_voted,
        }
        for p in (self.parties or []):
            # Create a stable, readable column name for the party probability
            col = p.name
            row[col] = p.probability
        return pd.Series(row)

def evaluate_result(respondent_id:int, res:'VotingResult', tol:float = 1e-2):
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
    names =  [p.name for p in res.parties]
    counts = Counter(names)
    dups = [n for n, c in counts.items() if n is not None and c > 1]
    if dups:
        issues['dup_parties'] = (respondent_id, dups)

    # Check that party probabilities sum to 1 (if any parties present)
    probs = [p.probability for p in res.parties]
    if probs and v > 0.0:  # only check if list is non-empty
        s_parties = sum(probs)
        if not math.isclose(s_parties, 1.0, rel_tol=0, abs_tol=tol):
            issues['bad_party_probs_sum'] = (respondent_id, s_parties, v)

    return issues

def evaluate_voting_results(voting_results, tol:float = 1e-2):
    bad_voted_sum, bad_party_probs_sum, duplicate_parties = [],[],[]
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