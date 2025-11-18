import math

import pandas as pd
from pandas import Series  # for type hints only
from pydantic import BaseModel, Field,  field_validator
from typing import List, Literal, TYPE_CHECKING
from collections import Counter

import Data_Utils
from Data_Utils import PARTY_COLUMNS_2021

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

class PartyProbability(BaseModel):
    """
    Strukturovaná reprezentace pravděpodobnosti hlasování pro konkrétní stranu.
    """
    name: Literal[*PARTY_COLUMNS_2021] = Field(description="Název politické strany.")
    probability: float = Field(
        ge=0, le=1, description="Pravděpodobnost, že respondent hlasoval pro tuto stranu."
    )

class VotingProbabilities(BaseModel):
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
            Data_Utils.VOTED: self.voted_or_not.voted,
            Data_Utils.NOT_VOTED: self.voted_or_not.not_voted,
        }
        for p in (self.parties or []):
            row[p.name] = p.probability

        for col in PARTY_COLUMNS_2021:
            if col not in row.keys():
                row[col] = 0.0

        return pd.Series(row)

class VotingPick(BaseModel):
    """
    Strukturovaný výstup pro volební chování respondenta ve volbách do poslanecké sněmovny 2021.
    """
    # Změna zde: použití nové třídy místo složitého Tuple
    voted_or_not: VotedProbability = Field(
        description="Pravděpodobnost, zda respondent volil, nebo nevolil. Součet musí být 1.0."
    )
    choice: Literal[*PARTY_COLUMNS_2021] = Field(
        description=(
            "Zhodnoť poskytnutý sociodemografický profil respondenta a na jeho základě "
            "urči nejpravděpodobnější politickou stranu, které odevzdal svůj hlas. "
            "Tato volba je relevantní pouze v případě, že respondent skutečně volil. "
            "Pokud profil neukazuje na jednoznačnou preferenci, vyber možnost, která je statisticky nejpravděpodobnější "
            "pro daný typ respondenta. Tvá volba musí přesně odpovídat jedné z povolených možností."
        ))

    def to_series(self) -> 'Series':
        row = ({
            Data_Utils.VOTED: self.voted_or_not.voted,
            Data_Utils.NOT_VOTED: self.voted_or_not.not_voted,
        })

        for col in PARTY_COLUMNS_2021:
            row[col] = 1 if self.choice == col else 0
        return pd.Series(row)

def evaluate_result(respondent_id:int, res: 'VotingProbabilities', tol:float = 1e-2):
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