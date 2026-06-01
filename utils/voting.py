import pandas as pd
from pandas import Series
from pydantic import BaseModel, Field
from typing import List, Literal
from collections import Counter

from .constants import VOTED, NOT_VOTED, PARTY_COLUMNS_2021

PARTY_NAMES = PARTY_COLUMNS_2021

class VotedProbability(BaseModel):
    """
    Pravděpodobnost, zda respondent volil, nebo nevolil.
    """
    voted: float = Field(ge=0, le=1, description="Pravděpodobnost, že respondent volil.")
    not_voted: float = Field(ge=0, le=1, description="Pravděpodobnost, že respondent nevolil.")

class PartyProbability(BaseModel):
    """
    Strukturovaná reprezentace pravděpodobnosti hlasování pro konkrétní stranu.
    """
    name: Literal[*PARTY_NAMES] = Field(description="Název politické strany.")
    probability: float = Field(
        ge=0, le=1, description="Pravděpodobnost, že respondent hlasoval pro tuto stranu."
    )

class VotingProbabilities(BaseModel):
    """
    Strukturovaný výstup pro volební chování respondenta ve volbách do poslanecké sněmovny 2021.
    """
    voted_or_not: VotedProbability = Field(
        description="Pravděpodobnost, zda respondent volil, nebo nevolil. Součet musí být 1.0."
    )
    parties: List[PartyProbability] = Field(
        description="Seznam možných stran s pravděpodobností volby. Součet pravděpodobností musí být 1.0."
    )

    def to_series(self) -> 'Series':
        row = {
            VOTED: self.voted_or_not.voted,
            NOT_VOTED: self.voted_or_not.not_voted,
        }
        for p in (self.parties or []):
            row[p.name] = p.probability

        for col in PARTY_NAMES:
            if col not in row.keys():
                row[col] = 0.0

        return pd.Series(row)

class VotingPick(BaseModel):
    """Zjednodušený výstup: vybraná strana + pravděpodobnost volby vs. nevolby."""
    voted_or_not: VotedProbability = Field(
        description="Pravděpodobnost, zda respondent volil, nebo nevolil. Součet musí být 1.0."
    )
    choice: Literal[*PARTY_NAMES] = Field(
        description=(
            "Zhodnoť poskytnutý sociodemografický profil respondenta a na jeho základě urči nejpravděpodobnější politickou stranu, které odevzdal svůj hlas."
        )
    )

    def to_series(self) -> 'Series':
        row = {
            VOTED: self.voted_or_not.voted,
            NOT_VOTED: self.voted_or_not.not_voted,
        }
        for col in PARTY_COLUMNS_2021:
            row[col] = 1 if self.choice == col else 0
        return pd.Series(row)
