from pydantic import BaseModel, Field,  field_validator
from typing import List, Tuple, Literal
import pandas as pd

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
