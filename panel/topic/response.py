"""
Structured response models. The shape is code; the vocabulary is topic data.

`make_response_models(topic)` substitutes the option `Literal` and drops the attendance
field for a topic that has none. Nothing else about the classes varies: the field
descriptions here are deliberately generic, because what a run is *about* belongs in the
system prompt, which is stored with the run (`RunSpec.instructions`). Field names are part
of the saved JSON, so they stay fixed - `voted_or_not` and `parties` read as election
vocabulary, and are kept anyway so old result files still parse.

The Czech descriptions are prompt-facing: they end up in the JSON schema the provider is
sent, so changing them changes the results.
"""
from typing import ClassVar, List, Literal, Tuple, Type, Union

import pandas as pd
from pydantic import BaseModel, Field, create_model


class TurnoutProbability(BaseModel):
    """Pravděpodobnost, že se respondent zúčastnil, a že se nezúčastnil."""
    voted: float = Field(ge=0, le=1, description="Pravděpodobnost, že se respondent zúčastnil.")
    not_voted: float = Field(ge=0, le=1, description="Pravděpodobnost, že se respondent nezúčastnil.")


class TopicResponse(BaseModel):
    """
    What every generated response model can do, whatever its topic.

    `OPTIONS` and `TURNOUT` are set by `make_response_models`; `voted_or_not` exists only
    on the models of a topic that has an attendance dimension.
    """
    OPTIONS: ClassVar[Tuple[str, ...]] = ()
    TURNOUT: ClassVar[Union[Tuple[str, str], None]] = None

    def turnout_row(self) -> dict:
        """{yes column: p, no column: p}, or empty for a topic without attendance."""
        if not self.TURNOUT:
            return {}
        yes, no = self.TURNOUT
        return {yes: self.voted_or_not.voted, no: self.voted_or_not.not_voted}


class ProbabilityResponse(TopicResponse):
    """Base of the per-option probability model."""
    SHORT_NAME: ClassVar[str] = "VProb"

    def to_series(self) -> pd.Series:
        """One results row: attendance if any, then every option, missing ones as 0."""
        row = self.turnout_row()
        for option in (self.parties or []):
            row[option.name] = option.probability
        for name in self.OPTIONS:
            row.setdefault(name, 0.0)
        return pd.Series(row)


class PickResponse(TopicResponse):
    """Base of the single-choice model."""
    SHORT_NAME: ClassVar[str] = "VPick"

    def to_series(self) -> pd.Series:
        """One results row: attendance if any, then 1 for the chosen option, 0 elsewhere."""
        row = self.turnout_row()
        for name in self.OPTIONS:
            row[name] = 1 if self.choice == name else 0
        return pd.Series(row)


def make_response_models(topic) -> Tuple[Type[ProbabilityResponse], Type[PickResponse]]:
    """
    Build a topic's two response models.

    Args:
        topic: a `panel.topic.Topic`; only `options` and `turnout` are used.

    Returns:
        (probabilities, pick). Their `__name__` and `SHORT_NAME` are fixed, because the
        results filename stem and the run record are built from them.
    """
    options = tuple(topic.options)
    option_model = create_model(
        "OptionProbability",
        name=(Literal[options], Field(description="Identifikátor jedné z nabízených možností.")),
        probability=(float, Field(ge=0, le=1, description="Pravděpodobnost této možnosti.")),
        __doc__="Pravděpodobnost jedné z nabízených možností.",
    )

    turnout_field = {}
    if topic.turnout:
        turnout_field["voted_or_not"] = (
            TurnoutProbability,
            Field(description="Pravděpodobnost účasti a neúčasti. Součet musí být 1.0."),
        )

    probabilities = create_model(
        "VotingProbabilities", __base__=ProbabilityResponse, **turnout_field,
        parties=(
            List[option_model],
            Field(description="Seznam nabízených možností s pravděpodobnostmi. Součet musí být 1.0."),
        ),
        __doc__="Strukturovaný výstup: rozdělení pravděpodobnosti mezi nabízené možnosti.",
    )
    pick = create_model(
        "VotingPick", __base__=PickResponse, **turnout_field,
        choice=(Literal[options], Field(description="Nejpravděpodobnější možnost pro tento profil.")),
        __doc__="Zjednodušený výstup: jedna vybraná možnost.",
    )

    for model in (probabilities, pick):
        model.OPTIONS = options
        model.TURNOUT = topic.turnout
    return probabilities, pick
