"""
Structured response models. The shape is code; the vocabulary and the wording are data.

`make_response_models(topic, framing)` builds the two classes a run may ask for: a probability
distribution over the options, and a single pick. The topic supplies the option ids, in order,
and whether there is a turnout dimension; the framing supplies the labels the options are
shown under and every piece of schema text - class titles and field descriptions - because the
provider sees them in the JSON schema, so they are prompt content and are recorded with the run.

Fixed, because saved files are keyed by them: the class names `VotingProbabilities` and
`VotingPick` and their `SHORT_NAME`s (the results stem), and the field names `voted_or_not`,
`voted`, `not_voted`, `parties`, `name`, `probability` and `choice` (every saved result JSON).
`to_series()` writes a results row keyed by option id, whatever label the model saw.
"""
from typing import ClassVar, Dict, List, Literal, Tuple, Type, Union

import pandas as pd
from pydantic import BaseModel, Field, create_model

from .framing import Framing
from .vocabulary import Topic


class TopicResponse(BaseModel):
    """
    What every generated response model can do, whatever its topic.

    `OPTIONS`, `LABELS` and `TURNOUT` are set by `make_response_models`; `voted_or_not` exists
    only on the models of a topic that has an attendance dimension.
    """
    OPTIONS: ClassVar[Tuple[str, ...]] = ()
    LABELS: ClassVar[Dict[str, str]] = {}
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
        """One results row: attendance if any, then every option id, missing ones as 0."""
        row = self.turnout_row()
        for option in (self.parties or []):
            row[self.LABELS[option.name]] = option.probability
        for option in self.OPTIONS:
            row.setdefault(option, 0.0)
        return pd.Series(row)


class PickResponse(TopicResponse):
    """Base of the single-choice model."""
    SHORT_NAME: ClassVar[str] = "VPick"

    def to_series(self) -> pd.Series:
        """One results row: attendance if any, then 1 for the chosen option id, 0 elsewhere."""
        row = self.turnout_row()
        chosen = self.LABELS[self.choice]
        for option in self.OPTIONS:
            row[option] = 1 if option == chosen else 0
        return pd.Series(row)


def make_response_models(topic: Topic, framing: Framing) -> Tuple[Type[ProbabilityResponse], Type[PickResponse]]:
    """
    Build a topic's two response models as one framing words them.

    Returns:
        (probabilities, pick). Their `__name__` and `SHORT_NAME` are fixed, because the
        results filename stem and the run record are built from them.

    Raises:
        ValueError: if the framing words another topic.
    """
    if framing.topic != topic.name:
        raise ValueError(f"framing '{framing.name}' words topic '{framing.topic}', not '{topic.name}'")
    labels = tuple(framing.labels[option] for option in topic.options)
    text = framing.schema

    option_model = create_model(
        text.option.title,
        name=(Literal[labels], Field(description=text.option.name)),
        probability=(float, Field(ge=0, le=1, description=text.option.probability)),
        __doc__=text.option.description,
    )
    turnout_field = {}
    if topic.turnout:
        turnout_model = create_model(
            text.turnout.title,
            voted=(float, Field(ge=0, le=1, description=text.turnout.voted)),
            not_voted=(float, Field(ge=0, le=1, description=text.turnout.not_voted)),
            __doc__=text.turnout.description,
        )
        turnout_field["voted_or_not"] = (turnout_model, Field(description=text.turnout.field))

    probabilities = create_model(
        "VotingProbabilities", __base__=ProbabilityResponse, **turnout_field,
        parties=(List[option_model], Field(description=text.probabilities.parties)),
        __doc__=text.probabilities.description,
    )
    pick = create_model(
        "VotingPick", __base__=PickResponse, **turnout_field,
        choice=(Literal[labels], Field(description=text.pick.choice)),
        __doc__=text.pick.description,
    )
    for model in (probabilities, pick):
        model.OPTIONS = tuple(topic.options)
        model.LABELS = dict(zip(labels, topic.options))
        model.TURNOUT = topic.turnout
    return probabilities, pick
