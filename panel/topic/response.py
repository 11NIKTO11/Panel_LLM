"""
Structured response models. The shape is code; the vocabulary and the wording are data.

`make_response_models(topic, framing)` builds the two classes a run may ask for: a probability
distribution over the options, and a single pick. The topic supplies the option ids, in order,
and whether there is a turnout dimension; the framing supplies the labels the options are
shown under and every piece of schema text - class titles and field descriptions - because the
provider sees them in the JSON schema, so they are prompt content and are recorded with the run.

Saved files are keyed by the class names `Probabilities` and `Pick`, their `SHORT_NAME`s
`Prob` and `Pick` (the results stem), and the field names `voted_or_not`, `voted`,
`not_voted`, `options`, `name`, `probability` and `choice` (every saved result JSON); a
rename orphans what is saved. None of them names a topic: the classes are general, and what
the options are is the framing's text.

`to_series()` writes a results row keyed by option id, whatever label the model saw.

The turnout field carries no description of its own: a description beside a `$ref` is dropped by
some providers' SDKs and inlined by others, so the framing's `turnout.field` text follows
`turnout.description` in the turnout model's own description, where every provider reads it.

A framing that states an answer menu instead of schema text (`DECISIONS.md` §40) gets one model,
`Probabilities`, of the same fields and with no text: it is never sent. A client that reads the
answer off the token distribution builds it with `from_tokens`:

- `mass` is the probability the model put on the menu at all; the rest went to tokens that are
  no answer. It is kept beside the answer and counted under no entry of the menu: what the model
  was about to say instead is neither a party nor an abstention.
- everything else is a share of `mass`, so each group sums to 1 as in a written answer: with
  turnout, `voted` is the options' tokens, `not_voted` the abstain token, and each option its
  share of `voted`; without, each option is its token.
"""
from typing import ClassVar, Dict, List, Literal, Mapping, Tuple, Type, Union

import pandas as pd
from pydantic import BaseModel, Field, create_model

from .framing import Framing
from .vocabulary import Topic


class TopicResponse(BaseModel):
    """
    `OPTIONS`, `LABELS` and `TURNOUT` are set by `make_response_models`; `voted_or_not` exists
    only on the models of a topic that has an attendance dimension.
    """
    OPTIONS: ClassVar[Tuple[str, ...]] = ()
    LABELS: ClassVar[Dict[str, str]] = {}
    TURNOUT: ClassVar[Union[Tuple[str, str], None]] = None

    def turnout_row(self) -> dict:
        if not self.TURNOUT:
            return {}
        yes, no = self.TURNOUT
        return {yes: self.voted_or_not.voted, no: self.voted_or_not.not_voted}


class ProbabilityResponse(TopicResponse):
    SHORT_NAME: ClassVar[str] = "Prob"

    def to_series(self) -> pd.Series:
        """One results row in the topic's column order: attendance if any, then every option id, missing ones as 0."""
        row = self.turnout_row()
        for option in (self.options or []):
            row[self.LABELS[option.name]] = option.probability
        for option in self.OPTIONS:
            row.setdefault(option, 0.0)
        return pd.Series(row).reindex(list(self.TURNOUT or ()) + list(self.OPTIONS))


class TokenProbabilityResponse(ProbabilityResponse):
    """`TOKENS` is token -> label; `ABSTAIN` the token for not having voted; `ANSWERS` both, what a client looks for."""
    TOKENS: ClassVar[Dict[str, str]] = {}
    ABSTAIN: ClassVar[Union[str, None]] = None
    ANSWERS: ClassVar[Tuple[str, ...]] = ()

    @classmethod
    def from_tokens(cls, found: Mapping[str, float]) -> "TokenProbabilityResponse":
        """`found` is answer token -> probability; a token it lacks counts as 0."""
        mass = sum(found.get(token, 0.0) for token in cls.ANSWERS)
        given = {label: found.get(token, 0.0) for token, label in cls.TOKENS.items()}
        voted = sum(given.values())
        payload = {"mass": min(mass, 1.0)}
        if cls.TURNOUT:
            payload["voted_or_not"] = {"voted": min(voted / mass, 1.0), "not_voted": min(found.get(cls.ABSTAIN, 0.0) / mass, 1.0)}
        whole = voted if cls.TURNOUT else mass
        payload["options"] = [{"name": label, "probability": min(value / whole, 1.0) if whole else 0.0} for label, value in given.items()]
        return cls.model_validate(payload)


class PickResponse(TopicResponse):
    SHORT_NAME: ClassVar[str] = "Pick"

    def to_series(self) -> pd.Series:
        row = self.turnout_row()
        chosen = self.LABELS[self.choice]
        for option in self.OPTIONS:
            row[option] = 1 if option == chosen else 0
        return pd.Series(row)


def _token_model(topic: Topic, framing: Framing, labels: Tuple[str, ...]) -> Type[TokenProbabilityResponse]:
    option_model = create_model("Option", name=(Literal[labels], ...), probability=(float, Field(ge=0, le=1)))
    turnout_field = {}
    if topic.turnout:
        turnout_model = create_model("Turnout", voted=(float, Field(ge=0, le=1)), not_voted=(float, Field(ge=0, le=1)))
        turnout_field["voted_or_not"] = (turnout_model, ...)
    model = create_model(
        "Probabilities", __base__=TokenProbabilityResponse, **turnout_field,
        options=(List[option_model], ...), mass=(float, Field(gt=0, le=1)),
    )
    menu = framing.answer
    model.TOKENS = {token: framing.labels[option] for option, token in menu.tokens.items()}
    model.ABSTAIN = menu.abstain.token if menu.abstain else None
    model.ANSWERS = tuple(model.TOKENS) + ((model.ABSTAIN,) if menu.abstain else ())
    return model


def make_response_models(topic: Topic, framing: Framing) -> Tuple[Type[TopicResponse], ...]:
    """
    (probabilities, pick), or (probabilities,) for a framing with an answer menu; their `__name__` and
    `SHORT_NAME` are fixed, because the results filename stem and the run record are built from them.
    """
    if framing.topic != topic.name:
        raise ValueError(f"framing '{framing.name}' words topic '{framing.topic}', not '{topic.name}'")
    labels = tuple(framing.labels[option] for option in topic.options)
    models = (_token_model(topic, framing, labels),) if framing.answer else _schema_models(topic, framing, labels)
    for model in models:
        model.OPTIONS = tuple(topic.options)
        model.LABELS = dict(zip(labels, topic.options))
        model.TURNOUT = topic.turnout
    return models


def _schema_models(topic: Topic, framing: Framing, labels: Tuple[str, ...]) -> Tuple[Type[ProbabilityResponse], Type[PickResponse]]:
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
            __doc__=f"{text.turnout.description}\n\n{text.turnout.field}",
        )
        turnout_field["voted_or_not"] = (turnout_model, ...)

    probabilities = create_model(
        "Probabilities", __base__=ProbabilityResponse, **turnout_field,
        options=(List[option_model], Field(description=text.probabilities.options)),
        __doc__=text.probabilities.description,
    )
    pick = create_model(
        "Pick", __base__=PickResponse, **turnout_field,
        choice=(Literal[labels], Field(description=text.pick.choice)),
        __doc__=text.pick.description,
    )
    return probabilities, pick
