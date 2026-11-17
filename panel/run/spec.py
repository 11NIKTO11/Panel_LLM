"""
A run, declared in full by names and numbers.

Six names say what a run is - dataset, description, topic, framing, question, and the model
(client, model id, temperature, effort) - plus how much of the data (sample) and how often (repeats). Every field is a
string or a number, so a spec serialises into the run record as it is and a record turns back
into the spec that wrote it. Resolving the names into loaded artifacts is `panel.run.resolve`;
sending the calls and saving what comes back is `panel.run.execute`.
"""
from dataclasses import asdict, dataclass, replace
from typing import Any, Dict, Union

from panel.llm import EFFORT_ORDER

SHAPES = ("Prob", "Pick")
# `none` is a provider's word for no reasoning; here it already stands for "set no effort" (presets.NONE).
EFFORTS = tuple(level for level in EFFORT_ORDER if level != "none")


@dataclass(frozen=True)
class RunSpec:
    """
    dataset: a folder under `datasets/`.
    description: a file in the dataset's `descriptions/` folder.
    topic: a folder under `config/topics/`, and the results folder.
    framing: a file in the topic's `framings/`.
    question: a file in the topic's `questions/`; None for a standalone description, which is the whole prompt.
    client: by its short name (`OpenRouter`, `OpenAI`, `Fake`, ...).
    shape: `Prob` (a distribution) or `Pick` (one choice).
    temperature: None sends none, for a provider that takes no such setting.
    effort: None sends none and leaves the model's default; only a client that takes an effort can be given one,
        and which levels a model takes is asked of the provider when the run is sent (`panel.run.effort`).
    repeats: calls per subject; each becomes its own results row.
    sum_low, sum_high: what an answer's probabilities may sum to and still be accepted (`panel.run.accept`),
        for the options and for turnout alike; None leaves that side open, as every run before 2026-11-17 did.
    sample: run this many subjects instead of all of them.
    """
    dataset: str
    description: str
    topic: str
    framing: str
    question: Union[str, None]
    client: str
    model: str
    shape: str = "Prob"
    temperature: Union[float, None] = 0.0
    effort: Union[str, None] = None
    repeats: int = 1
    sum_low: Union[float, None] = None
    sum_high: Union[float, None] = None
    sample: Union[int, None] = None
    random_state: Union[int, None] = 42

    def __post_init__(self):
        if self.shape not in SHAPES:
            raise ValueError(f"shape must be one of {SHAPES}, not {self.shape!r}")
        if self.effort is not None and self.effort not in EFFORTS:
            raise ValueError(f"effort must be one of {EFFORTS} or None, not {self.effort!r}")
        if self.repeats < 1:
            raise ValueError("repeats must be at least 1")
        if (self.sum_low is not None and self.sum_low > 1) or (self.sum_high is not None and self.sum_high < 1):
            raise ValueError(f"the accepted sums must include 1, not {self.sum_low} .. {self.sum_high}")
        if self.sample is not None and self.sample < 1:
            raise ValueError("sample must be at least 1")

    def variant(self, **changes) -> "RunSpec":
        return replace(self, **changes)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, record: Dict[str, Any]) -> "RunSpec":
        return cls(**record)
