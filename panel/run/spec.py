"""
A run, declared in full by names and numbers.

Five axes name a run - dataset, description, topic, framing, and the model (client, model id,
temperature) - plus how much of the data (sample) and how often (repeats). Every field is a
string or a number, so a spec serialises into the run record as it is and a record turns back
into the spec that wrote it. Resolving the names into loaded artifacts is `panel.run.resolve`;
sending the calls and saving what comes back is `panel.run.execute`.
"""
from dataclasses import asdict, dataclass, replace
from typing import Any, Dict, Union

SHAPES = ("VProb", "VPick")


@dataclass(frozen=True)
class RunSpec:
    """
    Attributes:
        dataset: whose subjects; a folder under `datasets/`.
        description: which description template; a file under `config/descriptions/`.
        topic: what the run is about; a folder under `config/topics/`, and the results folder.
        framing: which wording of the topic; a file in the topic folder.
        client: the LLM client, by its short name (`OpenRouter`, `OpenAI`, `Fake`, ...).
        model: provider model id.
        shape: which response model: `VProb` (a distribution) or `VPick` (one choice).
        temperature: sampling temperature.
        repeats: calls per subject; each becomes its own results row.
        sample: run this many subjects instead of all of them.
        random_state: seed for `sample`.
    """
    dataset: str
    description: str
    topic: str
    framing: str
    client: str
    model: str
    shape: str = "VProb"
    temperature: float = 0.0
    repeats: int = 1
    sample: Union[int, None] = None
    random_state: Union[int, None] = 42

    def __post_init__(self):
        """
        Raises:
            ValueError: on an unknown shape, or a repeats or sample below 1.
        """
        if self.shape not in SHAPES:
            raise ValueError(f"shape must be one of {SHAPES}, not {self.shape!r}")
        if self.repeats < 1:
            raise ValueError("repeats must be at least 1")
        if self.sample is not None and self.sample < 1:
            raise ValueError("sample must be at least 1")

    def variant(self, **changes) -> "RunSpec":
        """A copy with some fields changed, for declaring a family of runs."""
        return replace(self, **changes)

    def to_dict(self) -> Dict[str, Any]:
        """The spec as the run record stores it."""
        return asdict(self)

    @classmethod
    def from_dict(cls, record: Dict[str, Any]) -> "RunSpec":
        """
        The spec a record stores, back as an object.

        Raises:
            TypeError: on a legacy record's spec, whose fields are not these.
        """
        return cls(**record)
