from dataclasses import dataclass, field, replace
from typing import Any, Dict, Type, Union

import pandas as pd
from pydantic import BaseModel


def short_name(obj) -> str:
    """A class's SHORT_NAME if it declares one, else its __name__."""
    return getattr(obj, "SHORT_NAME", None) or obj.__name__


@dataclass(frozen=True)
class RunSpec:
    """
    One LLM run, declared in full: everything `run_llm_batch` needs to execute it.

    `to_dict` records everything that identifies a run except the rows themselves; the
    prompts those rows produced are stored with the results, in the `Prompt` column. No
    field holds a callable, so a record fully describes the run that wrote it.

    Attributes:
        name: label for the run; the first part of every filename.
        topic: what the run is about - an election, or an issue. Also the results
            folder the run lands in, so runs sharing a response vocabulary sit
            together. Selects that vocabulary in a later step.
        data: rows to run, one subject each. Excluded from equality and repr, so a spec
            still compares and hashes by its declaration.
        description: which description variant to send - a column of `data`, which is
            also the name of the template that rendered it and of its prompt table (see
            `panel/describe/template.py`). A string rather than a callable, so a spec is data.
        question: appended to every description; empty when the description carries the
            whole prompt itself.
        instructions: system prompt framing the task, normally the topic's. Recorded with
            the run, so a run's whole prompt is this plus its stored `Prompt` column.
        client_type: BaseLLMClient subclass to instantiate.
        model: provider model id.
        response_model: Pydantic model the provider must return.
        temperature: sampling temperature.
        sample: run this many rows instead of all of them.
        random_state: seed for `sample`.
    """
    name: str
    topic: str
    data: pd.DataFrame = field(compare=False, repr=False)
    description: str
    question: str
    client_type: Type
    model: str
    response_model: Type[BaseModel]
    instructions: Union[str, None] = None
    temperature: float = 0.0
    sample: Union[int, None] = None
    random_state: Union[int, None] = 42

    @property
    def stem(self) -> str:
        """
        Filename stem: enough to tell runs apart at a glance, not the whole spec.

        The topic is not in it - it is the folder the run lands in. `subset=<n>`
        appears only for a sampled run; a full run carries no suffix.
        """
        parts = [
            self.name,
            short_name(self.client_type),
            self.model,
            short_name(self.response_model),
            f"t={round(float(self.temperature), 2)}",
        ]
        if self.sample is not None:
            parts.append(f"subset={self.sample}")
        return "_".join(parts)

    def make_client(self, api_key: Union[str, None] = None):
        return self.client_type(api_key=api_key)

    def prompt(self, row: pd.Series) -> str:
        """
        What one row is sent: its stored description plus the topic's question.

        Raises:
            KeyError: if `data` carries no such description column.
        """
        return row[self.description] + self.question

    def rows(self) -> pd.DataFrame:
        """The rows this run covers: all of `data`, or a random `sample` of them."""
        if self.sample is None:
            return self.data
        return self.data.sample(n=self.sample, random_state=self.random_state)

    def variant(self, **changes) -> "RunSpec":
        """A copy with some fields changed, for declaring a family of runs."""
        return replace(self, **changes)

    def to_dict(self) -> Dict[str, Any]:
        """The identifying fields, for the run record. The rows themselves are omitted."""
        return {
            "name": self.name,
            "topic": self.topic,
            "description": self.description,
            "question": self.question,
            "instructions": self.instructions,
            "client_type": self.client_type.__name__,
            "model": self.model,
            "response_model": self.response_model.__name__,
            "temperature": float(self.temperature),
            "sample": self.sample,
            "random_state": self.random_state,
        }
