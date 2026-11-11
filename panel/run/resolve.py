"""
Turn a spec's names into the loaded artifacts and the exact rows a run sends.

Resolving is where the names are checked against each other: the dataset has to have the
description and its prompt table has to cover the values, the framing has to share the
dataset's language, the topic has to have the question in that language, a standalone
description takes no question and any other needs one, client and shape have to be known,
an effort needs a client that takes one,
and the model id has to fit a filename.
Anything wrong fails here, before a call is paid for.

A run reads the prompt table. The values table is analysis's; here it supplies the subjects
the prompt table must cover, so a table rendered before the values were rebuilt is refused,
and the one column a question worded by a shared vocabulary asks for.
"""
import hashlib
import inspect
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Type, Union

import pandas as pd

from panel import paths
from panel.dataset import Dataset, load_dataset, read_values, shared_keys
from panel.describe.render import read_prompts
from panel.describe.template import PromptTemplate, load_template
from panel.llm import BaseLLMClient
from panel.topic import Framing, Question, Topic, load_framing, load_question, load_topic, make_response_models
from panel.topic.response import TopicResponse

from .registry import client_class, short_name
from .spec import RunSpec

MODEL_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9.\-]*")
SUBJECT = "id"
REPEAT = "repeat"
QUESTION = "question"


@dataclass(frozen=True)
class Resolved:
    """
    A spec with its artifacts loaded and its rows chosen.

    Attributes:
        spec: the declaration.
        dataset, description, topic, framing: the four named artifacts.
        question: the fifth, or None after a standalone description.
        client_type: the client class the spec's `client` names.
        response_model: the generated model the spec's `shape` names.
        rows: the subjects sent, sampled if asked - the description column, and beside it
            `question`, the wording each subject reads, when the run has a question.
        digests: sha256 of the files the prompts came from: `values`, `prompts`, `forms`.
    """
    spec: RunSpec
    dataset: Dataset
    description: PromptTemplate
    topic: Topic
    framing: Framing
    question: Union[Question, None]
    client_type: Type[BaseLLMClient]
    response_model: Type[TopicResponse]
    rows: pd.DataFrame
    digests: Dict[str, str]

    @property
    def stem(self) -> str:
        """
        Filename stem, every part a short:
        `<dataset>_<description>_<framing>[_<question>]_<Client>_<model>_<Shape>_t=<t>[_e=<effort>][_x<repeats>][_subset=<n>]`;
        no question part after a standalone description; `t=default` when the spec sends no temperature.
        """
        spec = self.spec
        parts = [self.dataset.short, self.description.short, self.framing.short]
        if self.question:
            parts.append(self.question.short)
        parts += [
            short_name(self.client_type), spec.model, spec.shape,
            "t=default" if spec.temperature is None else f"t={round(float(spec.temperature), 2)}",
        ]
        if spec.effort:
            parts.append(f"e={spec.effort}")
        if spec.repeats > 1:
            parts.append(f"x{spec.repeats}")
        if spec.sample is not None:
            parts.append(f"subset={spec.sample}")
        return "_".join(parts)

    def prompt(self, row: pd.Series) -> str:
        """What one call sends as the user message: the description, then the question if there is one."""
        return f"{row[self.description.name]} {row[QUESTION]}" if self.question else row[self.description.name]

    @property
    def calls(self) -> pd.DataFrame:
        """One row per call - `id`, `repeat`, the columns of `rows` - indexed 0..n*repeats-1, subject-major."""
        rows = self.rows
        calls = rows.loc[rows.index.repeat(self.spec.repeats)].reset_index()
        calls[REPEAT] = [repeat for _ in range(len(rows)) for repeat in range(1, self.spec.repeats + 1)]
        calls.index.name = "call"
        return calls[[SUBJECT, REPEAT, *rows.columns]]


def resolve(
    spec: RunSpec,
    datasets_dir: Union[str, Path] = paths.DATASETS_DIR,
    topics_dir: Union[str, Path] = paths.TOPICS_DIR,
) -> Resolved:
    """
    Load what a spec names and lay out its rows.

    Raises:
        FileNotFoundError: if an artifact, the values or the prompt table is missing.
        ValueError: if the dataset has no such description, the languages disagree, a
            question is named after a standalone description or missing after any other,
            the question cannot be worded for a subject, the prompt table does not cover
            the values, the model id cannot sit in a filename, the client takes no effort
            and one is set, or the sample exceeds the
            subjects.
        KeyError: if the client is unknown.
    """
    dataset = load_dataset(spec.dataset, datasets_dir)
    topic = load_topic(spec.topic, topics_dir)
    framing = load_framing(topic, spec.framing, topics_dir)
    if spec.description not in dataset.descriptions:
        raise ValueError(
            f"dataset '{dataset.name}' has no description '{spec.description}'; it has {list(dataset.descriptions)}"
        )
    description = load_template(dataset.description_path(spec.description))
    if dataset.language != framing.language:
        raise ValueError(f"languages disagree: dataset '{dataset.language}', framing '{framing.language}'")
    if description.standalone != (spec.question is None):
        raise ValueError(
            f"description '{description.name}' is {'' if description.standalone else 'not '}standalone, "
            f"question is {spec.question!r}: a standalone description takes none, any other needs one"
        )
    question = load_question(topic, framing.language, spec.question, topics_dir) if spec.question else None
    if not MODEL_ID.fullmatch(spec.model):
        raise ValueError(f"model id {spec.model!r} cannot sit in a filename")
    client_type = client_class(spec.client)
    if spec.effort and "effort" not in inspect.signature(client_type.__init__).parameters:
        raise ValueError(f"client '{spec.client}' takes no effort; the spec sets {spec.effort!r}")
    response_model = {model.SHORT_NAME: model for model in make_response_models(topic, framing)}[spec.shape]

    values = read_values(dataset)
    rows = read_prompts(dataset, description.name, values.index)
    if question:
        shared = question.by is not None and question.by in dataset.shared
        keys = shared_keys(dataset, question.by, values) if shared else pd.Series(None, index=values.index, dtype=object)
        rows[QUESTION] = keys.map(question.wording, na_action=None)
    if spec.sample is not None:
        rows = rows.sample(n=spec.sample, random_state=spec.random_state)
    sources = {
        "values": dataset.values_path, "prompts": dataset.prompts_path(description.name), "forms": dataset.forms_path,
    }
    digests = {name: hashlib.sha256(path.read_bytes()).hexdigest() for name, path in sources.items()}
    return Resolved(spec, dataset, description, topic, framing, question, client_type, response_model, rows, digests)
