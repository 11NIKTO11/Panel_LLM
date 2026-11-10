"""
Turn a spec's names into the loaded artifacts and the exact rows a run sends.

Resolving is where the axes are checked against each other: the dataset has to declare the
description and its table has to cover the profile, description and framing have to share the
dataset's language, client and shape have to be known, and the model id has to fit a filename.
Anything wrong fails here, before a call is paid for.
"""
import hashlib
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Type, Union

import pandas as pd

from panel import paths
from panel.dataset import Dataset, load_dataset, profile_path, read_profile
from panel.describe.tables import read_table
from panel.describe.template import PromptTemplate, load_description
from panel.llm import BaseLLMClient
from panel.topic import Framing, Topic, load_framing, load_topic, make_response_models
from panel.topic.response import TopicResponse

from .registry import client_class, short_name
from .spec import RunSpec

MODEL_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9.\-]*")
SUBJECT = "id"
REPEAT = "repeat"


@dataclass(frozen=True)
class Resolved:
    """
    A spec with its artifacts loaded and its rows chosen.

    Attributes:
        spec: the declaration.
        dataset, description, topic, framing: the four named artifacts.
        client_type: the client class the spec's `client` names.
        response_model: the generated model the spec's `shape` names.
        rows: the subjects sent - profile joined with the description column, sampled if asked.
        profile_sha256: digest of the profile file the rows came from.
    """
    spec: RunSpec
    dataset: Dataset
    description: PromptTemplate
    topic: Topic
    framing: Framing
    client_type: Type[BaseLLMClient]
    response_model: Type[TopicResponse]
    rows: pd.DataFrame
    profile_sha256: str

    @property
    def stem(self) -> str:
        """
        Filename stem, every part a short:
        `<dataset>_<description>_<framing>_<Client>_<model>_<Shape>_t=<t>[_x<repeats>][_subset=<n>]`;
        `t=default` when the spec sends no temperature.
        """
        spec = self.spec
        parts = [
            self.dataset.short, self.description.short, self.framing.short, short_name(self.client_type),
            spec.model, spec.shape,
            "t=default" if spec.temperature is None else f"t={round(float(spec.temperature), 2)}",
        ]
        if spec.repeats > 1:
            parts.append(f"x{spec.repeats}")
        if spec.sample is not None:
            parts.append(f"subset={spec.sample}")
        return "_".join(parts)

    @property
    def question(self) -> str:
        """What follows the description: the framing's question, or nothing after a standalone description."""
        return "" if self.description.standalone else self.framing.question

    def prompt(self, row: pd.Series) -> str:
        """What one call sends as the user message."""
        return row[self.description.name] + self.question

    @property
    def calls(self) -> pd.DataFrame:
        """One row per call - `id`, `repeat`, the description column - indexed 0..n*repeats-1, subject-major."""
        rows = self.rows[[self.description.name]]
        calls = rows.loc[rows.index.repeat(self.spec.repeats)].reset_index()
        calls[REPEAT] = [repeat for _ in range(len(rows)) for repeat in range(1, self.spec.repeats + 1)]
        calls.index.name = "call"
        return calls[[SUBJECT, REPEAT, self.description.name]]


def resolve(
    spec: RunSpec,
    datasets_dir: Union[str, Path] = paths.DATASETS_DIR,
    descriptions_dir: Union[str, Path] = paths.DESCRIPTIONS_DIR,
    topics_dir: Union[str, Path] = paths.TOPICS_DIR,
    prompts_dir: Union[str, Path] = paths.PROMPTS_DIR,
) -> Resolved:
    """
    Load what a spec names and lay out its rows.

    Raises:
        FileNotFoundError: if an artifact, the profile or the prompt table is missing.
        ValueError: if the dataset does not declare the description, the languages disagree,
            the table does not cover the profile, the model id cannot sit in a filename, or
            the sample exceeds the subjects.
        KeyError: if the client is unknown.
    """
    dataset = load_dataset(spec.dataset, datasets_dir)
    description = load_description(spec.description, descriptions_dir)
    topic = load_topic(spec.topic, topics_dir)
    framing = load_framing(topic, spec.framing, topics_dir)
    if spec.description not in dataset.descriptions:
        raise ValueError(f"dataset '{dataset.name}' does not declare description '{description.name}'")
    languages = {"dataset": dataset.language, "description": description.language, "framing": framing.language}
    if len(set(languages.values())) != 1:
        raise ValueError(f"languages disagree: {languages}")
    if not MODEL_ID.fullmatch(spec.model):
        raise ValueError(f"model id {spec.model!r} cannot sit in a filename")
    client_type = client_class(spec.client)
    response_model = {model.SHORT_NAME: model for model in make_response_models(topic, framing)}[spec.shape]

    profile = read_profile(dataset)
    rows = profile.join(read_table(dataset.name, description.name, profile.index, prompts_dir))
    if spec.sample is not None:
        rows = rows.sample(n=spec.sample, random_state=spec.random_state)
    digest = hashlib.sha256(profile_path(dataset).read_bytes()).hexdigest()
    return Resolved(spec, dataset, description, topic, framing, client_type, response_model, rows, digest)
