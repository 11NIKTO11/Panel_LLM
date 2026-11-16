"""
The run record: what a saved run was, in enough detail to audit it without the code.

Names make runs groupable; the copied text makes a record self-contained, so a later edit of a
template or a framing cannot change what a past run meant.

A run that exists only as a table - one someone else sent and delivered, or one of ours of
which only the aggregate was kept - carries a `delivered` block saying what its rows are. It
is read but never checked; someone else's has a description and a framing that are names
without text, so it cannot be resolved either.
"""
import subprocess
import sys
from dataclasses import asdict
from datetime import datetime
from importlib import metadata
from typing import Any, Dict, Union

import pandas as pd
import pydantic

from panel import paths
from panel.digest import named
from panel.llm import BaseLLMClient

from .resolve import Resolved

SDKS = ("openai", "anthropic", "google-genai", "requests")
DELIVERED = "delivered"
MIGRATED = "migrated"
KEPT = (MIGRATED, DELIVERED)


def _git(*args: str) -> str:
    done = subprocess.run(["git", *args], capture_output=True, text=True, timeout=5, cwd=paths.ROOT)
    return done.stdout.strip()


def git_state() -> Dict[str, Any]:
    """`git_dirty` is whether tracked files differ from the commit; untracked ones do not count."""
    return {"git_commit": _git("rev-parse", "HEAD") or None, "git_dirty": bool(_git("status", "--porcelain", "--untracked-files=no"))}


def versions() -> Dict[str, str]:
    out = {"python": sys.version.split()[0], "pandas": pd.__version__, "pydantic": pydantic.__version__}
    for package in SDKS:
        try:
            out[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            continue
    return out


def build_record(resolved: Resolved, n: int, saved_at: datetime, client: Union[BaseLLMClient, None]) -> Dict[str, Any]:
    """
    `n` is how many calls came back. `client` gives its parameters, the model id the provider reported and
    the schema as it sends it; `response_schema` is the one the run generated.
    """
    spec, dataset, description, framing = resolved.spec, resolved.dataset, resolved.description, resolved.framing
    question = resolved.question
    shared = dataset.shared.get(question.by) if question else None
    return {
        "stem": resolved.stem,
        "n": n,
        "saved_at": saved_at.isoformat(timespec="seconds"),
        "spec": spec.to_dict(),
        "dataset": {
            "name": dataset.name, "short": dataset.short, "language": dataset.language,
            "rows": [int(subject) for subject in resolved.rows.index],
            **named(resolved.digests),
        },
        "description": {
            "name": description.name, "short": description.short,
            "standalone": description.standalone, "text": [list(block) for block in description.sentences],
        },
        "framing": {
            "name": framing.name, "short": framing.short, "language": framing.language, "title": framing.title,
            "instructions": framing.instructions, "labels": framing.labels,
            "schema_text": asdict(framing.schema),
        },
        "question": question and {
            "name": question.name, "short": question.short, "language": question.language,
            "text": question.text, "by": question.by, "forms": question.forms, "fallback": question.fallback,
            "shared": shared and {"column": shared.column, "map": shared.map},
        },
        "response_schema": resolved.response_model.model_json_schema(),
        "response_schema_sent": client.schema_sent(spec.model, resolved.response_model) if client else None,
        "client_parameters": client.parameters() if client else None,
        "model_resolved": getattr(client, "last_model", None) if client else None,
        **git_state(),
        "versions": versions(),
    }
