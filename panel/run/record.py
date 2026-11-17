"""
The run record: what a saved run was, in enough detail to audit it without the code.

Names make runs groupable; the copied text makes a record self-contained, so a later edit of a
template or a framing cannot change what a past run meant. A spec holds names only, so the
copied text is also how such an edit is noticed: `changed` sets it against today's files.

A run that exists only as a table - one someone else sent and delivered, or one of ours of
which only the aggregate was kept - carries a `delivered` block saying what its rows are. It
is read but never checked; someone else's has a description and a framing that are names
without text, so it cannot be resolved either.
"""
import json
import subprocess
import sys
from dataclasses import asdict
from datetime import datetime
from importlib import metadata
from typing import Any, Dict, List, Mapping, Union

import pandas as pd
import pydantic

from panel import paths
from panel.digest import named
from panel.llm import BaseLLMClient
from panel.topic import Framing

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


def framing_text(framing: Framing) -> Dict[str, Any]:
    return {
        "name": framing.name, "short": framing.short, "language": framing.language, "title": framing.title,
        "instructions": framing.instructions, "labels": framing.labels,
        "schema_text": framing.schema and asdict(framing.schema),
        **({"answer": asdict(framing.answer)} if framing.answer else {}),
    }


def texts(resolved: Resolved) -> Dict[str, Any]:
    """The texts a run is built from, as its record copies them: the `description`, `framing` and `question` blocks."""
    description, question = resolved.description, resolved.question
    shared = resolved.dataset.shared.get(question.by) if question else None
    return {
        "description": {
            "name": description.name, "short": description.short,
            "standalone": description.standalone, "text": [list(block) for block in description.sentences],
        },
        "framing": framing_text(resolved.framing),
        "question": question and {
            "name": question.name, "short": question.short, "language": question.language,
            "text": question.text, "by": question.by, "forms": question.forms, "fallback": question.fallback,
            "shared": shared and {"column": shared.column, "map": shared.map},
        },
    }


def changed(record: Mapping[str, Any], today: Mapping[str, Any]) -> List[str]:
    """
    Where today's texts are not the ones a record copied, as `block.key`; empty when a run sent today
    would read what that run read. `today` is `texts(resolved)` or a part of it. A key the record's
    block does not have is passed over: it did not exist when the run was saved.
    """
    today = json.loads(json.dumps(today))
    found = []
    for block, now in today.items():
        if block not in record:
            continue
        then = record[block]
        if not isinstance(then, dict) or not isinstance(now, dict):
            found += [block] if then != now else []
            continue
        found += [f"{block}.{key}" for key, value in then.items() if key not in now or now[key] != value]
    return found


def build_record(
    resolved: Resolved, n: int, saved_at: datetime, client: Union[BaseLLMClient, None],
    effort: Union[Dict[str, Any], None] = None, faulty: Union[Dict[str, Any], None] = None,
) -> Dict[str, Any]:
    """
    `faulty` is {"<id>" or "<id>:<repeat>": reasons} for the answers kept though flawed (`panel.run.accept`).
    `n` is how many calls came back. `client` gives its parameters, the model id the provider reported and
    the schema as it sends it; `response_schema` is the one the run generated. `effort` is where the run's
    effort stood among the model's levels on the day (`panel.run.effort`), None for a run without one.
    """
    spec, dataset = resolved.spec, resolved.dataset
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
        **texts(resolved),
        "response_schema": resolved.response_model.model_json_schema(),
        "response_schema_sent": client.schema_sent(spec.model, resolved.response_model) if client else None,
        "client_parameters": client.parameters() if client else None,
        "effort": effort,
        "faulty": faulty or {},
        "model_resolved": getattr(client, "last_model", None) if client else None,
        **git_state(),
        "versions": versions(),
    }
