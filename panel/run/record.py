"""
The run record: what a saved run was, in enough detail to audit it without the code.

Names make runs groupable; the copied text makes a record self-contained, so a later edit of a
template or a framing cannot change what a past run meant. Legacy records under
the legacy shape keep it; nothing rewrites them.
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
from panel.llm import BaseLLMClient

from .resolve import Resolved

SDKS = ("openai", "anthropic", "google-genai", "requests")


def _git(*args: str) -> str:
    done = subprocess.run(["git", *args], capture_output=True, text=True, timeout=5, cwd=paths.ROOT)
    return done.stdout.strip()


def versions() -> Dict[str, str]:
    """Python, the data stack and every provider SDK that is installed."""
    out = {"python": sys.version.split()[0], "pandas": pd.__version__, "pydantic": pydantic.__version__}
    for package in SDKS:
        try:
            out[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            continue
    return out


def build_record(resolved: Resolved, n: int, saved_at: datetime, client: Union[BaseLLMClient, None]) -> Dict[str, Any]:
    """
    Everything a saved run should carry beside its results.

    Args:
        resolved: the run as sent.
        n: how many calls came back.
        saved_at: when the files were written.
        client: the client that made the calls, for the model id the provider reported.
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
            **{f"{name}_sha256": digest for name, digest in resolved.digests.items()},
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
        "model_resolved": getattr(client, "last_model", None) if client else None,
        "git_commit": _git("rev-parse", "HEAD") or None,
        "git_dirty": bool(_git("status", "--porcelain", "--untracked-files=no")),
        "versions": versions(),
    }
