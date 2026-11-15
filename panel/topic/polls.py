"""
What others estimated for a topic's options: published polls, as data.

`config/topics/<name>/polls.json`, optional:

    {
        "polls": {
            "STEM": {"shares": {"ANO": 0.324, ...}, "url": "https://..."},
            ...
        },
        "note": "free text, ignored by code"
    }

`shares` holds a share for every option of the topic; `url` says where it was published. A
poll is one more source to hold against the actual result, beside the runs. No model sees it.
"""
import json

import pandas as pd

from .vocabulary import Topic

CONFIG_NAME = "polls.json"
KEYS = {"polls", "note"}
POLL_KEYS = {"shares", "url"}


def load_polls(topic: Topic) -> pd.DataFrame:
    """
    The topic's polls, one row per poll, a column per option; no rows if it has no `polls.json`.

    Raises:
        ValueError: on unknown keys, or a poll whose shares are not exactly the topic's options.
    """
    path = topic.dir / CONFIG_NAME
    options = list(topic.options)
    if not path.exists():
        return pd.DataFrame(columns=options, dtype=float)
    config = json.loads(path.read_text(encoding="utf-8"))
    polls = config.get("polls", {})
    unknown = sorted(set(config) - KEYS) + sorted(key for poll in polls.values() for key in set(poll) - POLL_KEYS)
    wrong = sorted(name for name, poll in polls.items() if set(poll.get("shares", {})) != set(options))
    if unknown or wrong:
        raise ValueError(f"{path}: unknown keys {unknown}, polls whose shares are not the topic's options {wrong}")
    return pd.DataFrame({name: poll["shares"] for name, poll in polls.items()}, dtype=float).T.reindex(columns=options)
