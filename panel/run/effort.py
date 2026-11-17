"""
Where a run's effort stands among the levels its model takes, asked of the provider when the run is sent.

The levels differ by provider and by model, and change with new models, so no list of them is
kept here: the client asks (`BaseLLMClient.efforts`). A level the model does not take is
refused before a call is paid for - a provider may otherwise move it to the nearest one it
has and answer as if nothing happened. The rank is a position, not an amount: the same rank
on two models does not mean the same thinking.
"""
from typing import Any, Dict, Union

from panel.llm import BaseLLMClient

from .spec import RunSpec


def effort_tier(client: BaseLLMClient, spec: RunSpec) -> Union[Dict[str, Any], None]:
    """
    None for a run without an effort. Else `level`, and from what the provider reports `levels`
    (lowest first), `rank` (1 is the lowest) and `of`; the three are None where the provider has no list.
    A client built with another effort than the spec's raises too: the client's is what is sent.
    """
    if spec.effort is None:
        return None
    sent = getattr(client, "effort", None)
    if sent != spec.effort:
        raise ValueError(f"the run sets effort {spec.effort!r}, the client was built with {sent!r}")
    levels = client.efforts(spec.model)
    if levels is None:
        return {"level": spec.effort, "rank": None, "of": None, "levels": None}
    if spec.effort not in levels:
        raise ValueError(f"'{spec.model}' takes no effort {spec.effort!r}; it takes {list(levels) or 'none at all'}")
    return {"level": spec.effort, "rank": levels.index(spec.effort) + 1, "of": len(levels), "levels": list(levels)}
