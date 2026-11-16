"""
Send a resolved run and save what comes back: the one loop that turns a spec into files.

The batch runner handles per-call retries; this module handles what is left after a batch -
capped rerun rounds over the skipped calls, then a refusal if any remain - and the save.
How the calls are sent (`max_workers`, `retries`, `rerun_rounds`) has its defaults here and
nowhere below.
"""
from panel.llm import BaseLLMClient, run_llm_batch

from .effort import effort_tier
from .resolve import Resolved
from .store import ResultStore, RunPaths


def execute(
    resolved: Resolved,
    store: ResultStore,
    client: BaseLLMClient,
    max_workers: int = 10,
    retries: int = 3,
    rerun_rounds: int = 5,
    overwrite: bool = False,
) -> RunPaths:
    """
    Refused before any call is made: a stem that holds a run of a different spec or one brought in
    (`FileExistsError`), a store of another topic, a client that cannot say what schema it sends,
    an effort the model does not take (`ValueError`). Returns the plain paths; an archived copy is written beside them.

    A call that fails in a way no retry mends (`panel.llm.BatchFailed`: a wrong key, no credit, a
    rejected request) or in a way no client foresaw stops the run with that error; nothing is saved.
    """
    calls = resolved.calls
    spec = resolved.spec
    store.refuse(resolved, overwrite)
    client.schema_sent(spec.model, resolved.response_model)
    effort = effort_tier(client, spec)
    settings = dict(
        prompt_creator=resolved.prompt, response_model=resolved.response_model, model=spec.model,
        temperature=spec.temperature, client=client, instructions=resolved.framing.instructions,
        max_workers=max_workers, retries=retries,
    )
    results, skipped = run_llm_batch(calls, **settings)
    rounds, before = 0, -1
    while skipped and len(skipped) != before and rounds < rerun_rounds:
        before, rounds = len(skipped), rounds + 1
        more, skipped = run_llm_batch(calls.loc[skipped], **settings)
        results.update(more)
    if skipped:
        raise RuntimeError(
            f"{len(skipped)} of {len(calls)} calls unprocessed after {rounds} rerun rounds, e.g. calls {skipped[:10]}"
        )
    return store.save(resolved, results, client, overwrite=overwrite, effort=effort)
