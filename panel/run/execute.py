"""
Send a resolved run and save what comes back: the one loop that turns a spec into files.

The batch runner handles per-call retries and keeps the best of the flawed answers
(`panel.run.accept`); this module handles what is left after a batch - capped rerun rounds
over the calls without a usable answer - and the save. What came back is always saved: a run
with a call missing goes under `<stem>_incomplete`, never under the stem.
How the calls are sent (`max_workers`, `retries`, `rerun_rounds`) has its defaults here and
nowhere below.
"""
from functools import partial

from panel.llm import BaseLLMClient, BatchStopped, run_llm_batch

from .accept import judge
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

    Raised after saving what came back as the incomplete run, if anything did: `RuntimeError` when
    calls are left without a usable answer after the rerun rounds; whatever stopped the batch when
    a call failed in a way no retry mends (`panel.llm.BatchFailed`: a wrong key, no credit, a
    rejected request), in a way no client foresaw, or the run was interrupted.
    """
    calls = resolved.calls
    spec = resolved.spec
    store.refuse(resolved, overwrite)
    client.schema_sent(spec.model, resolved.response_model)
    effort = effort_tier(client, spec)
    settings = dict(
        prompt_creator=resolved.prompt, response_model=resolved.response_model, model=spec.model,
        temperature=spec.temperature, client=client, instructions=resolved.framing.instructions,
        review=partial(judge, low=spec.sum_low, high=spec.sum_high), max_workers=max_workers, retries=retries,
    )
    results, flawed, skipped, stopped = {}, {}, list(calls.index), None
    rounds, before = -1, -1
    while skipped and len(skipped) != before and rounds < rerun_rounds:
        before, rounds = (len(skipped) if rounds >= 0 else -1), rounds + 1
        try:
            more, more_flawed, skipped = run_llm_batch(calls.loc[skipped], **settings)
        except BatchStopped as stop:
            more, more_flawed, stopped = stop.results, stop.flawed, stop
        results.update(more)
        flawed.update(more_flawed)
        if stopped:
            break
    written = store.save(resolved, results, client, overwrite=overwrite, effort=effort, flawed=flawed) if results else None
    kept = f"what came back is saved as '{written.stem}'" if written else "nothing came back, nothing is saved"
    if stopped:
        print(f"the run stopped with {len(results)} of {len(calls)} answers; {kept}")
        raise stopped.__cause__
    if len(results) < len(calls):
        raise RuntimeError(f"{len(calls) - len(results)} of {len(calls)} calls have no usable answer after {rounds} rerun rounds; {kept}")
    return written
