"""
Send a resolved run and save what comes back: the one loop that turns a spec into files.

The batch runner handles per-call retries; this module handles what is left after a batch -
capped rerun rounds over the skipped calls, then a refusal if any remain - and the double save:
an archived copy first, then the plain files a stem is loaded by.
"""
from panel.llm import BaseLLMClient, run_llm_batch

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
    Run every call of a resolved spec and save the results under its stem.

    Args:
        resolved: the run, from `resolve`.
        store: the results folder of the run's topic.
        client: makes the calls; constructed by the caller, which is where keys live.
        max_workers: parallel calls.
        retries: attempts per call within one batch.
        rerun_rounds: how many times the skipped calls are batched again.
        overwrite: replace a run of a different spec that already holds the stem.

    Returns:
        The plain paths written (the archived copy is written first).

    Raises:
        RuntimeError: if calls are still unprocessed after the rerun rounds.
        FileExistsError: if the stem already holds a run of a different spec.
        ValueError: if the store belongs to another topic.
    """
    calls = resolved.calls
    spec = resolved.spec
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
    store.save(resolved, results, client, timestamped=True, overwrite=overwrite)
    return store.save(resolved, results, client, timestamped=False, overwrite=overwrite)
