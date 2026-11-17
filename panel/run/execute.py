"""
Send a resolved run and save what comes back: the one loop that turns a spec into files.

The batch runner handles per-call retries and keeps the best of the flawed answers
(`panel.run.accept`); this module handles what is left after a batch - capped rerun rounds
over the calls without a usable answer - and the save. What came back is always saved: a run
with a call missing goes under `<stem>_incomplete`, never under the stem.
How the calls are sent (`max_workers`, `retries`, `rerun_rounds`) has its defaults here and
nowhere below.

An answer read off the token distribution is not judged: its sums are 1 by construction, and
asking again gives the same probabilities, so there is nothing a second attempt could mend.
"""
from collections import Counter
from functools import partial
from typing import Any, Dict, Mapping, Sequence

from panel.llm import FAILED, RATE_LIMITED, BaseLLMClient, BatchStopped, run_llm_batch

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
    missing: bool = False,
) -> RunPaths:
    """
    `missing` sends only the calls the run's incomplete sending (`<stem>_incomplete`) lacks and saves
    the whole under the stem; the record's `resumed` block says how many answers are the earlier ones
    and, per earlier sending, when and by which model version they were given and what that sending
    took. Without such a run it raises `FileNotFoundError`.

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
        review=None if client.READS_TOKENS else partial(judge, low=spec.sum_low, high=spec.sum_high), max_workers=max_workers, retries=retries,
    )
    results, flawed, attempts, resumed = {}, {}, {}, None
    if missing:
        results, flawed, earlier = store.answered(resolved)
        before_it = earlier.get("resumed", {}).get("sendings", [])
        this = {key: earlier.get(key) for key in ("saved_at", "model_resolved", "sending", "attempts")}
        resumed = {"answers": len(results), "sendings": before_it + [this]}
    sent = [call for call in calls.index if call not in results]
    skipped, stopped = sent, None
    rounds, before = -1, -1
    while skipped and len(skipped) != before and rounds < rerun_rounds:
        before, rounds = (len(skipped) if rounds >= 0 else -1), rounds + 1
        try:
            batch = run_llm_batch(calls.loc[skipped], **settings)
        except BatchStopped as stop:
            batch, stopped = stop.batch, stop
        results.update(batch.results)
        flawed.update(batch.flawed)
        for call, tried in batch.attempts.items():
            attempts[call] = attempts.get(call, ()) + tried
        skipped = batch.skipped
        if stopped:
            break
    accepted = [call for call in sent if call in results and call not in flawed]
    sending = {
        **_totals(attempts, len(accepted)), "answered": sum(call in results for call in sent), "rerun_rounds": max(rounds, 0),
        "settings": {"max_workers": max_workers, "retries": retries, "rerun_rounds": rerun_rounds},
    }
    written = store.save(
        resolved, results, client, overwrite=overwrite, effort=effort, flawed=flawed, resumed=resumed, sending=sending, attempts=attempts,
    ) if results else None
    kept = f"what came back is saved as '{written.stem}'" if written else "nothing came back, nothing is saved"
    if stopped:
        print(f"the run stopped with {len(results)} of {len(calls)} answers; {kept}")
        raise stopped.__cause__
    if len(results) < len(calls):
        raise RuntimeError(f"{len(calls) - len(results)} of {len(calls)} calls have no usable answer after {rounds} rerun rounds; {kept}")
    return written


def _totals(attempts: Mapping[int, Sequence[str]], accepted: int) -> Dict[str, Any]:
    """
    `calls_sent` is every attempt: those that gave no accepted answer and one per accepted answer.
    `refused` counts attempts per flaw; an attempt with two flaws is in both.
    """
    words = [word for tried in attempts.values() for word in tried]
    flaws = Counter(reason for word in words if word not in (FAILED, RATE_LIMITED) for reason in word.split("+"))
    return {
        "calls_sent": len(words) + accepted, "refused": dict(sorted(flaws.items())),
        "failed": words.count(FAILED), "rate_limited": words.count(RATE_LIMITED),
    }
