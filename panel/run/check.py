"""
A saved run looked over once, from the saved files alone, so it holds for any run a store has.

Two parts: `check` asks whether the run is complete and each answer consistent, and `report`
words that for a person; `means` is the first reading of what the run predicts, as a table.
Nothing here compares against a reference - that is analysis.
"""
from collections import Counter
from dataclasses import dataclass
from typing import Dict, Tuple, Union

import pandas as pd

from panel.topic import Topic

from .resolve import REPEAT, SUBJECT, Resolved
from .store import ResultStore

MEAN, TOP = "mean", "top"
ZERO = 0.001


@dataclass(frozen=True)
class Sums:
    """How a group of probabilities that should sum to 1 did: answers off by more than the tolerance, and the range."""
    off: int
    low: float
    high: float


@dataclass(frozen=True)
class Check:
    """
    Attributes:
        stem: the run.
        calls: calls the run laid out.
        answers: answers saved.
        same_calls: whether the saved answers are exactly those calls.
        model: the model id asked for.
        model_resolved: the one the provider reported, or None.
        subjects: subjects the run covers.
        identical_prompts: subjects whose prompt is word for word another subject's.
        missing: empty cells among the topic's columns of the results table.
        tolerance: how far a sum may be from 1.
        out_of_range: probabilities below 0 or above 1.
        turnout: the turnout pair's sums; None for a topic without turnout.
        options: the option probabilities' sums; None for a pick.
        listed: options listed in an answer -> answers; None for a pick.
        duplicates: answers naming an option twice; None for a pick.
        identical: answers equal to another subject's in every column; None for a pick.
        abstained: answers giving going to vote a probability of zero; None without turnout.
        undefined: answers whose option probabilities sum to zero, so no preference can be
            read from them; None for a pick.
        both: answers that are both; an undefined answer outside this overlap is a subject
            who may vote but prefers nothing. None for a pick or without turnout.
        highest: mean of the highest option probability per answer; None for a pick.
    """
    stem: str
    calls: int
    answers: int
    same_calls: bool
    model: str
    model_resolved: Union[str, None]
    subjects: int
    identical_prompts: int
    missing: int
    tolerance: float
    out_of_range: int
    turnout: Union[Sums, None]
    options: Union[Sums, None]
    listed: Union[Dict[int, int], None]
    duplicates: Union[int, None]
    identical: Union[int, None]
    abstained: Union[int, None]
    undefined: Union[int, None]
    both: Union[int, None]
    highest: Union[float, None]


def _sums(totals: pd.Series, tolerance: float) -> Sums:
    return Sums(int(((totals - 1).abs() > tolerance).sum()), float(totals.min()), float(totals.max()))


def check(resolved: Resolved, store: ResultStore, tolerance: float = 0.01) -> Check:
    """
    Look over the saved run of a resolved spec.

    Raises:
        FileNotFoundError: if the store holds no such run.
    """
    stem, topic = resolved.stem, resolved.topic
    record = store.record(stem)
    loaded = store.load(stem, resolved.response_model)
    responses = [response for _, response in loaded.values()]
    frame = topic.rename(store.frame(stem))
    expected = {(int(subject), int(repeat)) for subject, repeat in zip(resolved.calls[SUBJECT], resolved.calls[REPEAT])}
    probabilities = "options" in resolved.response_model.model_fields
    options = list(topic.options)

    values = []
    if topic.turnout:
        values += [p for response in responses for p in (response.voted_or_not.voted, response.voted_or_not.not_voted)]
    if probabilities:
        values += [option.probability for response in responses for option in response.options]
        names = [[option.name for option in response.options] for response in responses]
    abstained = frame[topic.turnout[0]] <= ZERO if topic.turnout else None
    undefined = frame[options].sum(axis=1) <= ZERO if probabilities else None

    return Check(
        stem=stem,
        calls=len(resolved.calls),
        answers=len(responses),
        same_calls=set(loaded) == expected,
        model=resolved.spec.model,
        model_resolved=record.get("model_resolved"),
        subjects=len(resolved.rows),
        identical_prompts=int(resolved.rows.apply(resolved.prompt, axis=1).duplicated(keep=False).sum()),
        missing=int(frame[topic.columns].isna().sum().sum()),
        tolerance=tolerance,
        out_of_range=sum(not 0 <= value <= 1 for value in values),
        turnout=_sums(pd.Series([r.voted_or_not.voted + r.voted_or_not.not_voted for r in responses]), tolerance) if topic.turnout else None,
        options=_sums(pd.Series([sum(o.probability for o in r.options) for r in responses]), tolerance) if probabilities else None,
        listed=dict(sorted(Counter(len(set(listed)) for listed in names).items())) if probabilities else None,
        duplicates=sum(len(listed) != len(set(listed)) for listed in names) if probabilities else None,
        identical=int(frame.duplicated(subset=topic.columns, keep=False).sum()) if probabilities else None,
        abstained=int(abstained.sum()) if topic.turnout else None,
        undefined=int(undefined.sum()) if probabilities else None,
        both=int((abstained & undefined).sum()) if topic.turnout and probabilities else None,
        highest=float(frame[options].max(axis=1).mean()) if probabilities else None,
    )


def means(frame: pd.DataFrame, topic: Topic) -> pd.DataFrame:
    """
    What a run's answers say on average, as saved, not normalised and compared with nothing.

    Args:
        frame: a run's results table, one row per answer.
        topic: the topic it answered.

    Returns:
        One row per topic column: `mean` over the answers, and for an option `top`, the share
        of the answers with a preference in which it has the highest probability. For a pick
        both are the share of answers choosing the option.
    """
    options = list(topic.options)
    defined = frame.loc[frame[options].sum(axis=1) > ZERO, options]
    table = pd.DataFrame({MEAN: frame[topic.columns].mean()})
    table[TOP] = defined.idxmax(axis=1).value_counts(normalize=True).reindex(options, fill_value=0.0)
    return table


def report(result: Check, total: int) -> str:
    """
    A check as text, for a person.

    Args:
        result: from `check`.
        total: how many options the topic has, to say whether answers list them all.
    """
    rows: Tuple[Tuple[str, str], ...] = (
        ("answers", f"{result.answers} of {result.calls} calls, {'the same' if result.same_calls else 'NOT the same'} subjects"),
        ("model", f"{result.model} -> {result.model_resolved}"),
        ("identical prompts", f"{result.identical_prompts} of {result.subjects} subjects"),
        ("missing values", str(result.missing)),
        ("out of range", str(result.out_of_range)),
    )
    if result.turnout:
        rows += (
            ("turnout sum", _sums_text(result.turnout, result.tolerance)),
            ("certain abstainers", f"{_share(result.abstained, result.answers)} give going to vote a probability of 0"),
        )
    if result.options:
        fewer = {count: answers for count, answers in result.listed.items() if count != total}
        rows += (
            ("option sum", _sums_text(result.options, result.tolerance)),
            ("options listed", f"all {total} in {result.listed.get(total, 0)} answers" + (f"; fewer: {fewer}" if fewer else "")),
            ("duplicate options", str(result.duplicates)),
            ("identical answers", str(result.identical)),
            ("no preference", f"{_share(result.undefined, result.answers)} have options summing to 0"),
            ("highest option", f"{result.highest:.3f} on average"),
        )
    if result.both is not None:
        rows += (("overlap", f"{result.both} are both, {result.abstained - result.both} abstain with a preference, {result.undefined - result.both} may vote with none"),)
    return "\n".join([result.stem] + [f"  {label:<18} {text}" for label, text in rows])


def _share(count: int, total: int) -> str:
    return f"{count} ({count / total:.1%})"


def _sums_text(sums: Sums, tolerance: float) -> str:
    return f"{sums.off} off by more than {tolerance} ({sums.low:.3f} .. {sums.high:.3f})"
