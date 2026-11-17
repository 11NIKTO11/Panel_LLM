"""
A saved run looked over once, from the saved files alone, so it holds for any run a store has.
Nothing here compares against a reference - that is analysis.
"""
from collections import Counter
from dataclasses import dataclass
from typing import Dict, Tuple, Union

import pandas as pd

from panel.topic import Framing, Topic, make_response_models

from .accept import ZERO
from .store import ResultStore

MEAN, TOP = "mean", "top"


@dataclass(frozen=True)
class Sums:
    """How a group of probabilities that should sum to 1 did: answers off by more than the tolerance, and the range."""
    off: int
    low: float
    high: float


@dataclass(frozen=True)
class Distinct:
    """How many different answers a run holds - two answers are the same when every column matches: how many
    there are, how many of them one answer alone gave, and how many answers gave the most common one."""
    answers: int
    once: int
    commonest: int


@dataclass(frozen=True)
class Check:
    """
    None for a topic without turnout: `turnout`, `abstained`, `both`. None for a pick: `options`, `listed`,
    `duplicates`, `distinct`, `undefined`, `both`, `highest`.

    calls: the calls the run laid out, its subjects times its repeats.
    same_calls: whether the saved answers are exactly those calls.
    model_resolved: the model id the provider reported, or None.
    identical_prompts: subjects whose saved prompt is word for word another subject's.
    missing: empty cells among the topic's columns of the results table.
    out_of_range: probabilities below 0 or above 1.
    listed: options listed in an answer -> answers.
    distinct: the different answers among all, over subjects and repeats alike.
    abstained: answers giving going to vote a probability of zero.
    undefined: answers whose option probabilities sum to zero, so no preference can be read from them.
    both: answers that are both; an undefined answer outside this overlap is a subject who may vote
        but prefers nothing.
    highest: mean of the highest option probability per answer.
    faulty: reason -> answers kept though flawed in that way (`panel.run.accept`); None for a run saved
        before answers were judged.
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
    distinct: Union[Distinct, None]
    abstained: Union[int, None]
    undefined: Union[int, None]
    both: Union[int, None]
    highest: Union[float, None]
    faulty: Union[Dict[str, int], None] = None


def _sums(totals: pd.Series, tolerance: float) -> Sums:
    return Sums(int(((totals - 1).abs() > tolerance).sum()), float(totals.min()), float(totals.max()))


def check(store: ResultStore, stem: str, topic: Topic, framing: Framing, tolerance: float = 0.01) -> Check:
    """
    Nothing of the dataset is read: the saved answers are parsed by the response model of `topic` and
    `framing`. A stem the store does not hold raises `FileNotFoundError` in `store.record`.
    """
    record = store.record(stem)
    spec, subjects = record["spec"], record["dataset"]["rows"]
    response_model = {model.SHORT_NAME: model for model in make_response_models(topic, framing)}[spec["shape"]]
    loaded = store.load(stem, response_model)
    responses = [response for _, response in loaded.values()]
    frame = topic.rename(store.frame(stem))
    expected = {(int(subject), repeat) for subject in subjects for repeat in range(1, spec["repeats"] + 1)}
    prompts = {}
    for (subject, _), (prompt, _) in sorted(loaded.items()):
        prompts.setdefault(subject, prompt)

    values = []
    turnout = abstained = None
    options = listed = duplicates = distinct = undefined = highest = both = None

    if topic.turnout:
        pairs = [(response.voted_or_not.voted, response.voted_or_not.not_voted) for response in responses]
        values += [value for pair in pairs for value in pair]
        turnout = _sums(pd.Series([sum(pair) for pair in pairs]), tolerance)
        stays_home = frame[topic.turnout[0]] <= ZERO
        abstained = int(stays_home.sum())

    if "options" in response_model.model_fields:
        columns = list(topic.options)
        names = [[option.name for option in response.options] for response in responses]
        sums = [sum(option.probability for option in response.options) for response in responses]
        values += [option.probability for response in responses for option in response.options]
        options = _sums(pd.Series(sums), tolerance)
        listed = dict(sorted(Counter(len(set(listing)) for listing in names).items()))
        duplicates = sum(len(listing) != len(set(listing)) for listing in names)
        given = frame.groupby(topic.columns, dropna=False).size()
        distinct = Distinct(len(given), int((given == 1).sum()), int(given.max()))
        prefers_nothing = frame[columns].sum(axis=1) <= ZERO
        undefined = int(prefers_nothing.sum())
        highest = float(frame[columns].max(axis=1).mean())
        if topic.turnout:
            both = int((stays_home & prefers_nothing).sum())

    return Check(
        stem=stem,
        calls=len(expected),
        answers=len(responses),
        same_calls=set(loaded) == expected,
        model=spec["model"],
        model_resolved=record.get("model_resolved"),
        subjects=len(subjects),
        identical_prompts=int(pd.Series(prompts, dtype=object).duplicated(keep=False).sum()),
        missing=int(frame[topic.columns].isna().sum().sum()),
        tolerance=tolerance,
        out_of_range=sum(not 0 <= value <= 1 for value in values),
        turnout=turnout, abstained=abstained,
        options=options, listed=listed, duplicates=duplicates, distinct=distinct, undefined=undefined, highest=highest,
        both=both,
        faulty=dict(Counter(reason for reasons in record["faulty"].values() for reason in reasons)) if "faulty" in record else None,
    )


def means(frame: pd.DataFrame, topic: Topic) -> pd.DataFrame:
    """
    One row per topic column, as saved, not normalised: `mean` over the answers and, for an option, `top`,
    the share of the answers with a preference in which it has the highest probability. For a pick both
    are the share of answers choosing the option.
    """
    options = list(topic.options)
    defined = frame.loc[frame[options].sum(axis=1) > ZERO, options]
    table = pd.DataFrame({MEAN: frame[topic.columns].mean()})
    table[TOP] = defined.idxmax(axis=1).value_counts(normalize=True).reindex(options, fill_value=0.0)
    return table


def report(result: Check, total: int) -> str:
    """`total` is how many options the topic has, to say whether answers list them all."""
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
            ("distinct answers", f"{_share(result.distinct.answers, result.answers)} of {result.answers}; {result.distinct.once} given once, the most common {result.distinct.commonest} times"),
            ("no preference", f"{_share(result.undefined, result.answers)} have options summing to 0"),
            ("highest option", f"{result.highest:.3f} on average"),
        )
    if result.faulty is not None:
        kinds = ", ".join(f"{reason} {count}" for reason, count in sorted(result.faulty.items()))
        rows += (("kept though flawed", kinds or "0"),)
    if result.both is not None:
        rows += (("overlap", f"{result.both} are both, {result.abstained - result.both} abstain with a preference, {result.undefined - result.both} may vote with none"),)
    return "\n".join([result.stem] + [f"  {label:<18} {text}" for label, text in rows])


def _share(count: int, total: int) -> str:
    return f"{count} ({count / total:.1%})"


def _sums_text(sums: Sums, tolerance: float) -> str:
    return f"{sums.off} off by more than {tolerance} ({sums.low:.3f} .. {sums.high:.3f})"
