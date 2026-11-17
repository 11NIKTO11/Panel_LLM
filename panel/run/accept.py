"""
Whether an answer that fits the schema is one a run accepts.

The schema cannot say that probabilities sum to 1 or that no option is listed twice, so a model
can answer within it and still answer badly. Each answer is judged once it is parsed:

- accepted: every group of probabilities sums to within the run's bounds, no option twice.
- flawed but usable: a sum outside the bounds, or a subject certain not to vote who is given
  no preference. The call is tried again; if no attempt does better the best is kept and the
  record marks it, because how badly a model answers is a result too.
- unusable: an option listed twice, a turnout summing to zero, or no preference for a subject
  who may vote - nothing a table can be made of. Tried again, and never kept.

The bounds are the run's (`RunSpec.sum_low`, `sum_high`); a sum of zero is refused whatever they are.
"""
from typing import Union

from panel.llm import Flaw
from panel.topic.response import TopicResponse

ZERO = 0.001
TURNOUT_SUM, TURNOUT_ZERO = "turnout_sum", "turnout_zero"
OPTIONS_SUM, OPTIONS_ZERO, NO_PREFERENCE, DUPLICATES = "options_sum", "options_zero", "no_preference", "duplicates"
UNUSABLE = (TURNOUT_ZERO, OPTIONS_ZERO, DUPLICATES)


def judge(answer: TopicResponse, low: Union[float, None], high: Union[float, None]) -> Union[Flaw, None]:
    """
    None for an accepted answer. `low` / `high` bound each sum, None leaving that side open.
    A flaw's `distance` is how far the sums are from 1, added up; of two flawed answers the nearer is the better.
    """
    def outside(total: float) -> bool:
        return (low is not None and total < low) or (high is not None and total > high)

    reasons, distance, stays_home = [], 0.0, False
    if answer.TURNOUT:
        voted, total = answer.voted_or_not.voted, answer.voted_or_not.voted + answer.voted_or_not.not_voted
        stays_home = voted <= ZERO < total
        if total <= ZERO:
            reasons.append(TURNOUT_ZERO)
        elif outside(total):
            reasons.append(TURNOUT_SUM)
        distance += abs(total - 1)
    if "options" in type(answer).model_fields:
        names = [option.name for option in answer.options]
        total = sum(option.probability for option in answer.options)
        if len(names) != len(set(names)):
            reasons.append(DUPLICATES)
        if total <= ZERO:
            reasons.append(NO_PREFERENCE if stays_home else OPTIONS_ZERO)
        elif outside(total):
            reasons.append(OPTIONS_SUM)
        distance += abs(total - 1)
    if not reasons:
        return None
    return Flaw(reasons=tuple(reasons), usable=not set(reasons) & set(UNUSABLE), distance=distance)
