"""
Shapley effects: the variance a set of factors explains, split fairly among them.

The value of a set of players is the leave-one-out R² of a ridge regression on their design
columns (`stats.variance.loo_r2`) - out of sample, so a player cannot gain by fitting noise. A
player's effect is its added value averaged over every order in which the players could join
(Owen 2014; Song, Nelson & Staum 2016), computed exactly over all 2^p subsets. The effects sum
to the value of all players together. A player may hold several columns that should share
credit, e.g. region and district as one "geography".
"""
import itertools
import math
from typing import Mapping, Tuple

import numpy as np
import pandas as pd

from .variance import loo_r2


def shapley_values(players: Mapping[str, np.ndarray], values: np.ndarray) -> Tuple[pd.Series, float]:
    """
    (the effects by player, the value of all players together); `players` is name -> its design
    columns, rows aligned with `values`, n x k or n.
    """
    names = list(players)
    count = len(names)
    if not count:
        raise ValueError("Shapley effects need at least one player")
    worth = {(): 0.0}
    for size in range(1, count + 1):
        for subset in itertools.combinations(range(count), size):
            worth[subset] = loo_r2(np.hstack([players[names[i]] for i in subset]), values)
    weight = [math.factorial(size) * math.factorial(count - size - 1) / math.factorial(count) for size in range(count)]
    effects = np.zeros(count)
    for subset, value in worth.items():
        for joining in set(range(count)) - set(subset):
            effects[joining] += weight[len(subset)] * (worth[tuple(sorted(subset + (joining,)))] - value)
    return pd.Series(effects, index=names), worth[tuple(range(count))]
