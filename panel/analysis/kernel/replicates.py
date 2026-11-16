"""
Where a kernel's replicates come from, and the replicates themselves.

- a sampled kernel: half-samples of its profiles, which are a draw from a population;
- a designed kernel with repeats: its repeats resampled - its cells are fixed, its randomness
  is the model's;
- a designed kernel without repeats, or of delivered means: none.
"""
from typing import List, Tuple, Union

import numpy as np

from ..stats.replicates import halves
from .matrix import Kernel

HALF_SAMPLES, RESAMPLED = "half-samples of the profiles", "resampled repeats"


def scheme_of(kernel: Kernel) -> Union[str, None]:
    if not kernel.designed:
        return HALF_SAMPLES
    return RESAMPLED if kernel.repeats > 1 else None


def replica_positions(kernel: Kernel, draws: int, seed: int = 0) -> List[np.ndarray]:
    """For half-sampling: the profiles' positions of each replicate; the same for every kernel of as many profiles, so replicates pair."""
    return halves(kernel.matrix.shape[1], draws, seed)


def replicas(kernel: Kernel, draws: int, seed: int = 0) -> List[Kernel]:
    scheme = scheme_of(kernel)
    if scheme == HALF_SAMPLES:
        return [kernel.subset(positions) for positions in replica_positions(kernel, draws, seed)]
    if scheme == RESAMPLED:
        random = np.random.default_rng(seed)
        return [kernel.resampled(random) for _ in range(draws)]
    return []


def replica_answers(kernel: Kernel, draws: int, seed: int = 0) -> List[Tuple[Union[np.ndarray, slice], np.ndarray]]:
    """
    Each replicate as (which of the kernel's profiles it holds, their answers, profiles x
    outcomes), for a statistic that keeps the kernel's own design: a half-sample holds half the
    profiles, by position, with their answers as they are; a resampling all of them, as a
    slice, with answers drawn again.
    """
    scheme, values = scheme_of(kernel), kernel.matrix.to_numpy().T
    if scheme == HALF_SAMPLES:
        return [(positions, values[positions]) for positions in replica_positions(kernel, draws, seed)]
    if scheme == RESAMPLED:
        random, everyone = np.random.default_rng(seed), slice(None)
        return [(everyone, kernel.resampled(random).matrix.to_numpy().T) for _ in range(draws)]
    return []
