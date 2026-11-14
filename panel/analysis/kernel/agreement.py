"""
Two kernels held against each other profile by profile, on the profiles both have - matched by
their values, so runs of one dataset (two models, two wordings, two languages) pair up.

- distance: the TV between the two answers to the same profile;
- direction: per outcome, the correlation over the profiles - whether the run raises and lowers
  it for the same profiles as the reference;
- size: per outcome, the ratio of the standard deviations over the profiles; above 1 the run
  spreads its profiles further apart than the reference does, which `llm_pushforward` calls
  stereotype amplification.

The reference is a kernel of the set, or the model trained on the subjects' own answers
(`benchmark`'s, given the run's factors), grouped into the run's profiles - the profiles of
subjects with a known answer.
"""
from typing import List, Tuple, Union

import numpy as np
import pandas as pd
from matplotlib.figure import Figure

from panel.run.resolve import REPEAT, SUBJECT

from ..benchmark.ground import TRAINED, load_ground
from ..comparison import RunSet
from ..output import Output
from ..plots.distributions import overlaid
from ..plots.matrices import grid
from ..saved import NotApplicable
from ..stats.distributions import tv
from .matrix import PROFILE, Kernel, KernelSet, assembled

NAME, OUTCOMES = "kernel_agreement", "kernel_agreement_outcomes"


def trained_kernel(kernels: KernelSet) -> Kernel:
    """
    The model trained on the subjects' own answers, given the first kernel's factors, as a kernel
    over that kernel's profiles.

    Raises:
        NotApplicable: if the runs are of different datasets or theirs has no answers to the topic.
    """
    first = next(iter(kernels.kernels.values()))
    runs = RunSet(topic=kernels.topic, runs={label: kernel.run for label, kernel in kernels.kernels.items()}, polls=pd.DataFrame(), style=kernels.style)
    trained = load_ground(runs, predictors=list(first.factors.columns)).trained
    profiles = first.calls.drop_duplicates(SUBJECT).set_index(SUBJECT)[PROFILE]
    calls = trained.reindex(columns=first.outcomes).assign(**{PROFILE: profiles.reindex(trained.index).to_numpy(), REPEAT: 1})
    calls = calls.rename_axis(SUBJECT).reset_index()[[PROFILE, SUBJECT, REPEAT, *first.outcomes]]
    found = assembled(first, calls, first.factors, designed=False)
    return Kernel(label=TRAINED, run=first.run, matrix=found.matrix, factors=found.factors, weights=found.weights, calls=calls, designed=False)


def _by_values(kernel: Kernel) -> pd.DataFrame:
    columns = kernel.matrix.T
    columns.index = pd.MultiIndex.from_frame(kernel.factors.loc[columns.index])
    return columns


def agreement(kernels: KernelSet, out: Output, reference: Union[str, None] = None) -> Tuple[pd.DataFrame, List[Figure]]:
    """
    Every kernel against the reference on the profiles they share.

    Args:
        kernels: the kernels.
        out: takes the tables and the figures.
        reference: a kernel's label, or `trained on claims`; the trained model when the runs'
            dataset has answers to the topic, else the first kernel.

    Returns:
        The table - a row per kernel: shared profiles, the mean, median and 90th percentile of
        their TV, the TV of the two mean answers, the mean correlation and the median ratio of
        standard deviations over the outcomes - with the outcomes' table beside it
        (`kernel_agreement_outcomes`), and the figures: the TV per profile, and per outcome
        the correlations and the ratios.

    Raises:
        NotApplicable: if no kernel shares a profile with the reference.
        KeyError: on a reference that is neither a label nor the trained model.
    """
    first = next(iter(kernels.kernels.values()))
    if reference is None:
        reference = TRAINED if first.run.claims and len({kernel.run.dataset.name for kernel in kernels.kernels.values()}) == 1 else next(iter(kernels.kernels))
    if reference == TRAINED:
        held = trained_kernel(kernels)
    elif reference in kernels.kernels:
        held = kernels.kernels[reference]
    else:
        raise KeyError(f"no kernel '{reference}' to hold the others against; there are {list(kernels.kernels)} and '{TRAINED}'")
    against = _by_values(held)
    rows, outcomes, distances = {}, {}, {}
    for label, kernel in kernels.kernels.items():
        if label == reference or list(kernel.factors.columns) != list(held.factors.columns):
            continue
        mine = _by_values(kernel)
        shared = mine.index.intersection(against.index)
        if len(shared) < 3:
            continue
        a, b = mine.loc[shared, held.outcomes].to_numpy(), against.loc[shared, held.outcomes].to_numpy()
        distance = tv(a, b)
        with np.errstate(invalid="ignore", divide="ignore"):
            correlation = np.array([np.corrcoef(a[:, k], b[:, k])[0, 1] for k in range(a.shape[1])])
            ratio = a.std(axis=0) / b.std(axis=0)
        rows[label] = {
            "shared profiles": len(shared), "mean TV": float(distance.mean()), "median TV": float(np.median(distance)),
            "q90 TV": float(np.quantile(distance, 0.9)), "TV of mean answers": tv(a.mean(axis=0), b.mean(axis=0)),
            "mean correlation": float(np.nanmean(correlation)), "median std ratio": float(np.nanmedian(ratio)),
        }
        outcomes[label] = pd.DataFrame({"correlation": correlation, "std ratio": ratio}, index=held.outcomes)
        distances[label] = distance
    if not rows:
        raise NotApplicable(f"no kernel shares three profiles or more with '{reference}'")
    table = out.table(NAME, pd.DataFrame(rows).T.rename_axis("kernel"))
    out.table(OUTCOMES, pd.concat(outcomes, names=["kernel", "outcome"]))
    correlations = pd.DataFrame({label: frame["correlation"] for label, frame in outcomes.items()}).rename(index=kernels.labels)
    ratios = pd.DataFrame({label: np.log2(frame["std ratio"]) for label, frame in outcomes.items()}).rename(index=kernels.labels)
    figures = [
        out.figure(NAME, overlaid({f"against {reference}": distances}, xlabel="TV between the two answers to the same profile",
                                  title=f"{kernels.topic.name}: how far each kernel is from {reference}, profile by profile")),
        out.figure(f"{OUTCOMES}_correlation", grid(correlations, decimals=2, signed=True, cmap="Greens",
                                                  title=f"{kernels.topic.name}: does each kernel move an outcome for the same profiles as {reference}?",
                                                  note="Pearson correlation over the shared profiles")),
        out.figure(f"{OUTCOMES}_ratio", grid(ratios, decimals=1, signed=True, cmap="Oranges",
                                            title=f"{kernels.topic.name}: how much more each kernel spreads an outcome than {reference}",
                                            note="log2 of the ratio of standard deviations over the shared profiles: +1 is twice the spread, -1 half")),
    ]
    return table, figures
