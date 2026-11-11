"""
Does the model place the options in a space that matches where experts place them?

Two options are close for the model when it raises and lowers them for the same subjects:
their distance is sqrt(2(1 - r)), r being the correlation of the two options' shares over
subjects. These distances are drawn as a two-dimensional map (MDS), the map is laid onto the
experts' plane (Procrustes), and the agreement is tested over every relabelling of the
options. Only the options the experts rate take part.

The fit has many freedoms and few points, so the correlation alone flatters; the p-values are
what to read. The colleague's analysis from `llm_pushforward` (`ches_comparison`), its
correlation distance.
"""
from typing import List, Tuple

import numpy as np
import pandas as pd
from matplotlib.figure import Figure

from .aggregate import subject_answers
from .output import Output
from .plots.maps import maps
from .saved import NotApplicable, SavedRun
from .stats.mapping import mantel, mds, procrustes, procrustes_test

NAME = "compass"
FIT = "compass_fit"
STRESS, R, R_P = "MDS stress-1", "Procrustes r", "Procrustes p"
PLANE, PLANE_P, ITEMS, ITEMS_P = "Spearman vs plane", "Spearman vs plane p", "Spearman vs all items", "Spearman vs all items p"


def _between(points: np.ndarray) -> np.ndarray:
    return np.linalg.norm(points[:, None] - points[None], axis=-1)


def option_distances(run: SavedRun, options: List[str]) -> pd.DataFrame:
    """The correlation distance between the named options over the run's subjects."""
    shares = subject_answers(run.normalized, run.topic)[options]
    return np.sqrt((2 * (1 - shares.corr())).clip(lower=0))


def compass(run: SavedRun, out: Output, seed: int = 0) -> Tuple[pd.DataFrame, List[Figure]]:
    """
    The model's map of the options laid onto the experts' plane, and how well they agree.

    Args:
        run: the saved run.
        out: takes the table and the figure, both named `compass`, and a second table
            `compass_fit`: the map's stress, the Procrustes correlation, and the Spearman
            correlation of the model's distances with the experts' in the plane and over
            all items, each with its exact permutation p-value.
        seed: of the MDS starts.

    Returns:
        The table - one row per rated option: where the model's map puts it in the plane
        and where the experts do - and its figure.

    Raises:
        NotApplicable: if the topic has no expert positions.
    """
    positions = run.positions
    if positions is None:
        raise NotApplicable(f"'{run.stem}': topic '{run.topic.name}' has no expert positions")
    options, (x, y) = list(positions.table.index), positions.axes
    distances = option_distances(run, options).to_numpy()
    plane = positions.plane.to_numpy()
    points, stress = mds(distances, seed)
    fitted, _ = procrustes(plane, points)
    r, r_p = procrustes_test(plane, points)
    flat, flat_p = mantel(distances, _between(plane))
    full, full_p = mantel(distances, _between(positions.standardized.to_numpy()))

    table = pd.DataFrame(fitted, index=options, columns=[f"model {x}", f"model {y}"]).join(positions.plane.add_prefix("experts "))
    out.table(FIT, pd.DataFrame({run.stem: {STRESS: stress, R: r, R_P: r_p, PLANE: flat, PLANE_P: flat_p, ITEMS: full, ITEMS_P: full_p}}).T)
    out.table(NAME, table)

    figure = maps(
        {f"Procrustes r = {r:.2f} (p = {r_p:.3f})": pd.DataFrame(fitted, index=options)}, positions.plane, (0, 10),
        positions.labels[x], positions.labels[y], ("the model's map,\nlaid onto the plane", "experts"),
        run.style.colors, run.labels,
        title=f"{run.spec.model} on {run.topic.name}: the model's map of the options",
        note=f"MDS of the options' correlation distances over {len(run.subjects)} subjects (stress {stress:.2f}),\nshifted, turned and rescaled to fit the experts' positions",
    )
    return table, [out.figure(NAME, figure)]
