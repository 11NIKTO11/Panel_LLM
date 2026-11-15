"""
Where an analysis leaves what it made, and whether it leaves anything.

An analysis hands every table and figure to an `Output` and gets it back, so the same code
serves a script that saves everything and a notebook that only looks. An output writes into
one folder of the results - a run's, or a named comparison's (`ResultStore.run_dir`,
`comparison_dir`): tables to `<folder>/analysis/<name>.csv`, figures to
`<folder>/plots/<name>.png`; both are regenerable.

An analysis is two halves - `compute(run, ...)` makes its tables, `draw(run, tables, ...)` its
figures - and a public function that hands both to an output through `save`. Whoever wants the
numbers alone, as a comparison of runs does, calls `compute` and no output is involved.
"""
from pathlib import Path
from typing import List, Mapping, Tuple, Union

import pandas as pd
from matplotlib.figure import Figure

from panel.run import ResultStore


class Output:
    """
    The derived tables and figures of one run, or of one named comparison of runs; with `save`
    off nothing touches the disk. `written` lists what was saved.

    Args:
        folder: the run's or the comparison's folder; may be None when nothing is saved.
        save: write what is handed in.
        dpi: of the figures.
        suffix: added to every file name, for a variant kept beside the plain output.
    """

    def __init__(self, folder: Union[Path, None], save: bool = True, dpi: int = 150, suffix: str = ""):
        self.folder = folder
        self.save = save
        self.dpi = dpi
        self.suffix = suffix
        self.written = []

    def table_path(self, name: str) -> Path:
        return self.folder / ResultStore.ANALYSIS_DIR / f"{name}{self.suffix}.csv"

    def figure_path(self, name: str) -> Path:
        return self.folder / ResultStore.PLOTS_DIR / f"{name}{self.suffix}.png"

    def table(self, name: str, frame: pd.DataFrame) -> pd.DataFrame:
        """The frame, written as `analysis/<name>.csv` when saving is on."""
        if self.save:
            path = self.table_path(name)
            path.parent.mkdir(parents=True, exist_ok=True)
            frame.to_csv(path, encoding="utf-8")
            self.written.append(path)
        return frame

    def figure(self, name: str, figure: Figure) -> Figure:
        """The figure, written as `plots/<name>.png` when saving is on."""
        if self.save:
            path = self.figure_path(name)
            path.parent.mkdir(parents=True, exist_ok=True)
            figure.savefig(path, dpi=self.dpi, bbox_inches="tight")
            self.written.append(path)
        return figure


def save(out: Output, tables: Mapping[str, pd.DataFrame], figures: Mapping[str, Figure]) -> Tuple[pd.DataFrame, List[Figure]]:
    """
    Hand an analysis's tables and figures to an output, each under its name.

    Returns:
        The first table - the analysis's main one - and the figures, in their order.
    """
    written = [out.table(name, table) for name, table in tables.items()]
    return written[0], [out.figure(name, figure) for name, figure in figures.items()]
