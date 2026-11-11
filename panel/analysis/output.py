"""
Where an analysis leaves what it made, and whether it leaves anything.

An analysis hands every table and figure to an `Output` and gets it back, so the same code
serves a script that saves everything and a notebook that only looks. Tables go to
`results/<topic>/analysis/<name>_<stem>.csv`, figures to
`results/<topic>/plots/<name>_<stem>.png`; both are regenerable.
"""
from pathlib import Path

import pandas as pd
from matplotlib.figure import Figure

from panel.run import ResultStore


class Output:
    """
    The derived tables and figures of one run, or of one named comparison of runs (`stem` is then its
    name); with `save` off nothing touches the disk. `written` lists what was saved.
    """

    def __init__(self, store: ResultStore, stem: str, save: bool = True, dpi: int = 150):
        self.store = store
        self.stem = stem
        self.save = save
        self.dpi = dpi
        self.written = []

    def table_path(self, name: str) -> Path:
        return self.store.analysis(f"{name}_{self.stem}.csv")

    def figure_path(self, name: str) -> Path:
        return self.store.plot(f"{name}_{self.stem}.png")

    def table(self, name: str, frame: pd.DataFrame) -> pd.DataFrame:
        """The frame, written as `<name>_<stem>.csv` when saving is on."""
        if self.save:
            frame.to_csv(self.table_path(name), encoding="utf-8")
            self.written.append(self.table_path(name))
        return frame

    def figure(self, name: str, figure: Figure) -> Figure:
        """The figure, written as `<name>_<stem>.png` when saving is on."""
        if self.save:
            figure.savefig(self.figure_path(name), dpi=self.dpi, bbox_inches="tight")
            self.written.append(self.figure_path(name))
        return figure
