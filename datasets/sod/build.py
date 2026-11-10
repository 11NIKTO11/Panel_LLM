"""The survey's values: the raw `.sav` named in `dataset.json`, one row per respondent through `clean`."""
import pandas as pd

from panel.dataset import Dataset

from .clean import process_respondent


def build(dataset: Dataset) -> pd.DataFrame:
    """One values row per respondent, keyed by the survey's own id."""
    raw = pd.read_spss(dataset.dir / dataset.source)
    return pd.DataFrame([process_respondent(raw.loc[i]) for i in raw.index]).set_index("id")
