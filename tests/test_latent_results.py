"""The place-level tables of 2025 brought in as runs: one row per area, the numbers of the table kept beside them."""
import pandas as pd
import pytest

from panel.analysis import load_run
from panel.run import ResultStore, RunSpec, resolve
from panel.run.record import DELIVERED

from tests.conftest import RESULTS_DIR, require

MODELS = ("gpt-4.1-nano", "gpt-4o-mini", "gpt-4.1", "gpt-4o")
STEMS = [f"regions_citizen_cslegacy_legacy_OpenAI_{model}_Prob_t=0.0_x10" for model in MODELS]
STEMS += [f"regions_directlegacy_cslegacy_OpenAI_{model}_Prob_t=0.0_x10" for model in MODELS]


@pytest.fixture(scope="module")
def store() -> ResultStore:
    found = ResultStore(RESULTS_DIR, "election_2021")
    require(found.paths(STEMS[0]).record)
    return found


@pytest.mark.parametrize("stem", STEMS)
def test_a_table_reads_as_a_run_of_means(stem, store):
    run = load_run(store, stem)
    resolved = resolve(RunSpec.from_dict(run.record["spec"]))
    delivered = run.record[DELIVERED]
    assert resolved.stem == stem and delivered["averaged"] == run.spec.repeats == 10
    assert len(run.results) == len(run.subjects) == 15 and (run.results["repeat"] == 1).all()

    first = resolved.calls[resolved.calls["repeat"] == 1].set_index("id")
    saved = store.frame(stem)
    assert all(saved.at[subject, "Prompt"] == resolved.prompt(first.loc[subject]) for subject in saved.index)

    table = run.topic.rename(pd.read_csv(store.run_dir(stem) / delivered["file"], index_col=0, encoding="utf-8"))
    table = table.set_index(table["region"].map(pd.Series(run.values.index, index=run.values["region"]))).sort_index()
    assert (table[run.topic.columns].to_numpy() == run.results[run.topic.columns].to_numpy()).all()


def test_direct_was_sent_without_the_vocalised_preposition(store):
    sent = store.frame(STEMS[-1])["Prompt"]
    assert sent.str.contains(" sněmovny v ").all() and not sent.str.contains(" ve ").any()
