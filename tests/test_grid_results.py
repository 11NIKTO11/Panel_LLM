"""The grid runs brought in from CoRe/_grid_separate: read like any run, sent what their record says, and the colleague's add up to their report."""
import pytest

from panel.analysis import load_run
from panel.run import ResultStore, resolve, RunSpec
from panel.run.check import check
from panel.run.record import DELIVERED

from tests.conftest import RESULTS_DIR, load_fixture, require

OURS = {
    "grid-sod-retired_default_cs-grid_legacy_OpenRouter_gpt-4o_Prob_t=0.3_x3": 3,
    "grid-sod45-retired_default_cs-grid_legacy_OpenRouter_gpt-4o_Prob_t=0.3": 1,
    "grid-simple_default_cs-grid_legacy_OpenRouter_gpt-4o_Prob_t=0.3": 1,
}
COLLEAGUE = "grid-simple-en_delivered_colleague_OpenRouter_gpt-4o-2024-11-20_Prob_t=0.3_x3"
REPORTED = {"ANO": 30.25, "SPOLU": 23.58, "PirSTAN": 16.72, "SPD": 10.96, "Prisaha": 5.28, "CSSD": 5.01, "KSCM": 3.68, "TSS": 2.30, "Other": 2.22}
REPORTED_TURNOUT = 72.6


@pytest.fixture(scope="module")
def store() -> ResultStore:
    found = ResultStore(RESULTS_DIR, "election_2021")
    require(found.paths(COLLEAGUE).record)
    return found


@pytest.mark.parametrize("stem", sorted(OURS))
def test_ours_resolve_check_and_carry_the_schema_sent(stem, store):
    record = store.record(stem)
    resolved = resolve(RunSpec.from_dict(record["spec"]))
    assert resolved.stem == stem and record["response_schema"] == load_fixture("grid_schema_sent")["VotingProbabilities"]
    assert len(record["migrated"]["repeats"]) == OURS[stem]
    done = check(store, stem, resolved.topic, resolved.framing)
    assert done.same_calls and done.missing == 0 and done.out_of_range == 0
    assert len(load_run(store, stem).results) == OURS[stem] * len(resolved.rows)


def test_colleague_adds_up_to_their_report(store):
    run = load_run(store, COLLEAGUE)
    assert DELIVERED in run.record and run.spec.repeats == 3 and (run.results["repeat"] == 1).all()
    shares = run.results[run.options].mul(run.results["Voted"], axis=0).mean()
    assert (100 * shares / shares.sum()).round(2).to_dict() == REPORTED
    assert round(100 * run.results["Voted"].mean(), 1) == REPORTED_TURNOUT
